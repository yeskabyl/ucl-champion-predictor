"""Monte Carlo simulator for the UEFA Champions League 36-team format (2024/25 onwards).

Official structure implemented (UEFA Regulations 2026/27: Article 18 "equality of points - league phase" and
Article 19 "draw system - knockout phase"; the Annex B bracket *diagrams* could not be read, so the quarter-/semi-final
tree below was derived from Article 19's block structure and then verified against the real 2024/25 and 2025/26 brackets)
--------------------------------------------------------------------------------------
League phase   36 teams, 8 matches each (4 home / 4 away), one table.
               Ranking (Article 18): points, goal difference, goals scored, away goals scored, wins, away wins,
               then the combined points / goal difference / goals scored of the 8 opponents.
               (Next come a disciplinary total and the club coefficient -- neither is in the data, so a
               remaining tie is broken at random, or by ``club_coefficients`` when supplied.)
Positions      1-8 -> round of 16; 9-24 -> knockout play-offs; 25-36 eliminated.
Play-offs      seeded 9/10 v unseeded 23/24, 11/12 v 21/22, 13/14 v 19/20, 15/16 v 17/18 (two legs; the
               seeded club hosts the return leg). The draw decides who meets whom inside each block.
Round of 16    seeds 1/2 face the winners of the 15/16-v-17/18 block, 3/4 the 13/14-v-19/20 block, 5/6 the
               11/12-v-21/22 block, 7/8 the 9/10-v-23/24 block (two legs, seed hosts the return leg).
Bracket        quarter-finals pair block(1/2) with block(7/8) and block(3/4) with block(5/6) *inside the same
               half of the draw*; each semi-final pairs the two quarter-final winners of one half; the final
               pairs the two halves. Seeds in a pair (1/2, 3/4, ...) always sit in opposite halves.
               (Verified against the actual 2024/25 and 2025/26 brackets in ``tests/test_simulator.py``.)
Hosting        play-offs / round of 16: the seeded club hosts the return leg. Quarter-finals: seeds 1-4 (blocks A and B)
               host the return leg; semi-finals: seeds 1-2 (the block A/D branch) do (Article 19.04, 2026/27; the same
               holds for every tie of 2025/26, but NOT for the 2024/25 quarter/semi-finals). A club that beats a seed
               takes over its bracket position.
Final          single match at a neutral venue.
Level ties     after two legs: extra time, then a penalty shoot-out (no away-goals rule).

What is *random* (and therefore simulated): match scores, extra time, penalties, and the draw (which team
goes to which place in the bracket). The draw is sampled uniformly from every outcome consistent with the
pairings that are already known.

Simplifications (also listed in the README and recorded in every output):
* ratings / form are frozen at the prediction cutoff -- they are not updated inside a simulated tournament;
* scores are independent Poisson draws;
* UEFA's association/pot restrictions are not modelled when *generating* a scenario schedule.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .data_loader import OfficialDataUnavailable
from .features import FEATURE_COLUMNS, FeatureBuilder
from .models import OutcomeModel
from .utils import Stage, binomial_std_error, get_logger, team_id_from_name, wilson_interval

log = get_logger("ucl.sim")

# --------------------------------------------------------------------------------------
# Format constants
# --------------------------------------------------------------------------------------
N_TEAMS = 36
MATCHES_PER_TEAM = 8
N_LEAGUE_MATCHES = N_TEAMS * MATCHES_PER_TEAM // 2  # 144
DIRECT_QUALIFIERS = 8
PLAYOFF_LAST_RANK = 24

# Block g = 0..3 ("A".."D"). Ranks are 1-based league-phase positions.
TOP_SEEDS = ((1, 2), (3, 4), (5, 6), (7, 8))
PO_SEEDED = ((15, 16), (13, 14), (11, 12), (9, 10))
PO_UNSEEDED = ((17, 18), (19, 20), (21, 22), (23, 24))

ROUNDS = (Stage.PLAYOFF, Stage.LAST_16, Stage.QUARTER, Stage.SEMI, Stage.FINAL)
STAGE_COLUMNS = ["reach_r16", "reach_quarter_final", "reach_semi_final", "reach_final", "champion"]

MODE_LIVE, MODE_REPLAY, MODE_SCENARIO = "LIVE_FORECAST", "HISTORICAL_REPLAY", "SCENARIO"

STANDING_ASSUMPTIONS = [
    "Ratings and form are frozen at the prediction cutoff and are not updated during a simulated tournament.",
    "Scores are independent Poisson draws; extra time uses one third of the 90-minute rates; shoot-outs are a coin flip.",
    "League-phase ties still level after the opponent-based criteria are broken at random "
    "(disciplinary record and club coefficient are not available).",
    "The knockout draw is sampled uniformly from all outcomes consistent with the pairings already known; "
    "no other draw restrictions are modelled.",
    "Monte Carlo intervals measure simulation noise only, not uncertainty in the match model.",
]


class SetupError(ValueError):
    """The tournament description is incomplete or violates the format."""


class InconsistentBracketError(SetupError):
    """Known knockout pairings cannot be reproduced by the official bracket structure."""


@dataclass
class SimConfig:
    n_sims: int = 10_000
    seed: int = 42
    penalty_home_win_prob: float = 0.5   # P(second-leg host / first-listed team wins a shoot-out)
    extra_time_fraction: float = 1.0 / 3.0
    keep_per_sim: bool = True


# --------------------------------------------------------------------------------------
# League phase table
# --------------------------------------------------------------------------------------
@dataclass
class LeagueTable:
    points: np.ndarray
    goal_diff: np.ndarray
    goals_for: np.ndarray
    away_goals_for: np.ndarray
    wins: np.ndarray
    away_wins: np.ndarray
    opp_points: np.ndarray
    opp_goal_diff: np.ndarray
    opp_goals_for: np.ndarray
    order: np.ndarray   # (S, T): team index finishing r-th (0-based)
    rank: np.ndarray    # (S, T): 1-based final position of each team


def compute_league_table(home_idx: np.ndarray, away_idx: np.ndarray, home_goals: np.ndarray, away_goals: np.ndarray,
                         n_teams: int, *, club_coefficients: np.ndarray | None = None,
                         rng: np.random.Generator | None = None) -> LeagueTable:
    """Standings for ``S`` simulated result sets at once. ``home_goals``/``away_goals`` have shape ``(S, M)``.

    Ranking criteria (Article 18, in order): points, goal difference, goals scored, away goals scored, wins,
    away wins, opponents' combined points, goal difference and goals scored; then (optionally) club
    coefficient; then a random draw. Works for any ``n_teams`` so it can be unit-tested on tiny leagues.
    """
    hg = np.atleast_2d(np.asarray(home_goals, dtype=np.float64))
    ag = np.atleast_2d(np.asarray(away_goals, dtype=np.float64))
    m = len(home_idx)
    if hg.shape[1] != m or ag.shape != hg.shape:
        raise ValueError("goal arrays must have shape (n_sims, n_matches)")
    H = np.zeros((m, n_teams)); H[np.arange(m), home_idx] = 1.0
    A = np.zeros((m, n_teams)); A[np.arange(m), away_idx] = 1.0

    home_win, away_win, draw = (hg > ag).astype(float), (ag > hg).astype(float), (hg == ag).astype(float)
    points = (3 * home_win + draw) @ H + (3 * away_win + draw) @ A
    gf = hg @ H + ag @ A
    ga = ag @ H + hg @ A
    gd = gf - ga
    away_gf = ag @ A
    wins = home_win @ H + away_win @ A
    away_wins = away_win @ A

    meetings = H.T @ A + A.T @ H  # (T, T): how often i and j met
    opp_pts, opp_gd, opp_gf = points @ meetings, gd @ meetings, gf @ meetings

    rng = rng or np.random.default_rng(0)
    keys = [rng.random(points.shape)]
    if club_coefficients is not None:
        keys.append(-np.broadcast_to(np.asarray(club_coefficients, dtype=float), points.shape))
    keys += [-opp_gf, -opp_gd, -opp_pts, -away_wins, -wins, -away_gf, -gf, -gd, -points]  # last key = primary
    order = np.lexsort(np.stack(keys), axis=-1)
    rank = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    rank[rows, order] = np.arange(1, n_teams + 1)[None, :]
    as_int = lambda a: np.rint(a).astype(np.int32)  # noqa: E731
    return LeagueTable(as_int(points), as_int(gd), as_int(gf), as_int(away_gf), as_int(wins), as_int(away_wins),
                       as_int(opp_pts), as_int(opp_gd), as_int(opp_gf), order, rank)


def standings_dataframe(table: LeagueTable, team_names: Sequence[str], sim: int = 0) -> pd.DataFrame:
    """Readable standings of one simulated (or real) result set."""
    order = table.order[sim]
    df = pd.DataFrame({
        "position": np.arange(1, len(order) + 1),
        "team": [team_names[i] for i in order],
        "points": table.points[sim, order], "gd": table.goal_diff[sim, order], "gf": table.goals_for[sim, order],
        "away_gf": table.away_goals_for[sim, order], "wins": table.wins[sim, order],
        "away_wins": table.away_wins[sim, order],
    })
    df["zone"] = np.where(df["position"] <= DIRECT_QUALIFIERS, "round of 16",
                          np.where(df["position"] <= PLAYOFF_LAST_RANK, "play-offs", "eliminated"))
    return df


# --------------------------------------------------------------------------------------
# Scenario schedule generator
# --------------------------------------------------------------------------------------
def generate_league_phase_fixtures(pots: Sequence[Sequence[int]], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random league-phase schedule: every team meets 2 teams from each of the 4 pots, once at home and once away.

    ``pots`` are four lists of nine team indices. Returns ``(home_idx, away_idx)`` with 144 matches. Cross-pot
    pairs use two permutations whose images differ everywhere (so nobody meets a team twice); same-pot pairs use
    a random 9-cycle. **Association restrictions of the real draw are not modelled** -- scenario use only.
    """
    pots = [list(p) for p in pots]
    if len(pots) != 4 or any(len(p) != 9 for p in pots) or len({t for p in pots for t in p}) != 36:
        raise SetupError("pots must be four disjoint lists of 9 team indices")
    home, away = [], []

    def derangement(n: int) -> np.ndarray:
        while True:
            d = rng.permutation(n)
            if not (d == np.arange(n)).any():
                return d

    for a in range(4):
        cycle = rng.permutation(pots[a])
        for i in range(9):  # c_i hosts c_{i+1}: each team hosts one and visits one same-pot opponent
            home.append(cycle[i]); away.append(cycle[(i + 1) % 9])
        for b in range(a + 1, 4):
            sigma = rng.permutation(9)
            tau = sigma[derangement(9)]
            for i in range(9):
                home.append(pots[a][i]); away.append(pots[b][sigma[i]])   # a_i hosts b_sigma(i)
                home.append(pots[b][tau[i]]); away.append(pots[a][i])     # b_tau(i) hosts a_i
    return np.array(home, dtype=int), np.array(away, dtype=int)


def validate_league_fixtures(home_idx: np.ndarray, away_idx: np.ndarray, n_teams: int = N_TEAMS) -> None:
    if len(home_idx) != n_teams * MATCHES_PER_TEAM // 2:
        raise SetupError(f"expected {n_teams * MATCHES_PER_TEAM // 2} league-phase matches, found {len(home_idx)}")
    h = np.bincount(home_idx, minlength=n_teams)
    a = np.bincount(away_idx, minlength=n_teams)
    if not ((h == MATCHES_PER_TEAM // 2).all() and (a == MATCHES_PER_TEAM // 2).all()):
        raise SetupError("every team must play exactly 4 home and 4 away league-phase matches")
    pairs = {frozenset(p) for p in zip(home_idx.tolist(), away_idx.tolist())}
    if len(pairs) != len(home_idx):
        raise SetupError("some pairing occurs more than once in the league phase")
    if (home_idx == away_idx).any():
        raise SetupError("a team is scheduled against itself")


# --------------------------------------------------------------------------------------
# Rates and known results
# --------------------------------------------------------------------------------------
@dataclass
class RateTables:
    """Expected goals for every ordered pairing; ``[i, j]`` = team ``i`` hosting team ``j``.

    ``league_*``/``ko_*`` differ only in the ``is_knockout`` feature; ``neutral[i, j]`` is team ``i``'s expected
    goals against ``j`` at a neutral venue (used for the final).
    """

    league_home: np.ndarray
    league_away: np.ndarray
    ko_home: np.ndarray
    ko_away: np.ndarray
    neutral: np.ndarray

    def __post_init__(self) -> None:
        shapes = {a.shape for a in (self.league_home, self.league_away, self.ko_home, self.ko_away, self.neutral)}
        if len(shapes) != 1 or len(next(iter(shapes))) != 2 or next(iter(shapes))[0] != next(iter(shapes))[1]:
            raise SetupError("rate tables must all be square and equally sized")
        for a in (self.league_home, self.league_away, self.ko_home, self.ko_away, self.neutral):
            if not np.isfinite(a).all() or (a <= 0).any():
                raise SetupError("expected goals must be finite and strictly positive")

    @classmethod
    def constant(cls, n_teams: int, home: float = 1.5, away: float = 1.2) -> "RateTables":
        full = lambda v: np.full((n_teams, n_teams), v)  # noqa: E731
        return cls(full(home), full(away), full(home), full(away), full((home + away) / 2))

    @classmethod
    def from_strengths(cls, strengths: Sequence[float], base_total: float = 2.75, home_edge: float = 0.25) -> "RateTables":
        """Toy generator for tests/demos: ``strengths`` act as log-scale goal-supremacy offsets."""
        s = np.asarray(strengths, dtype=float)
        sup = s[:, None] - s[None, :]
        tot = base_total
        def split(extra: float) -> tuple[np.ndarray, np.ndarray]:
            d = sup + extra
            return np.clip(tot / 2 * np.exp(d / 2), 0.05, None), np.clip(tot / 2 * np.exp(-d / 2), 0.05, None)
        lh, la = split(home_edge)
        nh, _ = split(0.0)
        return cls(lh, la, lh.copy(), la.copy(), nh)


@dataclass
class KnownKnockout:
    """Completed knockout results, indexed by team indices. ``-1`` = unknown."""

    goals: dict[str, np.ndarray]    # round -> (T, T, 2): goals of [home_i, away_j] in the match i hosted j
    winners: dict[str, np.ndarray]  # round -> (T, T): winner of the whole tie between i and j
    pairs: dict[str, list[tuple[int, int]]]  # round -> every tie whose participants are known (played or not)

    @classmethod
    def empty(cls, n_teams: int) -> "KnownKnockout":
        return cls({r: np.full((n_teams, n_teams, 2), -1, dtype=np.int16) for r in ROUNDS},
                   {r: np.full((n_teams, n_teams), -1, dtype=np.int16) for r in ROUNDS}, {r: [] for r in ROUNDS})

    @property
    def n_known_legs(self) -> int:
        return int(sum((g[..., 0] >= 0).sum() for g in self.goals.values()))


@dataclass
class TournamentSetup:
    team_ids: list[str]
    team_names: list[str]
    league_home: np.ndarray        # (144,) team indices
    league_away: np.ndarray
    league_played: np.ndarray      # (144,) bool
    league_home_goals: np.ndarray  # (144,) valid where played
    league_away_goals: np.ndarray
    rates: RateTables
    known_ko: KnownKnockout
    mode: str
    label: str
    cutoff: pd.Timestamp | None = None
    season: int | None = None
    club_coefficients: np.ndarray | None = None
    team_notes: dict[str, str] = field(default_factory=dict)  # team_id -> caveat (e.g. "no match history")
    assumptions: list[str] = field(default_factory=lambda: list(STANDING_ASSUMPTIONS))

    @property
    def n_teams(self) -> int:
        return len(self.team_ids)

    def validate(self) -> None:
        if self.n_teams != N_TEAMS:
            raise SetupError(f"the 2024+ format needs exactly {N_TEAMS} teams, got {self.n_teams}")
        validate_league_fixtures(self.league_home, self.league_away, self.n_teams)
        if self.rates.league_home.shape[0] != self.n_teams:
            raise SetupError("rate tables do not match the number of teams")
        played = self.league_played
        if np.any(self.league_home_goals[played] < 0) or np.any(self.league_away_goals[played] < 0):
            raise SetupError("played league matches need non-negative scores")
        if self.mode not in (MODE_LIVE, MODE_REPLAY, MODE_SCENARIO):
            raise SetupError(f"unknown mode {self.mode!r}")
        league_done = bool(played.all())
        if not league_done and any(len(p) for p in self.known_ko.pairs.values()):
            raise InconsistentBracketError("knockout pairings are known although the league phase is not finished")

    @property
    def league_phase_complete(self) -> bool:
        return bool(self.league_played.all())


# --------------------------------------------------------------------------------------
# Bracket / draw space
# --------------------------------------------------------------------------------------
def _locate(rank: int, m: np.ndarray, t: np.ndarray, s: np.ndarray) -> tuple[int, np.ndarray]:
    """(block, half) of the round-of-16 slot a team occupies, for every candidate draw (arrays over candidates)."""
    if rank <= DIRECT_QUALIFIERS:
        g, j = divmod(rank - 1, 2)
        return g, j ^ s[:, g]
    for g in range(4):
        if rank in PO_SEEDED[g]:
            return g, PO_SEEDED[g].index(rank) ^ t[:, g]
        if rank in PO_UNSEEDED[g]:
            u = PO_UNSEEDED[g].index(rank)
            return g, (u ^ m[:, g]) ^ t[:, g]
    raise ValueError(f"rank {rank} does not take part in the knockout phase")


def candidate_draws() -> np.ndarray:
    """All 4096 draw outcomes as bit rows ``[m(4) | t(4) | s(4)]``.

    m_g: which unseeded club each seeded play-off club meets; t_g: which round-of-16 half each of the two play-off
    ties of block g feeds; s_g: which half each of the two top-8 seeds of block g is placed in.
    """
    return np.array(list(product((0, 1), repeat=12)), dtype=np.int8)


def consistent_draws(known_pairs: Mapping[str, Sequence[tuple[int, int]]], rank_of_team: np.ndarray | None) -> np.ndarray:
    """Draw outcomes compatible with every knockout pairing that is already known.

    ``rank_of_team[i]`` is team ``i``'s final league-phase position (only available when the league phase is over).
    Raises :class:`InconsistentBracketError` if no draw explains the data (e.g. a wrong bracket assumption).
    """
    cand = candidate_draws()
    if not any(len(v) for v in known_pairs.values()):
        return cand
    if rank_of_team is None:
        raise InconsistentBracketError("known knockout pairings need the final league-phase ranking")
    m, t, s = cand[:, 0:4].astype(int), cand[:, 4:8].astype(int), cand[:, 8:12].astype(int)
    ok = np.ones(len(cand), dtype=bool)

    def loc(team: int):
        r = int(rank_of_team[team])
        if r > PLAYOFF_LAST_RANK:
            raise InconsistentBracketError(f"team index {team} finished {r}th in the league phase yet appears in the knockout phase")
        return _locate(r, m, t, s)

    for x, y in known_pairs.get(Stage.PLAYOFF, []):
        rx, ry = int(rank_of_team[x]), int(rank_of_team[y])
        if ry < rx:
            x, y, rx, ry = y, x, ry, rx
        gx = next((g for g in range(4) if rx in PO_SEEDED[g]), None)
        gy = next((g for g in range(4) if ry in PO_UNSEEDED[g]), None)
        if gx is None or gx != gy:
            raise InconsistentBracketError(f"play-off tie between league positions {rx} and {ry} is not an official pairing")
        k, u = PO_SEEDED[gx].index(rx), PO_UNSEEDED[gy].index(ry)
        ok &= (k ^ m[:, gx]) == u
    for x, y in known_pairs.get(Stage.LAST_16, []):
        (gx, hx), (gy, hy) = loc(x), loc(y)
        if gx != gy:
            raise InconsistentBracketError("round-of-16 tie pairs teams from different bracket blocks")
        ok &= hx == hy
    for x, y in known_pairs.get(Stage.QUARTER, []):
        (gx, hx), (gy, hy) = loc(x), loc(y)
        if {gx, gy} not in ({0, 3}, {1, 2}):
            raise InconsistentBracketError("quarter-final tie does not match the official bracket blocks")
        ok &= hx == hy
    for x, y in known_pairs.get(Stage.SEMI, []):
        (gx, hx), (gy, hy) = loc(x), loc(y)
        if not ((gx in (0, 3)) != (gy in (0, 3))):
            raise InconsistentBracketError("semi-final tie does not match the official bracket blocks")
        ok &= hx == hy
    for x, y in known_pairs.get(Stage.FINAL, []):
        (_, hx), (_, hy) = loc(x), loc(y)
        ok &= hx != hy
    if not ok.any():
        raise InconsistentBracketError(
            "no knockout draw outcome reproduces the known pairings -- the data contradicts the bracket structure")
    return cand[ok]


# --------------------------------------------------------------------------------------
# Tie simulation
# --------------------------------------------------------------------------------------
def _known(arr: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return arr[a, b]


def _two_legged_tie(rng: np.random.Generator, setup: TournamentSetup, rnd: str, first_home: np.ndarray,
                    second_home: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Winner of each two-legged tie. ``first_home`` hosts leg 1, ``second_home`` hosts leg 2 (arrays ``(S, K)``).

    Completed legs are never re-drawn; a completed tie keeps its real winner.
    """
    r, ko = setup.rates, setup.known_ko
    lam_h1, lam_a1 = r.ko_home[first_home, second_home], r.ko_away[first_home, second_home]
    lam_h2, lam_a2 = r.ko_home[second_home, first_home], r.ko_away[second_home, first_home]
    g1h, g1a = rng.poisson(lam_h1), rng.poisson(lam_a1)       # leg 1: first_home v second_home
    g2h, g2a = rng.poisson(lam_h2), rng.poisson(lam_a2)       # leg 2: second_home v first_home
    k1, k2 = ko.goals[rnd][first_home, second_home], ko.goals[rnd][second_home, first_home]
    g1h, g1a = np.where(k1[..., 0] >= 0, k1[..., 0], g1h), np.where(k1[..., 0] >= 0, k1[..., 1], g1a)
    g2h, g2a = np.where(k2[..., 0] >= 0, k2[..., 0], g2h), np.where(k2[..., 0] >= 0, k2[..., 1], g2a)

    agg_first, agg_second = g1h + g2a, g1a + g2h
    level = agg_first == agg_second
    if level.any():
        f = cfg.extra_time_fraction  # extra time is played at the second leg's ground
        et_second = rng.poisson(lam_h2 * f)
        et_first = rng.poisson(lam_a2 * f)
        agg_first, agg_second = agg_first + et_first, agg_second + et_second
        still = agg_first == agg_second
        pens_second = rng.random(agg_first.shape) < cfg.penalty_home_win_prob
        second_wins = np.where(still, pens_second, agg_second > agg_first)
    else:
        second_wins = agg_second > agg_first
    winner = np.where(second_wins, second_home, first_home)
    real = ko.winners[rnd][first_home, second_home]
    return np.where(real >= 0, real, winner)


def _final_match(rng: np.random.Generator, setup: TournamentSetup, a: np.ndarray, b: np.ndarray, cfg: SimConfig) -> np.ndarray:
    lam_a, lam_b = setup.rates.neutral[a, b], setup.rates.neutral[b, a]
    ga, gb = rng.poisson(lam_a), rng.poisson(lam_b)
    level = ga == gb
    if level.any():
        ga = ga + rng.poisson(lam_a * cfg.extra_time_fraction)
        gb = gb + rng.poisson(lam_b * cfg.extra_time_fraction)
        a_wins = np.where(ga == gb, rng.random(ga.shape) < 0.5, ga > gb)
    else:
        a_wins = ga > gb
    winner = np.where(a_wins, a, b)
    real = setup.known_ko.winners[Stage.FINAL][a, b]
    return np.where(real >= 0, real, winner)


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------
@dataclass
class SimulationResult:
    team_ids: list[str]
    team_names: list[str]
    n_sims: int
    seed: int
    mode: str
    label: str
    position_counts: np.ndarray                 # (T, 36)
    stage_counts: dict[str, np.ndarray]         # STAGE_COLUMNS -> (T,)
    playoff_counts: np.ndarray                  # (T,) finished 9..24
    champion_per_sim: np.ndarray | None = None  # (S,)
    rank_per_sim: np.ndarray | None = None      # (S, T)
    assumptions: list[str] = field(default_factory=list)
    n_draw_outcomes: int = 4096
    notes: dict = field(default_factory=dict)

    def champion_table(self, z: float = 1.96) -> pd.DataFrame:
        k = self.stage_counts["champion"]
        lo, hi = wilson_interval(k, self.n_sims, z)
        df = pd.DataFrame({
            "team_id": self.team_ids, "team_name": self.team_names,
            "simulated_championships": k.astype(int),
            "champion_probability": k / self.n_sims,
            "mc_std_error": binomial_std_error(k, self.n_sims),
            "ci95_low": lo, "ci95_high": hi,
        })
        df = df.sort_values(["simulated_championships", "team_name"], ascending=[False, True]).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
        df.insert(0, "forecast_mode", self.mode)
        return df

    def qualification_table(self) -> pd.DataFrame:
        pc, n = self.position_counts, self.n_sims
        top8 = pc[:, :DIRECT_QUALIFIERS].sum(1)
        po = pc[:, DIRECT_QUALIFIERS:PLAYOFF_LAST_RANK].sum(1)
        out = pd.DataFrame({
            "forecast_mode": self.mode, "team_id": self.team_ids, "team_name": self.team_names,
            "p_top8_direct_r16": top8 / n, "p_playoff": po / n, "p_eliminated": pc[:, PLAYOFF_LAST_RANK:].sum(1) / n,
            "p_reach_r16": self.stage_counts["reach_r16"] / n,
            "p_reach_quarter_final": self.stage_counts["reach_quarter_final"] / n,
            "p_reach_semi_final": self.stage_counts["reach_semi_final"] / n,
            "p_reach_final": self.stage_counts["reach_final"] / n,
            "p_champion": self.stage_counts["champion"] / n,
        })
        return out.sort_values("p_champion", ascending=False).reset_index(drop=True)

    def final_standings_table(self) -> pd.DataFrame:
        probs = self.position_counts / self.n_sims
        cols = {f"p_pos_{i + 1}": probs[:, i] for i in range(probs.shape[1])}
        df = pd.DataFrame({"forecast_mode": self.mode, "team_id": self.team_ids, "team_name": self.team_names,
                           "expected_position": probs @ np.arange(1, probs.shape[1] + 1), **cols})
        return df.sort_values("expected_position").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# The simulation
# --------------------------------------------------------------------------------------
def simulate_tournament(setup: TournamentSetup, config: SimConfig | None = None) -> SimulationResult:
    cfg = config or SimConfig()
    if cfg.n_sims < 1:
        raise ValueError("n_sims must be >= 1")
    setup.validate()
    rng = np.random.default_rng(cfg.seed)
    S, T = cfg.n_sims, setup.n_teams

    # ---- 1. league phase: keep every completed result, simulate only the rest ----------
    M = len(setup.league_home)
    hg = np.empty((S, M)); ag = np.empty((S, M))
    played = setup.league_played
    hg[:, played], ag[:, played] = setup.league_home_goals[played], setup.league_away_goals[played]
    todo = ~played
    if todo.any():
        h, a = setup.league_home[todo], setup.league_away[todo]
        hg[:, todo] = rng.poisson(setup.rates.league_home[h, a], size=(S, int(todo.sum())))
        ag[:, todo] = rng.poisson(setup.rates.league_away[h, a], size=(S, int(todo.sum())))
    if setup.league_phase_complete:  # nothing is random: compute once, tile
        table = compute_league_table(setup.league_home, setup.league_away, hg[:1], ag[:1], T,
                                     club_coefficients=setup.club_coefficients, rng=rng)
        order, rank = np.tile(table.order, (S, 1)), np.tile(table.rank, (S, 1))
    else:
        table = compute_league_table(setup.league_home, setup.league_away, hg, ag, T,
                                     club_coefficients=setup.club_coefficients, rng=rng)
        order, rank = table.order, table.rank

    # ---- 2. knockout draw, conditioned on whatever is already known -------------------
    rank_static = rank[0] if setup.league_phase_complete else None
    cands = consistent_draws(setup.known_ko.pairs, rank_static)
    bits = cands[rng.integers(0, len(cands), size=S)].astype(int)
    m, t, s = bits[:, 0:4], bits[:, 4:8], bits[:, 8:12]
    ar = np.arange(S)

    def at_rank(rank_arr: np.ndarray) -> np.ndarray:  # team index at 1-based league positions (broadcast over S)
        return order[ar, rank_arr - 1]

    # ---- 3. play-offs ------------------------------------------------------------------
    po_by_half = np.zeros((S, 4, 2), dtype=np.int64)
    for g in range(4):
        first, second = [], []
        for k in range(2):
            seeded = at_rank(np.full(S, PO_SEEDED[g][k]))
            u_rank = np.where((k ^ m[:, g]) == 0, PO_UNSEEDED[g][0], PO_UNSEEDED[g][1])
            unseeded = at_rank(u_rank)
            first.append(unseeded); second.append(seeded)   # seeded club hosts the return leg
        win = _two_legged_tie(rng, setup, Stage.PLAYOFF, np.stack(first, 1), np.stack(second, 1), cfg)  # (S, 2)
        for k in range(2):
            po_by_half[ar, g, k ^ t[:, g]] = win[:, k]

    # ---- 4. round of 16 ------------------------------------------------------------------
    r16_win = np.zeros((S, 4, 2), dtype=np.int64)
    reach_r16 = np.zeros((S, T), dtype=bool)
    for g in range(4):
        first, second = [], []
        for h in range(2):
            j = h ^ s[:, g]
            top = at_rank(np.where(j == 0, TOP_SEEDS[g][0], TOP_SEEDS[g][1]))
            first.append(po_by_half[:, g, h]); second.append(top)
            reach_r16[ar, top] = True
            reach_r16[ar, po_by_half[:, g, h]] = True
        win = _two_legged_tie(rng, setup, Stage.LAST_16, np.stack(first, 1), np.stack(second, 1), cfg)  # (S, 2)
        r16_win[:, g, :] = win

    # ---- 5. quarter-finals (block A v D, B v C inside each half) -----------------------
    qf_ad = np.zeros((S, 2), dtype=np.int64)
    qf_bc = np.zeros((S, 2), dtype=np.int64)
    reach_qf = np.zeros((S, T), dtype=bool)
    for g in range(4):
        for h in range(2):
            reach_qf[ar, r16_win[:, g, h]] = True
    for h in range(2):
        a, d, b, c = r16_win[:, 0, h], r16_win[:, 3, h], r16_win[:, 1, h], r16_win[:, 2, h]
        qf_ad[:, h] = _two_legged_tie(rng, setup, Stage.QUARTER, d[:, None], a[:, None], cfg)[:, 0]   # A-side hosts return
        qf_bc[:, h] = _two_legged_tie(rng, setup, Stage.QUARTER, c[:, None], b[:, None], cfg)[:, 0]   # B-side hosts return

    # ---- 6. semi-finals and final --------------------------------------------------------
    reach_sf = np.zeros((S, T), dtype=bool)
    sf_win = np.zeros((S, 2), dtype=np.int64)
    for h in range(2):
        reach_sf[ar, qf_ad[:, h]] = True
        reach_sf[ar, qf_bc[:, h]] = True
        sf_win[:, h] = _two_legged_tie(rng, setup, Stage.SEMI, qf_bc[:, h][:, None], qf_ad[:, h][:, None], cfg)[:, 0]
    reach_final = np.zeros((S, T), dtype=bool)
    reach_final[ar, sf_win[:, 0]] = True
    reach_final[ar, sf_win[:, 1]] = True
    champion = _final_match(rng, setup, sf_win[:, 0], sf_win[:, 1], cfg)

    # ---- 7. aggregate --------------------------------------------------------------------
    pos_counts = np.stack([np.bincount(rank[:, i] - 1, minlength=T) for i in range(T)])
    champ_mask = np.zeros((S, T), dtype=bool); champ_mask[ar, champion] = True
    stage_counts = {
        "reach_r16": reach_r16.sum(0), "reach_quarter_final": reach_qf.sum(0),
        "reach_semi_final": reach_sf.sum(0), "reach_final": reach_final.sum(0), "champion": champ_mask.sum(0),
    }
    playoff_counts = pos_counts[:, DIRECT_QUALIFIERS:PLAYOFF_LAST_RANK].sum(1)
    return SimulationResult(
        setup.team_ids, setup.team_names, S, cfg.seed, setup.mode, setup.label, pos_counts, stage_counts, playoff_counts,
        champion if cfg.keep_per_sim else None, rank if cfg.keep_per_sim else None, list(setup.assumptions),
        n_draw_outcomes=len(cands),
        notes={"league_matches_known": int(played.sum()), "league_matches_simulated": int(todo.sum()),
               "known_knockout_legs": setup.known_ko.n_known_legs, "team_notes": setup.team_notes},
    )


# --------------------------------------------------------------------------------------
# Building a TournamentSetup from data
# --------------------------------------------------------------------------------------
def build_rate_tables(model: OutcomeModel, builder: FeatureBuilder, team_ids: Sequence[str], season: int) -> RateTables:
    """Expected goals for every ordered pairing, from the fitted model and the team state frozen in ``builder``.

    Rest days are blanked (unknown for unscheduled ties) so the model's imputer uses its *training* median.
    """
    n = len(team_ids)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    ii, jj = (np.array(v) for v in zip(*pairs))

    def predict(stage: str, neutral: bool) -> tuple[np.ndarray, np.ndarray]:
        recs = [builder.features_for(team_ids[i], team_ids[j], None, neutral=neutral, stage=stage, season=season,
                                     rest_known=False) for i, j in pairs]
        df = pd.DataFrame(recs, columns=FEATURE_COLUMNS)
        lam_h, lam_a = model.predict_goal_rates(df)
        out_h, out_a = np.ones((n, n)), np.ones((n, n))
        out_h[ii, jj], out_a[ii, jj] = lam_h, lam_a
        return out_h, out_a

    lg_h, lg_a = predict(Stage.LEAGUE, False)
    ko_h, ko_a = predict(Stage.LAST_16, False)
    nt_h, nt_a = predict(Stage.FINAL, True)
    neutral = 0.5 * (nt_h + nt_a.T)  # symmetrise: i as nominal home v j, and j as nominal home v i
    np.fill_diagonal(neutral, 1.0)
    return RateTables(lg_h, lg_a, ko_h, ko_a, neutral)


def derive_known_knockout(matches: pd.DataFrame, team_index: Mapping[str, int]) -> KnownKnockout:
    """Completed knockout legs and ties from the (as-of-cutoff) match table.

    A tie is *complete* when all its legs (2, or 1 for the final) are completed; its winner is then taken from
    the real aggregate including extra time (and, if still level, the real shoot-out) and is never re-simulated.
    """
    n = len(team_index)
    ko = KnownKnockout.empty(n)
    sub = matches[matches["stage"].isin(ROUNDS) & matches["home_team_id"].notna() & matches["away_team_id"].notna()]
    for rnd in ROUNDS:
        ties: dict[frozenset, list] = defaultdict(list)
        for r in sub[sub["stage"] == rnd].itertuples(index=False):
            if r.home_team_id not in team_index or r.away_team_id not in team_index:
                raise SetupError(f"{rnd} match {r.match_id} involves a team that is not in the league phase")
            ties[frozenset((team_index[r.home_team_id], team_index[r.away_team_id]))].append(r)
        expected_legs = 1 if rnd == Stage.FINAL else 2
        for key, legs in ties.items():
            a, b = sorted(key)
            ko.pairs[rnd].append((a, b))
            done = [r for r in legs if r.is_completed]
            for r in done:
                hi, ai = team_index[r.home_team_id], team_index[r.away_team_id]
                ko.goals[rnd][hi, ai] = (int(r.home_goals_90), int(r.away_goals_90))
            if len(done) >= expected_legs:
                goals = {a: 0.0, b: 0.0}
                for r in done:
                    goals[team_index[r.home_team_id]] += r.home_goals_aet
                    goals[team_index[r.away_team_id]] += r.away_goals_aet
                if goals[a] != goals[b]:
                    w = a if goals[a] > goals[b] else b
                else:
                    last = max(done, key=lambda r: r.utc_date)
                    if last.winner == "HOME_TEAM":
                        w = team_index[last.home_team_id]
                    elif last.winner == "AWAY_TEAM":
                        w = team_index[last.away_team_id]
                    else:
                        log.warning("completed %s tie has a level aggregate and no recorded winner; it will be re-simulated", rnd)
                        continue
                ko.winners[rnd][a, b] = ko.winners[rnd][b, a] = w
    return ko


def _team_notes(team_ids: Sequence[str], builder: FeatureBuilder) -> dict[str, str]:
    known = builder.known_teams()
    return {t: "no earlier Champions League match in the data: default Elo rating, form unknown"
            for t in team_ids if t not in known}


def build_setup_from_matches(matches: pd.DataFrame, season: int, model: OutcomeModel, builder: FeatureBuilder, *,
                             mode: str, label: str, cutoff: pd.Timestamp | None = None,
                             club_coefficients: Mapping[str, float] | None = None) -> TournamentSetup:
    """TournamentSetup for ``season`` from a match table already restricted to what was known at ``cutoff``
    (see ``data_loader.as_of``). Raises :class:`OfficialDataUnavailable` when the 144 league-phase fixtures
    and 36 participants are not all present -- nothing is guessed."""
    sm = matches[matches["season"] == season]
    league = sm[sm["stage"] == Stage.LEAGUE].dropna(subset=["home_team_id", "away_team_id"])
    if len(league) != N_LEAGUE_MATCHES:
        raise OfficialDataUnavailable(
            f"season {season}/{str(season + 1)[-2:]}: found {len(league)} league-phase fixtures with known teams, "
            f"expected {N_LEAGUE_MATCHES}. The official draw/fixture list is not in the data (yet). "
            f"Use scenario mode with explicit participants, or a historical replay.")
    team_ids = sorted(set(league["home_team_id"]) | set(league["away_team_id"]))
    if len(team_ids) != N_TEAMS:
        raise OfficialDataUnavailable(f"found {len(team_ids)} distinct teams in the league phase, expected {N_TEAMS}")
    idx = {t: i for i, t in enumerate(team_ids)}
    names = {}
    for r in league.sort_values("utc_date").itertuples(index=False):
        names[r.home_team_id], names[r.away_team_id] = r.home_team, r.away_team
    played = league["is_completed"].to_numpy(bool)
    coeffs = None
    if club_coefficients is not None:
        coeffs = np.array([club_coefficients.get(t, 0.0) for t in team_ids], dtype=float)
    setup = TournamentSetup(
        team_ids=team_ids, team_names=[names[t] for t in team_ids],
        league_home=league["home_team_id"].map(idx).to_numpy(int),
        league_away=league["away_team_id"].map(idx).to_numpy(int),
        league_played=played,
        league_home_goals=np.where(played, league["home_goals_90"].fillna(0), 0).astype(int),
        league_away_goals=np.where(played, league["away_goals_90"].fillna(0), 0).astype(int),
        rates=build_rate_tables(model, builder, team_ids, season),
        known_ko=derive_known_knockout(sm, idx), mode=mode, label=label, cutoff=cutoff, season=season,
        club_coefficients=coeffs, team_notes=_team_notes(team_ids, builder),
    )
    setup.validate()
    return setup


def build_scenario_setup(pots: Sequence[Sequence[str]], model: OutcomeModel, builder: FeatureBuilder, *, season: int,
                         seed: int = 0,
                         label: str = "SCENARIO (assumed participants and random league-phase schedule)") -> TournamentSetup:
    """What-if tournament for participants the user specifies: four pots of nine club names.

    Everything about the league-phase schedule is an *assumption* (a random pot-respecting schedule). Nothing here is
    an official fixture, and the output is labelled ``SCENARIO``.
    """
    ids = [[team_id_from_name(n) for n in pot] for pot in pots]
    flat = [t for pot in ids for t in pot]
    if len(pots) != 4 or any(len(p) != 9 for p in pots) or len(set(flat)) != 36:
        raise SetupError("scenario needs four pots of nine distinct clubs")
    idx = {t: i for i, t in enumerate(flat)}
    names = [n for pot in pots for n in pot]
    home, away = generate_league_phase_fixtures([[idx[t] for t in pot] for pot in ids], np.random.default_rng(seed))
    m = len(home)
    setup = TournamentSetup(
        team_ids=flat, team_names=names, league_home=home, league_away=away, league_played=np.zeros(m, bool),
        league_home_goals=np.zeros(m, int), league_away_goals=np.zeros(m, int),
        rates=build_rate_tables(model, builder, flat, season), known_ko=KnownKnockout.empty(36),
        mode=MODE_SCENARIO, label=label, season=season, team_notes=_team_notes(flat, builder),
        assumptions=list(STANDING_ASSUMPTIONS) + [
            "Participants, pots and the whole league-phase schedule are user-supplied / randomly generated assumptions, "
            "not official UEFA data."],
    )
    setup.validate()
    return setup
