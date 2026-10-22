"""Standings, bracket, ties, completed-result preservation, reproducibility -- and real-season replays.

Synthetic data is used only to isolate a rule (a tie-breaker, a tie, a bracket constraint); the end-to-end tests at the
bottom use the real, unmodified 2024/25 and 2025/26 seasons from ``tests/fixtures`` (CC0).
"""
import time

import numpy as np
import pandas as pd
import pytest

from src import simulator as sim
from src.data_loader import OfficialDataUnavailable, as_of
from src.features import FeatureBuilder
from src.simulator import (
    InconsistentBracketError, KnownKnockout, RateTables, SetupError, SimConfig, TournamentSetup, candidate_draws,
    compute_league_table, consistent_draws, derive_known_knockout, generate_league_phase_fixtures, simulate_tournament,
    standings_dataframe, validate_league_fixtures,
)
from src.utils import Stage, binomial_std_error, wilson_interval

from .helpers import ConstantRatesModel, real_season


# ======================================================================================
# helpers
# ======================================================================================
def table_for(names, results, coefficients=None, seed=0):
    idx = {n: i for i, n in enumerate(names)}
    h = np.array([idx[r[0]] for r in results])
    a = np.array([idx[r[1]] for r in results])
    hg = np.array([[r[2] for r in results]])
    ag = np.array([[r[3] for r in results]])
    tab = compute_league_table(h, a, hg, ag, len(names), club_coefficients=coefficients, rng=np.random.default_rng(seed))
    return tab, idx


def position(tab, idx, name):
    return int(tab.rank[0, idx[name]])


def toy_setup(strengths=None, seed=0, rates=None, mode=sim.MODE_SCENARIO) -> TournamentSetup:
    ids = [f"t{i:02d}" for i in range(36)]
    pots = [list(range(9 * k, 9 * k + 9)) for k in range(4)]
    home, away = generate_league_phase_fixtures(pots, np.random.default_rng(seed))
    if rates is None:
        rates = RateTables.from_strengths(strengths) if strengths is not None else RateTables.constant(36)
    m = len(home)
    return TournamentSetup(ids, [i.upper() for i in ids], home, away, np.zeros(m, bool), np.zeros(m, int), np.zeros(m, int),
                           rates, KnownKnockout.empty(36), mode, "toy")


def real_setup(season, cutoff, rates_model=None):
    full = real_season(season)
    visible = as_of(full, cutoff, mask_undetermined_knockout=True)
    return sim.build_setup_from_matches(visible, season, rates_model or ConstantRatesModel(), FeatureBuilder(),
                                        mode=sim.MODE_REPLAY, label="replay", cutoff=pd.Timestamp(cutoff))


# ======================================================================================
# league phase: points, goal difference, ranking criteria
# ======================================================================================
def test_points_goal_difference_and_goals_for_against():
    names = ["A", "B", "C"]
    tab, idx = table_for(names, [("A", "B", 2, 0), ("B", "C", 1, 1), ("C", "A", 0, 3)])
    a, b, c = idx["A"], idx["B"], idx["C"]
    assert tab.points[0, [a, b, c]].tolist() == [6, 1, 1]
    assert tab.goal_diff[0, [a, b, c]].tolist() == [5, -2, -3]   # B: 1 for, 2+1 against; C: 1 for, 1+3 against
    assert tab.goals_for[0, [a, b, c]].tolist() == [5, 1, 1]
    assert tab.wins[0, [a, b, c]].tolist() == [2, 0, 0] and tab.away_wins[0, [a, b, c]].tolist() == [1, 0, 0]
    assert tab.away_goals_for[0, [a, b, c]].tolist() == [3, 0, 1]   # A scored 3 at C; C scored 1 at B
    assert tab.order[0][0] == a and set(tab.order[0][1:].tolist()) == {b, c}  # A first; B and C are level on 1 point...
    assert tab.rank[0, b] < tab.rank[0, c]                                    # ...and B's better goal difference separates them


# Each case returns results where team `w` must outrank `l` ONLY because of the criterion under test
# (everything ranked above it in Article 18 is level). X, Y, Z... are neutral opponents.
def case_goal_difference(w, l):
    return [(w, "X", 2, 0), (l, "Y", 1, 0)]


def case_goals_scored(w, l):
    return [(w, "X", 3, 2), (l, "Y", 1, 0)]                                  # both +1 GD


def case_away_goals(w, l):
    return [("X", w, 1, 2), (l, "Y", 2, 1)]                                  # both 3 pts, +1 GD, 2 GF; w scored them away


def case_wins(w, l):
    return [(w, "X", 1, 0), ("Y", w, 1, 0),                                  # w: 1 win 1 loss -> 3 pts, GD 0, GF 1
            (l, "X", 1, 1), (l, "Y", 0, 0), ("Z", l, 0, 0)]                  # l: 3 draws     -> 3 pts, GD 0, GF 1


def case_away_wins(w, l):
    return [("X", w, 0, 1), (w, "Y", 1, 2),                                  # w: away win + home loss
            ("X", l, 2, 1), (l, "Y", 1, 0)]                                  # l: away loss + home win; same pts/GD/GF/away GF/wins


def case_opponents_points(w, l):
    return [(w, "C", 0, 0), (l, "D", 0, 0), ("C", "E", 1, 0), ("D", "F", 0, 1)]  # w's opponent C: 4 pts, l's D: 1 pt


def case_opponents_goal_difference(w, l):
    return [(w, "C", 0, 0), (l, "D", 0, 0), ("C", "E", 3, 0), ("D", "F", 1, 0)]  # C, D both 4 pts; GD +3 v +1


def case_opponents_goals_scored(w, l):
    return [(w, "C", 0, 0), (l, "D", 0, 0), ("C", "E", 2, 1), ("D", "F", 1, 0)]  # C, D both 4 pts, +1 GD; 2 v 1 goals


CRITERIA = [case_goal_difference, case_goals_scored, case_away_goals, case_wins, case_away_wins,
            case_opponents_points, case_opponents_goal_difference, case_opponents_goals_scored]
NAMES = ["A", "B", "X", "Y", "Z", "C", "D", "E", "F"]


@pytest.mark.parametrize("case", CRITERIA, ids=lambda f: f.__name__.replace("case_", ""))
@pytest.mark.parametrize("winner,loser", [("A", "B"), ("B", "A")])  # both ways round: no dependence on team order
def test_each_tiebreaker_decides_when_everything_above_it_is_level(case, winner, loser):
    tab, idx = table_for(NAMES, case(winner, loser))
    w, l = idx[winner], idx[loser]
    # the criteria that come BEFORE the one under test really are level:
    keys = ["points", "goal_diff", "goals_for", "away_goals_for", "wins", "away_wins", "opp_points", "opp_goal_diff", "opp_goals_for"]
    order_of_case = {"case_goal_difference": 1, "case_goals_scored": 2, "case_away_goals": 3, "case_wins": 4,
                     "case_away_wins": 5, "case_opponents_points": 6, "case_opponents_goal_difference": 7,
                     "case_opponents_goals_scored": 8}[case.__name__]
    for key in keys[:order_of_case]:
        assert getattr(tab, key)[0, w] == getattr(tab, key)[0, l], f"{key} should be level"
    assert getattr(tab, keys[order_of_case])[0, w] > getattr(tab, keys[order_of_case])[0, l], "criterion must separate them"
    assert position(tab, idx, winner) < position(tab, idx, loser)


def test_lower_criteria_never_override_higher_ones():
    # A has the better opponents (criterion 6) but the worse goal difference (criterion 2): goal difference wins.
    results = [("A", "C", 1, 0), ("B", "Y", 5, 0), ("C", "E", 1, 0), ("Y", "F", 0, 1)]
    tab, idx = table_for(NAMES, results)
    assert tab.points[0, idx["A"]] == tab.points[0, idx["B"]] == 3
    assert tab.opp_points[0, idx["A"]] != tab.opp_points[0, idx["B"]]
    assert position(tab, idx, "B") < position(tab, idx, "A")


def test_exact_ties_use_the_club_coefficient_when_given_and_are_random_otherwise():
    names = ["A", "B", "X", "Y"]
    results = [("A", "X", 0, 0), ("B", "Y", 0, 0)]
    tab, idx = table_for(names, results, coefficients=np.array([10.0, 5.0, 0.0, 0.0]))
    assert position(tab, idx, "A") < position(tab, idx, "B")
    tab, idx = table_for(names, results, coefficients=np.array([5.0, 10.0, 0.0, 0.0]))
    assert position(tab, idx, "B") < position(tab, idx, "A")
    first = {position(table_for(names, results, seed=s)[0], idx, "A") < position(table_for(names, results, seed=s)[0], idx, "B")
             for s in range(30)}
    assert first == {True, False}  # unresolved ties are broken at random...
    a = table_for(names, results, seed=3)[0]
    b = table_for(names, results, seed=3)[0]
    np.testing.assert_array_equal(a.order, b.order)  # ...but reproducibly


def _reference_order(home, away, hg, ag, n, coef):
    """Slow, obviously-correct implementation of Article 18 written independently of the vectorised one."""
    pts, gf, ga, agf, win, awin = ([0] * n for _ in range(6))
    opponents = [[] for _ in range(n)]
    for h, a, x, y in zip(home, away, hg, ag):
        gf[h] += x; ga[h] += y; gf[a] += y; ga[a] += x; agf[a] += y
        opponents[h].append(a); opponents[a].append(h)
        if x > y:
            pts[h] += 3; win[h] += 1
        elif x < y:
            pts[a] += 3; win[a] += 1; awin[a] += 1
        else:
            pts[h] += 1; pts[a] += 1
    gd = [gf[t] - ga[t] for t in range(n)]
    key = lambda t: (-pts[t], -gd[t], -gf[t], -agf[t], -win[t], -awin[t],  # noqa: E731
                     -sum(pts[o] for o in opponents[t]), -sum(gd[o] for o in opponents[t]), -sum(gf[o] for o in opponents[t]),
                     -coef[t])
    return sorted(range(n), key=key)


def test_vectorised_standings_match_an_independent_reference_on_random_leagues():
    rng = np.random.default_rng(123)
    pots = [list(range(9 * k, 9 * k + 9)) for k in range(4)]
    n_ties = 0
    for trial in range(6):
        home, away = generate_league_phase_fixtures(pots, np.random.default_rng(trial))
        S = 40
        hg = rng.poisson(0.9, size=(S, len(home)))  # low scoring => plenty of ties on the early criteria
        ag = rng.poisson(0.7, size=(S, len(home)))
        coef = rng.permutation(36).astype(float)
        tab = compute_league_table(home, away, hg, ag, 36, club_coefficients=coef, rng=rng)
        for s in range(S):
            ref = _reference_order(home, away, hg[s], ag[s], 36, coef)
            assert tab.order[s].tolist() == ref, f"trial {trial} sim {s}"
            assert tab.rank[s, tab.order[s]].tolist() == list(range(1, 37))
            n_ties += int(len(set(tab.points[s])) < 36)
    assert n_ties > 100  # the comparison really exercised tie-breaking


def test_zones_are_8_16_12_and_table_is_readable():
    t = toy_setup(seed=5)
    rng = np.random.default_rng(0)
    hg, ag = rng.poisson(1.4, (1, 144)), rng.poisson(1.1, (1, 144))
    tab = compute_league_table(t.league_home, t.league_away, hg, ag, 36, rng=rng)
    df = standings_dataframe(tab, t.team_names)
    assert df["zone"].value_counts().to_dict() == {"play-offs": 16, "eliminated": 12, "round of 16": 8}
    assert df["position"].tolist() == list(range(1, 37)) and df["points"].is_monotonic_decreasing


def test_league_table_input_validation():
    with pytest.raises(ValueError, match="shape"):
        compute_league_table(np.array([0, 1]), np.array([1, 0]), np.zeros((1, 3)), np.zeros((1, 3)), 2)


# ======================================================================================
# scenario schedule generator + fixture validation
# ======================================================================================
def test_generated_schedule_follows_the_pot_structure():
    pots = [list(range(9 * k, 9 * k + 9)) for k in range(4)]
    for seed in range(5):
        home, away = generate_league_phase_fixtures(pots, np.random.default_rng(seed))
        validate_league_fixtures(home, away)  # 144 matches, 4H/4A each, no repeated pairing
        pot_of = {t: k for k, p in enumerate(pots) for t in p}
        for team in range(36):
            opp = [pot_of[a] for h, a in zip(home, away) if h == team] + [pot_of[h] for h, a in zip(home, away) if a == team]
            assert len(opp) == 8 and sorted(opp) == [0, 0, 1, 1, 2, 2, 3, 3]  # two opponents from each pot, own pot included
            hosts_pot = [pot_of[a] for h, a in zip(home, away) if h == team]
            assert sorted(hosts_pot) == [0, 1, 2, 3]  # one home match against each pot


def test_schedule_generator_is_seeded_and_validates_its_input():
    pots = [list(range(9 * k, 9 * k + 9)) for k in range(4)]
    a = generate_league_phase_fixtures(pots, np.random.default_rng(1))
    b = generate_league_phase_fixtures(pots, np.random.default_rng(1))
    c = generate_league_phase_fixtures(pots, np.random.default_rng(2))
    assert all((x == y).all() for x, y in zip(a, b)) and not all((x == y).all() for x, y in zip(a, c))
    with pytest.raises(SetupError):
        generate_league_phase_fixtures(pots[:3], np.random.default_rng(0))
    with pytest.raises(SetupError):
        generate_league_phase_fixtures([list(range(9))] * 4, np.random.default_rng(0))  # overlapping pots


def test_fixture_validation_rejects_broken_schedules():
    home, away = generate_league_phase_fixtures([list(range(9 * k, 9 * k + 9)) for k in range(4)], np.random.default_rng(0))
    with pytest.raises(SetupError, match="expected 144"):
        validate_league_fixtures(home[:-1], away[:-1])
    # turn one of team home[0]'s away matches into a home match: that team now hosts 5 and visits 3
    idx = int(np.where(away == home[0])[0][0])
    h2, a2 = home.copy(), away.copy()
    h2[idx], a2[idx] = away[idx], home[idx]
    with pytest.raises(SetupError, match="4 home and 4 away"):
        validate_league_fixtures(h2, a2)
    # the same pairing scheduled twice
    dup_h, dup_a = home.copy(), away.copy()
    dup_h[1], dup_a[1] = home[0], away[0]
    with pytest.raises(SetupError):
        validate_league_fixtures(dup_h, dup_a)


# ======================================================================================
# bracket structure
# ======================================================================================
def test_every_draw_outcome_respects_the_official_bracket_constraints():
    cand = candidate_draws()
    assert cand.shape == (4096, 12) and len({tuple(r) for r in cand}) == 4096
    m, t, s = (cand[:, i:i + 4].astype(int) for i in (0, 4, 8))
    loc = lambda rank: sim._locate(rank, m, t, s)  # noqa: E731
    for g in range(4):
        gs = [loc(r) for r in sim.TOP_SEEDS[g]]
        assert (gs[0][1] != gs[1][1]).all() and gs[0][0] == gs[1][0] == g          # seeds 1/2, 3/4, ... sit in opposite halves
        ps = [loc(r) for r in sim.PO_SEEDED[g]]
        us = [loc(r) for r in sim.PO_UNSEEDED[g]]
        assert (ps[0][1] != ps[1][1]).all() and (us[0][1] != us[1][1]).all()
        for k in (0, 1):  # a seeded club and its play-off opponent feed the SAME round-of-16 slot
            partner_half = np.where((k ^ m[:, g]) == 0, us[0][1], us[1][1])
            assert (partner_half == ps[k][1]).all()
            # ...which is the slot of exactly one top seed of the matching block
            top_half = [gs[0][1], gs[1][1]]
            assert ((ps[k][1] == top_half[0]) ^ (ps[k][1] == top_half[1])).all()


@pytest.mark.parametrize("season", [2024, 2025])
def test_the_real_brackets_of_the_last_two_seasons_fit_the_official_structure(season):
    """Strongest check available: all 23 real knockout pairings of a season are reproduced by the bracket model."""
    full = real_season(season)
    league = full[full["stage"] == Stage.LEAGUE]
    teams = sorted(set(league["home_team_id"]) | set(league["away_team_id"]))
    idx = {t: i for i, t in enumerate(teams)}
    tab = compute_league_table(league["home_team_id"].map(idx).to_numpy(), league["away_team_id"].map(idx).to_numpy(),
                               league["home_goals_90"].to_numpy()[None], league["away_goals_90"].to_numpy()[None], 36)
    ko = derive_known_knockout(full, idx)
    assert {k: len(v) for k, v in ko.pairs.items()} == {Stage.PLAYOFF: 8, Stage.LAST_16: 8, Stage.QUARTER: 4, Stage.SEMI: 2, Stage.FINAL: 1}
    po_teams = {x for pair in ko.pairs[Stage.PLAYOFF] for x in pair}
    assert po_teams == set(tab.order[0][8:24].tolist())  # my standings reproduce the real play-off field
    rank = tab.rank[0]
    allowed = consistent_draws(ko.pairs, rank)
    assert len(allowed) == 2  # exactly the draw and its mirror image: the real bracket is pinned down uniquely
    # narrowing is monotone as more rounds become known
    sizes = [len(consistent_draws({r: (ko.pairs[r] if r in rounds else []) for r in sim.ROUNDS}, rank)) for rounds in
             ([Stage.PLAYOFF], [Stage.PLAYOFF, Stage.LAST_16], [Stage.PLAYOFF, Stage.LAST_16, Stage.QUARTER], list(sim.ROUNDS))]
    assert sizes == [256, 16, 4, 2]
    # Second-leg hosting. Play-offs and round of 16: the seeded club hosts the return leg (both seasons). Quarter-finals:
    # the club in a seed 1-4 position (block A or B) hosts it; semi-finals: the club on the seed 1/2 side (block A/D branch).
    # Article 19.04 of the 2026/27 regulations says exactly that, and 2025/26 follows it in all 6 ties. The 2024/25 season
    # did NOT (e.g. Arsenal and Barcelona hosted the FIRST leg), so those two rounds are only asserted from 2025/26 on.
    m, t, s = (allowed[:1, i:i + 4].astype(int) for i in (0, 4, 8))
    block = lambda team: sim._locate(int(rank[team]), m, t, s)[0]  # noqa: E731
    rows = full[full["is_completed"]].sort_values("utc_date")
    rules = [(Stage.PLAYOFF, lambda team: 9 <= rank[team] <= 16), (Stage.LAST_16, lambda team: rank[team] <= 8)]
    if season >= 2025:
        rules += [(Stage.QUARTER, lambda team: block(team) in (0, 1)), (Stage.SEMI, lambda team: block(team) in (0, 3))]
    for rnd, ok in rules:
        by_pair = {}
        for r in rows[rows["stage"] == rnd].itertuples():
            by_pair[frozenset((idx[r.home_team_id], idx[r.away_team_id]))] = idx[r.home_team_id]  # last leg overwrites => 2nd leg host
        assert by_pair and all(ok(host) for host in by_pair.values()), rnd


def test_contradicting_pairings_are_rejected_not_silently_accepted():
    rank = np.arange(1, 37)  # team i finished (i+1)-th
    bad = {
        Stage.PLAYOFF: [(8, 9)],         # positions 9 v 10: two seeded clubs
        Stage.LAST_16: [(0, 2)],         # position 1 v 3: different blocks
        Stage.QUARTER: [(0, 2)],         # block A v block B
        Stage.SEMI: [(0, 6)],            # block A v block D (both on the same side of the semi-final)
    }
    for rnd, pairs in bad.items():
        with pytest.raises(InconsistentBracketError):
            consistent_draws({rnd: pairs}, rank)
    with pytest.raises(InconsistentBracketError, match="final league-phase ranking"):
        consistent_draws({Stage.PLAYOFF: [(8, 23)]}, None)
    with pytest.raises(InconsistentBracketError, match="finished 30th"):
        consistent_draws({Stage.LAST_16: [(0, 29)]}, rank)
    assert len(consistent_draws({}, None)) == 4096  # nothing known: every outcome possible


def test_seeds_one_and_two_cannot_meet_before_the_final():
    rank = np.arange(1, 37)
    ok = consistent_draws({Stage.QUARTER: [(0, 6)]}, rank)  # block A v block D inside one half: fine
    assert 0 < len(ok) < 4096
    with pytest.raises(InconsistentBracketError):  # ...but seeds 1 and 2 can never meet before the final
        consistent_draws({Stage.QUARTER: [(0, 1)]}, rank)


# ======================================================================================
# ties
# ======================================================================================
def _tie_setup(lam=1e-9):
    s = toy_setup(rates=RateTables.constant(36, home=lam, away=lam))
    return s


def _run_tie(setup, rnd, first, second, n=2000, seed=0, **cfg):
    rng = np.random.default_rng(seed)
    f = np.full((n, 1), first)
    sc = np.full((n, 1), second)
    return sim._two_legged_tie(rng, setup, rnd, f, sc, SimConfig(**cfg))[:, 0]


def test_two_legs_are_aggregated_not_treated_as_independent_matches():
    s = _tie_setup()
    ko = s.known_ko.goals[Stage.LAST_16]               # ko[home, away] = (home goals, away goals)
    # Team 7 wins the SECOND leg 1-0 at home but loses the tie 1-3 on aggregate.
    ko[5, 7] = (3, 0)
    ko[7, 5] = (1, 0)
    assert (_run_tie(s, Stage.LAST_16, 5, 7) == 5).all()
    assert (_run_tie(s, Stage.LAST_16, 7, 5) == 5).all()  # same answer whichever side is nominally 'first'
    # Team 5 loses the first leg 0-1 at home, then wins the second leg 2-0 away: 2-1 on aggregate.
    ko[5, 7] = (0, 1)
    ko[7, 5] = (0, 2)
    assert (_run_tie(s, Stage.LAST_16, 5, 7) == 5).all()
    # Team 7 wins both legs.
    ko[5, 7] = (0, 1)
    ko[7, 5] = (3, 0)
    assert (_run_tie(s, Stage.LAST_16, 5, 7) == 7).all()


def test_the_away_goals_rule_no_longer_exists():
    s = _tie_setup()
    ko = s.known_ko.goals[Stage.LAST_16]
    ko[5, 7] = (2, 1)   # leg 1 at 5: 5 wins 2-1 (7 scored 1 away goal)
    ko[7, 5] = (1, 0)   # leg 2 at 7: 7 wins 1-0 (5 scored 0 away goals) -> level 2-2, 7 has more away goals
    assert (_run_tie(s, Stage.LAST_16, 5, 7, penalty_home_win_prob=0.0) == 5).all()  # shoot-out decides (second host loses it)
    assert (_run_tie(s, Stage.LAST_16, 5, 7, penalty_home_win_prob=1.0) == 7).all()  # ...not the away goals
    frac = (_run_tie(s, Stage.LAST_16, 5, 7, n=6000, penalty_home_win_prob=0.5) == 7).mean()
    assert frac == pytest.approx(0.5, abs=0.04)


def test_extra_time_is_played_before_penalties_and_only_when_level():
    rates = RateTables.constant(36, home=1e-9, away=1e-9)
    rates.ko_home[7, 5] = 60.0           # if extra time is played at 7's ground, 7 will score ~20 goals in it
    s = toy_setup(rates=rates)
    ko = s.known_ko.goals[Stage.LAST_16]
    ko[5, 7] = (1, 1); ko[7, 5] = (1, 1)  # level on aggregate -> extra time
    assert (_run_tie(s, Stage.LAST_16, 5, 7, penalty_home_win_prob=0.0) == 7).all()  # decided in extra time, not by pens
    ko[5, 7] = (2, 0); ko[7, 5] = (0, 0)  # not level: extra time must NOT be played even though 7 would 'score' in it
    assert (_run_tie(s, Stage.LAST_16, 5, 7, penalty_home_win_prob=1.0) == 5).all()


def test_a_completed_tie_keeps_its_real_winner():
    s = _tie_setup()
    s.known_ko.goals[Stage.QUARTER][5, 7] = (1, 1)
    s.known_ko.goals[Stage.QUARTER][7, 5] = (1, 1)
    s.known_ko.winners[Stage.QUARTER][5, 7] = s.known_ko.winners[Stage.QUARTER][7, 5] = 7  # 7 won the real shoot-out
    assert (_run_tie(s, Stage.QUARTER, 5, 7, penalty_home_win_prob=0.0) == 7).all()


def test_only_the_unplayed_leg_is_simulated():
    s = _tie_setup(lam=1e-9)
    s.known_ko.goals[Stage.SEMI][5, 7] = (3, 0)  # first leg played; second leg will be drawn (almost surely 0-0)
    assert (_run_tie(s, Stage.SEMI, 5, 7) == 5).all()
    s2 = toy_setup(rates=RateTables.constant(36, home=1e-9, away=1e-9))
    s2.rates.ko_home[7, 5] = 40.0  # 7 hosts leg 2 and will score ~40: overturns 0-3
    s2.known_ko.goals[Stage.SEMI][5, 7] = (3, 0)
    assert (_run_tie(s2, Stage.SEMI, 5, 7) == 7).all()


def test_equal_teams_win_a_two_legged_tie_equally_often():
    s = toy_setup(rates=RateTables.constant(36, home=1.4, away=1.4))
    wins = (_run_tie(s, Stage.LAST_16, 3, 9, n=20000, seed=4) == 3).mean()
    assert wins == pytest.approx(0.5, abs=0.015)


def test_the_stronger_side_wins_more_often_and_home_advantage_matters_over_two_legs():
    strengths = np.zeros(36); strengths[3] = 1.0
    s = toy_setup(strengths=strengths)
    assert (_run_tie(s, Stage.LAST_16, 3, 9, n=8000) == 3).mean() > 0.6
    s2 = toy_setup(rates=RateTables.constant(36, home=2.0, away=0.8))  # identical teams, strong home advantage
    wins = (_run_tie(s2, Stage.LAST_16, 3, 9, n=20000) == 9).mean()    # 9 hosts the second leg and any extra time
    assert 0.5 < wins < 0.56


def test_final_is_a_single_neutral_match_with_extra_time_and_penalties():
    s = toy_setup(rates=RateTables.constant(36, home=1e-9, away=1e-9))
    s.rates.neutral[4, 8] = 25.0   # team 4 is overwhelmingly stronger on the day
    rng = np.random.default_rng(0)
    a, b = np.full(3000, 4), np.full(3000, 8)
    assert (sim._final_match(rng, s, a, b, SimConfig()) == 4).all()
    assert (sim._final_match(rng, s, b, a, SimConfig()) == 4).all()  # nominal home/away is irrelevant
    even = toy_setup(rates=RateTables.constant(36, home=1e-9, away=1e-9))  # 0-0 then 0-0 in extra time: pure shoot-out
    frac = (sim._final_match(np.random.default_rng(1), even, a, b, SimConfig()) == 4).mean()
    assert frac == pytest.approx(0.5, abs=0.04)
    even.known_ko.winners[Stage.FINAL][4, 8] = even.known_ko.winners[Stage.FINAL][8, 4] = 8
    assert (sim._final_match(np.random.default_rng(1), even, a, b, SimConfig()) == 8).all()  # the real result stands


# ======================================================================================
# whole-tournament properties
# ======================================================================================
@pytest.fixture(scope="module")
def toy_result():
    strengths = np.linspace(-0.6, 0.6, 36)
    return simulate_tournament(toy_setup(strengths=strengths, seed=2), SimConfig(n_sims=3000, seed=9)), strengths


def test_format_invariants_hold_in_every_single_simulation(toy_result):
    res, _ = toy_result
    n = res.n_sims
    rank = res.rank_per_sim
    assert (np.sort(rank, axis=1) == np.arange(1, 37)).all()                      # every sim: positions 1..36, each once
    assert ((rank <= 8).sum(1) == 8).all() and (((rank > 8) & (rank <= 24)).sum(1) == 16).all() and ((rank > 24).sum(1) == 12).all()
    assert res.position_counts.sum(0).tolist() == [n] * 36 and res.position_counts.sum(1).tolist() == [n] * 36
    sc = res.stage_counts
    assert [int(sc[k].sum()) for k in sim.STAGE_COLUMNS] == [16 * n, 8 * n, 4 * n, 2 * n, n]
    assert res.playoff_counts.sum() == 16 * n


def test_eliminated_clubs_can_never_win_and_champions_come_from_the_top_24(toy_result):
    res, _ = toy_result
    champ_rank = res.rank_per_sim[np.arange(res.n_sims), res.champion_per_sim]
    assert (champ_rank <= 24).all()
    qualified_for_r16 = res.stage_counts["reach_r16"]
    assert (res.stage_counts["champion"][qualified_for_r16 == 0] == 0).all()
    # nobody finishing 25th or lower reaches even the round of 16
    elim = res.rank_per_sim > 24
    assert not elim[np.arange(res.n_sims), res.champion_per_sim].any()
    assert (res.stage_counts["champion"] <= res.stage_counts["reach_final"]).all()
    assert (res.stage_counts["reach_final"] <= res.stage_counts["reach_semi_final"]).all()
    assert (res.stage_counts["reach_semi_final"] <= res.stage_counts["reach_quarter_final"]).all()
    assert (res.stage_counts["reach_quarter_final"] <= res.stage_counts["reach_r16"]).all()
    assert (res.stage_counts["reach_r16"] <= res.n_sims).all()


def test_top_8_clubs_skip_the_playoffs_and_playoff_clubs_play_an_extra_round():
    res = simulate_tournament(toy_setup(seed=3), SimConfig(n_sims=1500, seed=1))
    q = res.qualification_table()
    assert q["p_top8_direct_r16"].sum() == pytest.approx(8) and q["p_playoff"].sum() == pytest.approx(16)
    assert q["p_eliminated"].sum() == pytest.approx(12)
    # a club that finishes top-8 is in the round of 16 for sure; a play-off club reaches it only by winning a tie
    assert (q["p_reach_r16"] >= q["p_top8_direct_r16"] - 1e-12).all()
    assert (q["p_reach_r16"] <= q["p_top8_direct_r16"] + q["p_playoff"] + 1e-12).all()


def test_champion_probabilities_are_a_proper_distribution(toy_result):
    res, strengths = toy_result
    tab = res.champion_table()
    assert tab["simulated_championships"].sum() == res.n_sims
    assert tab["champion_probability"].sum() == pytest.approx(1.0, abs=1e-12)
    assert tab["rank"].tolist() == list(range(1, 37)) and tab["champion_probability"].is_monotonic_decreasing
    assert (tab["ci95_low"] <= tab["champion_probability"]).all() and (tab["champion_probability"] <= tab["ci95_high"]).all()
    assert (tab["ci95_low"] >= 0).all() and (tab["ci95_high"] <= 1).all()
    assert tab["forecast_mode"].eq(sim.MODE_SCENARIO).all()
    np.testing.assert_allclose(tab["mc_std_error"], np.sqrt(tab["champion_probability"] * (1 - tab["champion_probability"]) / res.n_sims))
    best = tab.iloc[0]["team_id"]
    assert best in {f"t{i:02d}" for i in range(26, 36)}  # one of the ten strongest sides leads the table
    assert res.final_standings_table()["expected_position"].is_monotonic_increasing
    fs = res.final_standings_table()
    assert fs[[f"p_pos_{i}" for i in range(1, 37)]].sum(axis=1).tolist() == pytest.approx([1.0] * 36)


def test_identical_teams_are_equally_likely_champions():
    res = simulate_tournament(toy_setup(seed=1), SimConfig(n_sims=12000, seed=5))
    p = res.champion_table()["champion_probability"].to_numpy()
    assert p.max() - p.min() < 0.014 and p.mean() == pytest.approx(1 / 36)


def test_a_much_stronger_club_is_the_favourite_but_never_certain():
    strengths = np.zeros(36); strengths[0] = 0.8
    res = simulate_tournament(toy_setup(strengths=strengths), SimConfig(n_sims=3000, seed=2))
    tab = res.champion_table()
    assert tab.iloc[0]["team_id"] == "t00" and 0.15 < tab.iloc[0]["champion_probability"] < 0.9
    assert tab.iloc[1]["champion_probability"] < tab.iloc[0]["champion_probability"] / 2


def test_same_seed_same_result_different_seed_different_result():
    setup = toy_setup(strengths=np.linspace(-0.4, 0.4, 36))
    a = simulate_tournament(setup, SimConfig(n_sims=800, seed=11))
    b = simulate_tournament(setup, SimConfig(n_sims=800, seed=11))
    c = simulate_tournament(setup, SimConfig(n_sims=800, seed=12))
    np.testing.assert_array_equal(a.position_counts, b.position_counts)
    np.testing.assert_array_equal(a.champion_per_sim, b.champion_per_sim)
    assert a.champion_table().equals(b.champion_table())
    assert not np.array_equal(a.champion_per_sim, c.champion_per_sim)


def test_simulation_count_is_configurable_and_validated():
    setup = toy_setup()
    assert simulate_tournament(setup, SimConfig(n_sims=50, seed=1)).n_sims == 50
    with pytest.raises(ValueError, match="n_sims"):
        simulate_tournament(setup, SimConfig(n_sims=0))


def test_default_run_of_ten_thousand_simulations_is_fast_enough_to_use():
    t0 = time.time()
    res = simulate_tournament(toy_setup(strengths=np.linspace(-0.5, 0.5, 36)), SimConfig())
    assert res.n_sims == 10_000 and time.time() - t0 < 60
    assert res.champion_table()["simulated_championships"].sum() == 10_000


def test_wilson_interval_and_standard_error_formulas():
    lo, hi = wilson_interval(50, 100)
    assert (float(lo), float(hi)) == pytest.approx((0.4038, 0.5962), abs=1e-4)
    lo0, hi0 = wilson_interval(0, 1000)
    assert float(lo0) == 0.0 and 0 < float(hi0) < 0.01
    assert float(binomial_std_error(50, 100)) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    more = wilson_interval(500, 1000)
    assert float(more[1] - more[0]) < float(hi - lo)  # more simulations => tighter interval


# ======================================================================================
# completed results are preserved, not re-simulated
# ======================================================================================
def test_known_league_results_are_never_resimulated():
    setup = toy_setup(strengths=np.zeros(36), seed=4)
    mine = (setup.league_home == 0) | (setup.league_away == 0)
    setup.league_played[mine] = True
    setup.league_home_goals[mine] = np.where(setup.league_home[mine] == 0, 5, 0)   # team 0 wins all 8 by 5-0
    setup.league_away_goals[mine] = np.where(setup.league_home[mine] == 0, 0, 5)
    res = simulate_tournament(setup, SimConfig(n_sims=1500, seed=3))
    assert res.position_counts[0, 0] == res.n_sims        # first in EVERY simulation: 24 points, huge goal difference
    assert res.notes["league_matches_known"] == 8 and res.notes["league_matches_simulated"] == 136
    assert res.stage_counts["reach_r16"][0] == res.n_sims


def test_a_finished_league_phase_is_deterministic():
    setup = toy_setup(seed=6)
    rng = np.random.default_rng(0)
    setup.league_played[:] = True
    setup.league_home_goals, setup.league_away_goals = rng.poisson(1.4, 144), rng.poisson(1.1, 144)
    setup.club_coefficients = np.arange(36, dtype=float)  # fixed final tie-break, so the seed cannot matter
    res = simulate_tournament(setup, SimConfig(n_sims=400, seed=1))
    assert (res.position_counts.max(axis=1) == res.n_sims).all()  # every club ends in one position in all simulations
    assert res.notes["league_matches_simulated"] == 0
    again = simulate_tournament(setup, SimConfig(n_sims=400, seed=99))
    np.testing.assert_array_equal(res.position_counts, again.position_counts)


# ======================================================================================
# setup validation
# ======================================================================================
def test_setup_validation_catches_malformed_tournaments():
    good = toy_setup()
    good.validate()
    bad = toy_setup(); bad.mode = "WISHFUL_THINKING"
    with pytest.raises(SetupError, match="unknown mode"):
        bad.validate()
    bad = toy_setup(); bad.team_ids = bad.team_ids[:35]; bad.team_names = bad.team_names[:35]
    with pytest.raises(SetupError, match="exactly 36"):
        bad.validate()
    bad = toy_setup(); bad.league_played[0] = True; bad.league_home_goals[0] = -1
    with pytest.raises(SetupError, match="non-negative"):
        bad.validate()
    bad = toy_setup(); bad.known_ko.pairs[Stage.PLAYOFF].append((0, 1))
    with pytest.raises(InconsistentBracketError, match="league phase is not finished"):
        bad.validate()
    with pytest.raises(SetupError, match="strictly positive"):
        RateTables(*(np.zeros((36, 36)) for _ in range(5)))
    with pytest.raises(SetupError, match="square"):
        RateTables(np.ones((36, 36)), np.ones((36, 36)), np.ones((36, 36)), np.ones((36, 36)), np.ones((35, 35)))


# ======================================================================================
# real-season replays (2025/26): the simulator against what really happened
# ======================================================================================
def _champion_probs(res):
    t = res.champion_table()
    return dict(zip(t["team_name"], t["champion_probability"]))


def test_with_every_result_known_the_real_champion_wins_in_every_simulation():
    setup = real_setup(2025, "2026-06-01")
    res = simulate_tournament(setup, SimConfig(n_sims=500, seed=1))
    p = _champion_probs(res)
    assert p["Paris Saint-Germain FC"] == 1.0 and sum(p.values()) == pytest.approx(1.0)
    assert res.notes["league_matches_simulated"] == 0 and res.notes["known_knockout_legs"] == 45
    assert (res.position_counts[res.team_names.index("Arsenal FC")].argmax()) == 0  # Arsenal finished 1st in the league phase


def test_before_the_final_only_the_two_finalists_can_win():
    setup = real_setup(2025, "2026-05-29")
    res = simulate_tournament(setup, SimConfig(n_sims=4000, seed=2))
    p = _champion_probs(res)
    assert {k for k, v in p.items() if v > 0} == {"Paris Saint-Germain FC", "Arsenal FC"}
    assert sum(p.values()) == pytest.approx(1.0)
    assert 0.4 < p["Arsenal FC"] < 0.6  # identical rates for both => a coin flip (up to noise)


def test_after_the_round_of_16_only_the_eight_quarter_finalists_can_win():
    res = simulate_tournament(real_setup(2025, "2026-04-01"), SimConfig(n_sims=3000, seed=3))
    t = res.champion_table()
    alive = {"Real Madrid CF", "FC Bayern München", "Sporting Clube de Portugal", "Arsenal FC", "Paris Saint-Germain FC",
             "Liverpool FC", "FC Barcelona", "Club Atlético de Madrid"}
    assert set(t.loc[t["champion_probability"] > 0, "team_name"]) <= alive
    assert t["champion_probability"].sum() == pytest.approx(1.0)
    reach_qf = dict(zip(res.team_names, res.stage_counts["reach_quarter_final"]))
    assert {k for k, v in reach_qf.items() if v == res.n_sims} == alive  # the 8 real winners reach the QF in every simulation
    assert sum(1 for v in reach_qf.values() if v > 0) == 8
    assert res.n_draw_outcomes == 4  # play-off, round-of-16 and quarter-final pairings are all known


def test_after_a_quarter_final_first_leg_the_played_leg_is_kept():
    setup = real_setup(2025, "2026-04-10")  # QF first legs played (Apr 7-8), second legs not
    assert setup.known_ko.n_known_legs == 16 + 16 + 4   # play-offs + round of 16 + the four first legs
    res = simulate_tournament(setup, SimConfig(n_sims=2000, seed=4))
    # Real first legs: Real 1-2 Bayern (Bayern won away), Sporting 0-1 Arsenal, PSG 2-0 Liverpool, Barcelona 0-2 Atletico.
    # With identical rates for every club the first-leg winner is clearly ahead, but nobody is certain.
    p = dict(zip(res.team_names, res.stage_counts["reach_semi_final"] / res.n_sims))
    assert p["Paris Saint-Germain FC"] > 0.6 and p["Club Atlético de Madrid"] > 0.6 and p["FC Bayern München"] > 0.6
    assert 0 < p["Liverpool FC"] < 0.4
    assert sum(p.values()) == pytest.approx(4.0)
    assert res.n_draw_outcomes == 4  # play-offs, round of 16 and quarter-final pairings known


def test_mid_league_phase_cutoff_keeps_played_matches_and_simulates_the_rest():
    setup = real_setup(2025, "2025-11-26")
    assert 0 < setup.league_played.sum() < 144 and not setup.league_phase_complete
    assert setup.known_ko.n_known_legs == 0
    res = simulate_tournament(setup, SimConfig(n_sims=1500, seed=5))
    assert res.notes["league_matches_known"] + res.notes["league_matches_simulated"] == 144
    assert res.n_draw_outcomes == 4096  # no knockout pairing is known yet
    assert res.champion_table()["champion_probability"].sum() == pytest.approx(1.0)


def test_playoff_draw_conditioning_when_the_pairings_are_already_known():
    res = simulate_tournament(real_setup(2025, "2026-01-30"), SimConfig(n_sims=800, seed=6))
    assert res.n_draw_outcomes == 256  # the eight play-off ties are fixed; the rest of the draw is still open
    only_po_known = res.stage_counts["reach_r16"]
    top8 = {"Arsenal FC", "FC Bayern München", "Liverpool FC", "Tottenham Hotspur FC", "FC Barcelona", "Chelsea FC",
            "Sporting Clube de Portugal", "Manchester City FC"}
    reach = dict(zip(res.team_names, only_po_known))
    assert all(reach[t] == res.n_sims for t in top8)  # the real top 8 go straight to the round of 16
    eliminated = {"Olympique de Marseille", "Paphos FC"}
    assert all(reach[t] == 0 for t in eliminated)


def test_incomplete_official_data_is_reported_instead_of_guessed():
    full = real_season(2025)
    league = full[full["stage"] == Stage.LEAGUE]
    truncated = pd.concat([league.iloc[:100], full[full["stage"] != Stage.LEAGUE]])
    with pytest.raises(OfficialDataUnavailable, match="expected 144"):
        sim.build_setup_from_matches(truncated, 2025, ConstantRatesModel(), FeatureBuilder(), mode=sim.MODE_LIVE, label="x")
    with pytest.raises(OfficialDataUnavailable):
        sim.build_setup_from_matches(full[full["season"] == 2099], 2099, ConstantRatesModel(), FeatureBuilder(),
                                     mode=sim.MODE_LIVE, label="x")


def test_replay_never_sees_anything_after_its_cutoff():
    visible = as_of(real_season(2025), "2025-09-01", mask_undetermined_knockout=True)
    setup = sim.build_setup_from_matches(visible, 2025, ConstantRatesModel(), FeatureBuilder(), mode=sim.MODE_REPLAY,
                                         label="r", cutoff=pd.Timestamp("2025-09-01"))
    assert not setup.league_played.any() and setup.known_ko.n_known_legs == 0
    assert not any(setup.known_ko.pairs.values())  # not even the future knockout pairings leak
    assert setup.mode == sim.MODE_REPLAY


def test_scenario_mode_is_labelled_and_flags_clubs_without_history():
    names = [f"Club {chr(65 + i)}{j}" for i in range(4) for j in range(9)]
    pots = [names[i * 9:(i + 1) * 9] for i in range(4)]
    setup = sim.build_scenario_setup(pots, ConstantRatesModel(), FeatureBuilder(), season=2026, seed=1)
    assert setup.mode == sim.MODE_SCENARIO and len(setup.team_notes) == 36
    assert any("not official" in a for a in setup.assumptions)
    res = simulate_tournament(setup, SimConfig(n_sims=300, seed=1))
    assert res.mode == sim.MODE_SCENARIO and res.champion_table()["forecast_mode"].eq(sim.MODE_SCENARIO).all()
    with pytest.raises(SetupError, match="four pots"):
        sim.build_scenario_setup(pots[:3], ConstantRatesModel(), FeatureBuilder(), season=2026)
    with pytest.raises(SetupError, match="four pots"):
        sim.build_scenario_setup([pots[0]] * 4, ConstantRatesModel(), FeatureBuilder(), season=2026)


# ======================================================================================
# wiring check against the REAL bracket: simulated ties must be the real ties
# ======================================================================================
@pytest.mark.parametrize("season,cutoff,rounds", [
    (2025, "2026-01-30", [Stage.PLAYOFF]),
    (2025, "2026-03-01", [Stage.PLAYOFF, Stage.LAST_16]),
    (2025, "2026-04-01", [Stage.PLAYOFF, Stage.LAST_16, Stage.QUARTER]),
    (2025, "2026-04-20", [Stage.PLAYOFF, Stage.LAST_16, Stage.QUARTER, Stage.SEMI]),
    (2024, "2025-04-20", [Stage.PLAYOFF, Stage.LAST_16, Stage.QUARTER, Stage.SEMI]),
])
def test_simulated_ties_are_exactly_the_real_ties_once_the_pairings_are_known(season, cutoff, rounds):
    """If the bracket wiring in the simulator disagreed with the real one, some real tie would not be simulated as such and
    both of its clubs could advance (or neither): for every known tie exactly one club must advance, in every simulation."""
    setup = real_setup(season, cutoff, rates_model=ConstantRatesModel(1.5, 1.2))
    res = simulate_tournament(setup, SimConfig(n_sims=1500, seed=8))
    advance = {Stage.PLAYOFF: "reach_r16", Stage.LAST_16: "reach_quarter_final", Stage.QUARTER: "reach_semi_final",
               Stage.SEMI: "reach_final", Stage.FINAL: "champion"}
    for rnd in rounds:
        pairs = setup.known_ko.pairs[rnd]
        assert pairs, rnd
        for a, b in pairs:
            both = res.stage_counts[advance[rnd]][a] + res.stage_counts[advance[rnd]][b]
            assert both == res.n_sims, f"{rnd}: {res.team_names[a]} v {res.team_names[b]}: {both} advancing in {res.n_sims} sims"
    # from the round of 16 on, every club still in the tournament is in a known tie, so nobody else can win it
    # (in the play-off round the eight direct qualifiers are in no tie yet)
    last = rounds[-1]
    if last != Stage.PLAYOFF:
        alive = {t for a, b in setup.known_ko.pairs[last] for t in (a, b)}
        assert set(np.nonzero(res.stage_counts["champion"])[0].tolist()) <= alive
