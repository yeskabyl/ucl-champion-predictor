"""Elo rating system for football.

Conventions (World-Football-Elo style, which is what makes it interview-friendly to explain):

* ``E_home = 1 / (1 + 10 ** (-(R_home + HA - R_away) / 400))`` where ``HA`` is a fixed home advantage
  in rating points (0 at a neutral venue).
* The "actual score" is 1 / 0.5 / 0 for a home win / draw / away win in **90 minutes**.
* ``delta = K * G * (S - E)`` with a goal-difference multiplier ``G`` (1, 1.5, then (11 + gd) / 8).
* Updates are zero-sum, so the average rating of the pool stays at the initial rating.
* Between seasons ratings shrink slightly toward the mean (squads change).

The class only ever mutates state in :meth:`update` / :meth:`advance_season`; reading a rating or an
expectation is side-effect free, which is what the leakage tests rely on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EloConfig:
    initial_rating: float = 1500.0
    k_factor: float = 20.0
    home_advantage: float = 65.0
    scale: float = 400.0
    use_goal_difference: bool = True
    season_regression: float = 0.10  # share of the distance to ``regression_target`` removed per new season
    regression_target: float = 1500.0


@dataclass(frozen=True)
class EloUpdate:
    home_pre: float
    away_pre: float
    expected_home: float
    actual_home: float
    delta_home: float  # change applied to the home team (the away team receives -delta_home)


def goal_difference_multiplier(goal_diff: int) -> float:
    gd = abs(int(goal_diff))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def actual_score(home_goals: float, away_goals: float) -> float:
    if home_goals > away_goals:
        return 1.0
    if home_goals < away_goals:
        return 0.0
    return 0.5


class EloRatingSystem:
    def __init__(self, config: EloConfig | None = None, ratings: dict[str, float] | None = None,
                 season: int | None = None):
        self.config = config or EloConfig()
        self.ratings: dict[str, float] = dict(ratings or {})
        self.season = season

    # ---- reads (no side effects) ------------------------------------------------------
    def rating(self, team_id: str) -> float:
        return self.ratings.get(team_id, self.config.initial_rating)

    def rating_for_season(self, team_id: str, season: int | None) -> float:
        """Rating as it would be after the between-season shrinkage up to ``season`` (does not mutate)."""
        r = self.rating(team_id)
        if season is None or self.season is None or season <= self.season:
            return r
        shrink = (1.0 - self.config.season_regression) ** (season - self.season)
        target = self.config.regression_target
        return target + (r - target) * shrink

    def expected_home(self, home_rating: float, away_rating: float, neutral: bool = False) -> float:
        ha = 0.0 if neutral else self.config.home_advantage
        return 1.0 / (1.0 + 10.0 ** (-(home_rating + ha - away_rating) / self.config.scale))

    # ---- writes ------------------------------------------------------------------------
    def advance_season(self, season: int) -> None:
        """Apply the between-season shrinkage once per season boundary crossed."""
        if self.season is None:
            self.season = season
            return
        if season < self.season:
            raise ValueError(f"matches must be processed chronologically (got season {season} after {self.season})")
        gap = season - self.season
        if gap:
            shrink = (1.0 - self.config.season_regression) ** gap
            target = self.config.regression_target
            self.ratings = {t: target + (r - target) * shrink for t, r in self.ratings.items()}
            self.season = season

    def update(self, home: str, away: str, home_goals: float, away_goals: float, neutral: bool = False) -> EloUpdate:
        if home == away:
            raise ValueError("a team cannot play itself")
        if home_goals is None or away_goals is None or np.isnan(home_goals) or np.isnan(away_goals):
            raise ValueError("cannot update Elo from a match without a score")
        rh, ra = self.rating(home), self.rating(away)
        expected = self.expected_home(rh, ra, neutral)
        actual = actual_score(home_goals, away_goals)
        mult = goal_difference_multiplier(int(home_goals - away_goals)) if self.config.use_goal_difference else 1.0
        delta = self.config.k_factor * mult * (actual - expected)
        self.ratings[home] = rh + delta
        self.ratings[away] = ra - delta
        return EloUpdate(rh, ra, expected, actual, delta)

    def snapshot(self) -> dict[str, float]:
        return dict(self.ratings)


def run_elo(matches: pd.DataFrame, config: EloConfig | None = None) -> tuple[pd.DataFrame, EloRatingSystem]:
    """Walk completed matches in time order. ``elo_*_pre`` columns are recorded BEFORE each update."""
    system = EloRatingSystem(config)
    done = matches[matches["is_completed"]].sort_values(["utc_date", "match_id"])
    rows = []
    for r in done.itertuples(index=False):
        system.advance_season(int(r.season))
        upd = system.update(r.home_team_id, r.away_team_id, r.home_goals_90, r.away_goals_90, bool(r.neutral))
        rows.append((r.match_id, upd.home_pre, upd.away_pre, upd.expected_home))
    out = pd.DataFrame(rows, columns=["match_id", "elo_home_pre", "elo_away_pre", "elo_expected_home"])
    return out, system
