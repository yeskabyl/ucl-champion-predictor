"""Builders for *synthetic* test data. Everything invented lives here and is only ever used inside tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import MATCH_COLUMNS, _finalise_types, parse_openfootball_text

FIXTURES = Path(__file__).parent / "fixtures"


def make_matches(rows: list[dict]) -> pd.DataFrame:
    """Normalised match table from compact dicts: ``home, away, hg, ag, date`` (+ optional season/stage/completed/...)."""
    out = []
    for i, r in enumerate(rows):
        completed = r.get("completed", True) and r.get("hg") is not None
        hg, ag = (r["hg"], r["ag"]) if completed else (np.nan, np.nan)
        stage = r.get("stage", "GROUP_STAGE")
        out.append({
            "match_id": r.get("match_id", f"t:{i:04d}"), "season": r.get("season", 2020),
            "utc_date": pd.Timestamp(r["date"]), "stage": stage, "matchday": None, "group": None,
            "home_team_id": r["home"], "home_team": r["home"].upper() if r["home"] else None,
            "away_team_id": r["away"], "away_team": r["away"].upper() if r["away"] else None,
            "status": "FINISHED" if completed else "SCHEDULED", "is_completed": completed,
            "home_goals_90": hg, "away_goals_90": ag, "home_goals_aet": hg, "away_goals_aet": ag,
            "decided_by": "REGULAR" if completed else None, "pens_home": np.nan, "pens_away": np.nan,
            "winner": None if not completed else ("HOME_TEAM" if hg > ag else "AWAY_TEAM" if hg < ag else "DRAW"),
            "neutral": r.get("neutral", stage == "FINAL"), "source": "synthetic-test", "source_match_id": f"t{i}",
            "home_source_id": None, "away_source_id": None, "retrieved_at": "2000-01-01T00:00:00+00:00",
        })
    return _finalise_types(pd.DataFrame(out, columns=MATCH_COLUMNS))


def random_league(n_teams: int = 6, n_rounds: int = 10, seed: int = 0, start: str = "2018-09-01",
                  seasons: int = 1) -> pd.DataFrame:
    """Round-robin-ish synthetic league with random scores, one match per team per 7 days."""
    rng = np.random.default_rng(seed)
    teams = [f"t{i}" for i in range(n_teams)]
    rows, day = [], pd.Timestamp(start)
    for season_offset in range(seasons):
        for rnd in range(n_rounds):
            order = rng.permutation(n_teams)
            for k in range(0, n_teams - 1, 2):
                h, a = teams[order[k]], teams[order[k + 1]]
                rows.append({"home": h, "away": a, "hg": int(rng.poisson(1.5)), "ag": int(rng.poisson(1.1)),
                             "date": day + pd.Timedelta(days=7 * rnd, hours=int(rng.integers(0, 3))),
                             "season": 2018 + season_offset})
        day = day + pd.Timedelta(days=365)
    return make_matches(rows)


def real_season(season_start: int) -> pd.DataFrame:
    """A real, unmodified openfootball season (CC0) from tests/fixtures."""
    text = (FIXTURES / f"cl_{season_start}-{str(season_start + 1)[-2:]}.txt").read_text(encoding="utf-8")
    return parse_openfootball_text(text, season_start, retrieved_at="2000-01-01T00:00:00+00:00")


class ConstantRatesModel:
    """Stand-in for a fitted model: every match has the same expected goals (used to test the simulator on real fixtures)."""

    name = "constant"

    def __init__(self, home: float = 1.5, away: float = 1.2):
        self.home, self.away = home, away

    def predict_goal_rates(self, df: pd.DataFrame):
        return np.full(len(df), self.home), np.full(len(df), self.away)
