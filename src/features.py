"""Leakage-safe feature engineering.

The single most important property of this module: **the features of a match are computed from team
state that is frozen *before* the match, and the state is updated *afterwards***. There is exactly one
code path (:class:`FeatureBuilder`) for historical rows, for upcoming fixtures, and for the
hypothetical pairings the tournament simulator needs, so training and serving cannot drift apart.

Features (home team ``h``, away team ``a``):

========================  =====================================================================
``elo_home/away/diff``     pre-match Elo ratings (see :mod:`src.elo`)
``elo_expected_home``      Elo win expectancy including home advantage (0 advantage if neutral)
``gf{5,10}_*``/``ga*``     mean goals scored / conceded over the team's previous 5 / 10 matches
``pts5_*``                 mean league points (3/1/0, 90-minute result) over the previous 5 matches
``opp_elo5_*``             mean pre-match Elo of the previous 5 opponents (strength of schedule)
``rest_days_*``            days since the team's previous *Champions League* match (capped)
``n_prev_*``               number of previous matches available (capped at the longest window)
``*_diff``                 home minus away of the quantities above
``neutral``, ``is_knockout``
========================  =====================================================================

Caveat that is worth saying out loud in an interview: only Champions League matches are observed, so
"form" and "rest" ignore domestic fixtures. xG and domestic-league strength are deliberately not
included because no reliable source is wired in.
"""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .elo import EloConfig, EloRatingSystem
from .utils import Stage, get_logger, outcome_label

log = get_logger("ucl.features")

WINDOWS = (5, 10)
MAX_REST_DAYS = 30.0

ID_COLUMNS = ["match_id", "season", "utc_date", "stage", "home_team_id", "away_team_id", "is_completed"]
LABEL_COLUMNS = ["result", "home_goals_90", "away_goals_90"]

FEATURE_COLUMNS = [
    "elo_home", "elo_away", "elo_diff", "elo_expected_home",
    "gf5_home", "ga5_home", "gf10_home", "ga10_home", "pts5_home", "opp_elo5_home", "rest_days_home", "n_prev_home",
    "gf5_away", "ga5_away", "gf10_away", "ga10_away", "pts5_away", "opp_elo5_away", "rest_days_away", "n_prev_away",
    "gd5_diff", "gd10_diff", "pts5_diff", "opp_elo5_diff", "rest_diff",
    "neutral", "is_knockout",
]

# Compact sets for the simple models (few, interpretable inputs generalise better on ~2k matches).
LOGISTIC_FEATURES = ["elo_diff", "neutral", "is_knockout", "gd5_diff", "gd10_diff", "pts5_diff", "opp_elo5_diff", "rest_diff"]
POISSON_FEATURES = ["elo_diff", "neutral", "is_knockout", "gf5_home", "ga5_home", "gf5_away", "ga5_away", "gd10_diff"]
ELO_FEATURES = ["elo_expected_home"]
# "Slow" state only: Elo moves over seasons, unlike 5/10-match form, so it survives being frozen for months.
ELO_STATE_FEATURES = ["elo_diff", "neutral", "is_knockout"]


@dataclass
class _Played:
    date: pd.Timestamp
    gf: float
    ga: float
    pts: float
    opp_elo: float


def _mean_last(history, n: int, attr: str) -> float:
    vals = [getattr(p, attr) for p in list(history)[-n:]]
    return float(np.mean(vals)) if vals else np.nan


class FeatureBuilder:
    """Mutable per-team state. ``features_for`` reads it, ``update`` advances it."""

    def __init__(self, elo_config: EloConfig | None = None, windows: tuple[int, ...] = WINDOWS):
        if tuple(windows) != WINDOWS:
            raise ValueError(f"feature names are defined for windows {WINDOWS}")
        self.elo = EloRatingSystem(elo_config)
        self.windows = tuple(windows)
        self._history: dict[str, deque[_Played]] = {}
        self.n_updates = 0

    # ---- read ---------------------------------------------------------------------
    def _team_block(self, team: str, date: pd.Timestamp | None, season: int | None, suffix: str) -> dict[str, float]:
        hist = self._history.get(team, ())
        hist = list(hist)
        out = {
            f"gf5_{suffix}": _mean_last(hist, 5, "gf"), f"ga5_{suffix}": _mean_last(hist, 5, "ga"),
            f"gf10_{suffix}": _mean_last(hist, 10, "gf"), f"ga10_{suffix}": _mean_last(hist, 10, "ga"),
            f"pts5_{suffix}": _mean_last(hist, 5, "pts"), f"opp_elo5_{suffix}": _mean_last(hist, 5, "opp_elo"),
            f"n_prev_{suffix}": float(min(len(hist), max(self.windows))),
            f"rest_days_{suffix}": np.nan,
        }
        if hist and date is not None and not pd.isna(date):
            days = (pd.Timestamp(date) - hist[-1].date).total_seconds() / 86400.0
            out[f"rest_days_{suffix}"] = float(min(max(days, 0.0), MAX_REST_DAYS))
        return out

    def features_for(self, home: str, away: str, date: pd.Timestamp | None = None, *, neutral: bool = False,
                     stage: str | None = None, season: int | None = None, rest_known: bool = True) -> dict[str, float]:
        """Feature dict from current state. Never mutates it.

        ``rest_known=False`` blanks the rest-day features (the simulator does this for hypothetical
        pairings because the kick-off date of a not-yet-scheduled tie is unknown; the model's imputer
        then substitutes the *training* median).
        """
        rh = self.elo.rating_for_season(home, season)
        ra = self.elo.rating_for_season(away, season)
        f: dict[str, float] = {
            "elo_home": rh, "elo_away": ra, "elo_diff": rh - ra,
            "elo_expected_home": self.elo.expected_home(rh, ra, neutral),
            "neutral": float(bool(neutral)),
            "is_knockout": float(stage in Stage.KNOCKOUT) if stage else 0.0,
        }
        d = date if rest_known else None
        f.update(self._team_block(home, d, season, "home"))
        f.update(self._team_block(away, d, season, "away"))
        f["gd5_diff"] = (f["gf5_home"] - f["ga5_home"]) - (f["gf5_away"] - f["ga5_away"])
        f["gd10_diff"] = (f["gf10_home"] - f["ga10_home"]) - (f["gf10_away"] - f["ga10_away"])
        f["pts5_diff"] = f["pts5_home"] - f["pts5_away"]
        f["opp_elo5_diff"] = f["opp_elo5_home"] - f["opp_elo5_away"]
        f["rest_diff"] = f["rest_days_home"] - f["rest_days_away"]
        return f

    # ---- write --------------------------------------------------------------------
    def update(self, home: str, away: str, home_goals: float, away_goals: float, date: pd.Timestamp,
               *, season: int, neutral: bool = False) -> None:
        """Fold a finished match into the state. Call only AFTER ``features_for`` for that match."""
        self.elo.advance_season(season)
        rh, ra = self.elo.rating(home), self.elo.rating(away)  # pre-match, for the opponents' strength history
        self.elo.update(home, away, home_goals, away_goals, neutral)
        pts_h = 3.0 if home_goals > away_goals else 1.0 if home_goals == away_goals else 0.0
        pts_a = 3.0 if away_goals > home_goals else 1.0 if home_goals == away_goals else 0.0
        maxlen = max(self.windows)
        self._history.setdefault(home, deque(maxlen=maxlen)).append(_Played(date, home_goals, away_goals, pts_h, ra))
        self._history.setdefault(away, deque(maxlen=maxlen)).append(_Played(date, away_goals, home_goals, pts_a, rh))
        self.n_updates += 1

    def known_teams(self) -> set[str]:
        return set(self._history)


def _prepare(matches: pd.DataFrame) -> pd.DataFrame:
    if matches["match_id"].duplicated().any():
        raise ValueError("duplicate match_id values; run data_loader.clean_matches first")
    df = matches.sort_values(["utc_date", "match_id"], kind="mergesort").reset_index(drop=True)
    has_teams = df["home_team_id"].notna() & df["away_team_id"].notna()
    if (~has_teams).any():
        log.info("skipping %d rows with an undetermined team", int((~has_teams).sum()))
    return df[has_teams]


def _is_completed(r) -> bool:
    return bool(r.is_completed) and not (pd.isna(r.home_goals_90) or pd.isna(r.away_goals_90))


def _record(r, completed: bool, feats: dict) -> dict:
    rec = {
        "match_id": r.match_id, "season": int(r.season), "utc_date": r.utc_date, "stage": r.stage,
        "home_team_id": r.home_team_id, "away_team_id": r.away_team_id, "is_completed": completed,
        "result": outcome_label(r.home_goals_90, r.away_goals_90) if completed else None,
        "home_goals_90": r.home_goals_90 if completed else np.nan,
        "away_goals_90": r.away_goals_90 if completed else np.nan,
    }
    rec.update(feats)
    return rec


def build_features(matches: pd.DataFrame, elo_config: EloConfig | None = None) -> tuple[pd.DataFrame, FeatureBuilder]:
    """Chronological pass over ``matches``.

    * Completed matches get features from the state *before* them, then update the state.
    * Not-completed matches (upcoming fixtures) get features from the state as it stands then, and
      never update anything.
    * Rows without both team ids (TBD knockout slots) are skipped and counted.
    * Duplicated ``match_id`` values are an error -- clean the table first (``clean_matches``).

    Returns the feature table (``FEATURE_COLUMNS`` + ids + labels) and the builder whose state is the
    snapshot after the last completed match (used by the simulator).
    """
    df = _prepare(matches)
    builder = FeatureBuilder(elo_config)
    rows = []
    for r in df.itertuples(index=False):
        completed = _is_completed(r)
        feats = builder.features_for(r.home_team_id, r.away_team_id, r.utc_date, neutral=bool(r.neutral),
                                     stage=r.stage, season=int(r.season))
        rows.append(_record(r, completed, feats))
        if completed:  # <- state changes only after the features were recorded
            builder.update(r.home_team_id, r.away_team_id, r.home_goals_90, r.away_goals_90, r.utc_date,
                           season=int(r.season), neutral=bool(r.neutral))
    return pd.DataFrame(rows, columns=ID_COLUMNS + LABEL_COLUMNS + FEATURE_COLUMNS), builder


def build_frozen_features(matches: pd.DataFrame, elo_config: EloConfig | None = None) -> pd.DataFrame:
    """Features of EVERY match of a season computed from the team state at the START of that season.

    This reproduces how the tournament simulator predicts: ratings and form are frozen at the cutoff and applied to
    matches played weeks or months later. Evaluating models on these rows tests them at the horizon they are used at,
    where fast-moving form features go stale while Elo does not. Rest days are blanked, as in the simulator.
    Still leakage-safe: the snapshot for season ``s`` contains only matches played before season ``s``.
    """
    df = _prepare(matches)
    builder = FeatureBuilder(elo_config)
    snapshots: dict[int, FeatureBuilder] = {}
    current = None
    for r in df.itertuples(index=False):
        if r.season != current:
            snapshots[int(r.season)] = copy.deepcopy(builder)
            current = r.season
        if _is_completed(r):
            builder.update(r.home_team_id, r.away_team_id, r.home_goals_90, r.away_goals_90, r.utc_date,
                           season=int(r.season), neutral=bool(r.neutral))
    rows = []
    for r in df.itertuples(index=False):
        feats = snapshots[int(r.season)].features_for(r.home_team_id, r.away_team_id, None, neutral=bool(r.neutral),
                                                      stage=r.stage, season=int(r.season), rest_known=False)
        rows.append(_record(r, _is_completed(r), feats))
    return pd.DataFrame(rows, columns=ID_COLUMNS + LABEL_COLUMNS + FEATURE_COLUMNS)
