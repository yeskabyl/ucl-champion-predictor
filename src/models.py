"""Match-outcome models. All of them share one interface and one probability convention.

``predict_proba(df)`` -> ``(n, 3)`` array, columns ordered ``CLASSES = (H, D, A)``, rows sum to 1,
every entry strictly positive (so log-loss is always finite).

Models
------
A. :class:`EloBaseline`          -- Elo win expectancy + a one-parameter draw model fitted on the training period.
B. :class:`LogisticOutcomeModel` -- multinomial logistic regression (``logistic``: Elo + form; ``logistic_elo``: Elo state only).
C. :class:`CatBoostOutcomeModel` -- gradient-boosted trees (optional dependency).
D. :class:`PoissonGoalsModel`    -- two Poisson regressions (home goals, away goals) -> scoreline matrix -> H/D/A
   (``poisson``: Elo + form; ``poisson_elo``: Elo state only).
   :class:`ClimatologyBaseline`  -- class frequencies only; the "no skill" yardstick every model must beat.

Preprocessing (median imputation, scaling) lives *inside* each model's ``fit`` so it is learnt from the
training rows only.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import ELO_FEATURES, ELO_STATE_FEATURES, FEATURE_COLUMNS, LOGISTIC_FEATURES, POISSON_FEATURES
from .utils import CLASSES, get_logger

log = get_logger("ucl.models")

PROB_FLOOR = 1e-4
DEFAULT_TOTAL_GOALS = 2.75


# --------------------------------------------------------------------------------------
# Probability hygiene
# --------------------------------------------------------------------------------------
def normalize_probabilities(p, floor: float = PROB_FLOOR) -> np.ndarray:
    """Clip to ``floor`` and renormalise rows. Refuses NaN/inf instead of silently repairing them."""
    arr = np.asarray(p, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != len(CLASSES):
        raise ValueError(f"expected shape (n, {len(CLASSES)}), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("probabilities contain NaN or infinity")
    arr = np.clip(arr, floor, None)
    return arr / arr.sum(axis=1, keepdims=True)


def align_class_probabilities(proba: np.ndarray, fitted_classes, floor: float = PROB_FLOOR) -> np.ndarray:
    """Re-order a classifier's columns into ``CLASSES`` order, giving unseen classes the floor probability.

    Estimators sort ``classes_`` alphabetically (A, D, H) and drop classes absent from the training data;
    both would silently mis-assign columns if used raw.
    """
    proba = np.asarray(proba, dtype=float)
    out = np.full((proba.shape[0], len(CLASSES)), 0.0)
    for j, cls in enumerate(fitted_classes):
        if cls not in CLASSES:
            raise ValueError(f"unexpected class label {cls!r}")
        out[:, CLASSES.index(cls)] = proba[:, j]
    return normalize_probabilities(out, floor)


# --------------------------------------------------------------------------------------
# Poisson machinery (shared by the goals model and the simulator)
# --------------------------------------------------------------------------------------
def poisson_pmf(lam, max_goals: int = 10) -> np.ndarray:
    lam = np.clip(np.asarray(lam, dtype=float), 1e-6, None)[:, None]
    k = np.arange(max_goals + 1)[None, :]
    return np.exp(k * np.log(lam) - lam - gammaln(k + 1))


def outcome_probs_from_rates(lam_home, lam_away, max_goals: int = 10) -> np.ndarray:
    """H/D/A probabilities of two independent Poisson scores (truncated at ``max_goals``, renormalised)."""
    ph, pa = poisson_pmf(lam_home, max_goals), poisson_pmf(lam_away, max_goals)
    cdf_h, cdf_a = np.cumsum(ph, axis=1), np.cumsum(pa, axis=1)
    p_home = (ph[:, 1:] * cdf_a[:, :-1]).sum(axis=1)
    p_away = (pa[:, 1:] * cdf_h[:, :-1]).sum(axis=1)
    p_draw = (ph * pa).sum(axis=1)
    return normalize_probabilities(np.column_stack([p_home, p_draw, p_away]), floor=1e-12)


def rates_from_probabilities(proba: np.ndarray, total_goals: float = DEFAULT_TOTAL_GOALS,
                             iterations: int = 50, max_goals: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Poisson goal rates whose win-probability *difference* P(H)-P(A) equals the model's.

    Outcome-only models (Elo, logistic, CatBoost) say nothing about scorelines, yet the simulator needs
    goals for tie-breakers and two-legged aggregates. We fix the match's expected total goals at
    ``total_goals`` (the training-period average) and split it between the teams by bisection so that
    the implied supremacy matches. The draw probability then follows from the Poisson assumption rather
    than from the model -- an explicit modelling assumption, documented in the README.
    """
    p = normalize_probabilities(proba)
    target = p[:, 0] - p[:, 2]
    lo, hi = np.full(len(p), 1e-3), np.full(len(p), 1 - 1e-3)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        probs = outcome_probs_from_rates(total_goals * mid, total_goals * (1 - mid), max_goals)
        too_high = (probs[:, 0] - probs[:, 2]) > target
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
    share = 0.5 * (lo + hi)
    return np.clip(total_goals * share, 0.05, None), np.clip(total_goals * (1 - share), 0.05, None)


# --------------------------------------------------------------------------------------
# Base class
# --------------------------------------------------------------------------------------
class OutcomeModel(ABC):
    name = "model"
    feature_columns: list[str] = []
    complexity = 0  # used by the one-standard-error selection rule (lower = simpler)

    def __init__(self) -> None:
        self.mean_total_goals_ = DEFAULT_TOTAL_GOALS
        self.is_fitted_ = False

    def _require(self, df: pd.DataFrame, cols: list[str]) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"{self.name}: missing feature columns {missing}")

    def fit(self, train: pd.DataFrame) -> "OutcomeModel":
        train = train[train["is_completed"]] if "is_completed" in train else train
        if train.empty:
            raise ValueError(f"{self.name}: no completed training matches")
        if train["result"].isna().any():
            raise ValueError(f"{self.name}: training rows with a missing result")
        self._require(train, self.feature_columns + ["result", "home_goals_90", "away_goals_90"])
        self.mean_total_goals_ = float((train["home_goals_90"] + train["away_goals_90"]).mean())
        self._fit(train)
        self.is_fitted_ = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError(f"{self.name} is not fitted")
        self._require(df, self.feature_columns)
        return normalize_probabilities(self._predict_proba(df))

    def predict_goal_rates(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(home, away) expected goals. Default: derived from the outcome probabilities (see above)."""
        return rates_from_probabilities(self.predict_proba(df), self.mean_total_goals_)

    @abstractmethod
    def _fit(self, train: pd.DataFrame) -> None: ...

    @abstractmethod
    def _predict_proba(self, df: pd.DataFrame) -> np.ndarray: ...


class ClimatologyBaseline(OutcomeModel):
    """Predicts the training-period class frequencies for every match."""

    name = "climatology"
    complexity = 0

    def _fit(self, train: pd.DataFrame) -> None:
        counts = train["result"].value_counts().reindex(CLASSES).fillna(0).to_numpy(float) + 1.0  # Laplace
        self.freq_ = counts / counts.sum()

    def _predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.tile(self.freq_, (len(df), 1))


class EloBaseline(OutcomeModel):
    """Model A. ``P(D) = d * 4E(1-E)``; the single constant ``d`` is fitted so the average predicted draw rate
    equals the training draw rate. Then ``P(H) = E - P(D)/2`` and ``P(A) = 1 - E - P(D)/2``."""

    name = "elo"
    feature_columns = ELO_FEATURES
    complexity = 1

    def _fit(self, train: pd.DataFrame) -> None:
        e = train["elo_expected_home"].to_numpy(float)
        draw_rate = float((train["result"] == "D").mean())
        self.draw_scale_ = draw_rate / float(np.mean(4 * e * (1 - e)))

    def _predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        e = df["elo_expected_home"].to_numpy(float)
        p_d = np.minimum(self.draw_scale_ * 4 * e * (1 - e), 0.9 * 2 * np.minimum(e, 1 - e))
        return np.column_stack([e - p_d / 2, p_d, 1 - e - p_d / 2])


def _preprocessor() -> list[tuple[str, object]]:
    return [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]


class LogisticOutcomeModel(OutcomeModel):
    """Model B: multinomial logistic regression, L2-regularised (C=1) -- no tuning on ~2k matches."""

    name = "logistic"
    feature_columns = LOGISTIC_FEATURES
    complexity = 5

    def __init__(self, C: float = 1.0, features: list[str] | None = None, name: str | None = None, complexity: int | None = None):
        super().__init__()
        self.C = C
        if features is not None:
            self.feature_columns = list(features)
        if name is not None:
            self.name = name
        if complexity is not None:
            self.complexity = complexity

    def _fit(self, train: pd.DataFrame) -> None:
        self.pipe_ = Pipeline(_preprocessor() + [("clf", LogisticRegression(C=self.C, max_iter=2000))])
        self.pipe_.fit(train[self.feature_columns], train["result"])

    def _predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.pipe_.predict_proba(df[self.feature_columns])
        return align_class_probabilities(raw, self.pipe_.classes_)


class PoissonGoalsModel(OutcomeModel):
    """Model D: independent Poisson regressions for home and away goals (log link, L2 alpha=1)."""

    name = "poisson"
    feature_columns = POISSON_FEATURES
    complexity = 6

    def __init__(self, alpha: float = 1.0, max_goals: int = 10, features: list[str] | None = None,
                 name: str | None = None, complexity: int | None = None):
        super().__init__()
        self.alpha, self.max_goals = alpha, max_goals
        if features is not None:
            self.feature_columns = list(features)
        if name is not None:
            self.name = name
        if complexity is not None:
            self.complexity = complexity

    def _fit(self, train: pd.DataFrame) -> None:
        def make() -> Pipeline:
            return Pipeline(_preprocessor() + [("reg", PoissonRegressor(alpha=self.alpha, max_iter=1000))])

        self.home_ = make().fit(train[self.feature_columns], train["home_goals_90"])
        self.away_ = make().fit(train[self.feature_columns], train["away_goals_90"])

    def predict_goal_rates(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted_:
            raise RuntimeError(f"{self.name} is not fitted")
        self._require(df, self.feature_columns)
        x = df[self.feature_columns]
        return np.clip(self.home_.predict(x), 0.05, 6.0), np.clip(self.away_.predict(x), 0.05, 6.0)

    def _predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        lam_h, lam_a = self.predict_goal_rates(df)
        return outcome_probs_from_rates(lam_h, lam_a, self.max_goals)


class CatBoostOutcomeModel(OutcomeModel):
    """Model C (optional). Shallow trees, small learning rate, fixed iteration count: deliberately conservative."""

    name = "catboost"
    feature_columns = FEATURE_COLUMNS
    complexity = 7

    def __init__(self, iterations: int = 300, depth: int = 4, learning_rate: float = 0.05, l2_leaf_reg: float = 5.0,
                 seed: int = 42):
        super().__init__()
        self.params = dict(iterations=iterations, depth=depth, learning_rate=learning_rate, l2_leaf_reg=l2_leaf_reg,
                           random_seed=seed)

    def _fit(self, train: pd.DataFrame) -> None:
        from catboost import CatBoostClassifier  # imported lazily: the dependency is optional

        self.clf_ = CatBoostClassifier(loss_function="MultiClass", verbose=False, allow_writing_files=False,
                                       thread_count=1, **self.params)
        self.clf_.fit(train[self.feature_columns], train["result"])

    def _predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.clf_.predict_proba(df[self.feature_columns])
        return align_class_probabilities(raw, list(self.clf_.classes_))


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------
def catboost_available() -> bool:
    try:
        import catboost  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - any import problem means "not available"
        return False


def make_model_factories(seed: int = 42, include_catboost: bool = True) -> dict[str, Callable[[], OutcomeModel]]:
    """Name -> zero-argument constructor, so every fold trains a fresh, independent model."""
    factories: dict[str, Callable[[], OutcomeModel]] = {
        "climatology": ClimatologyBaseline,
        "elo": EloBaseline,
        # same learners restricted to slow-moving Elo state: robust when ratings are frozen for a whole season
        "logistic_elo": lambda: LogisticOutcomeModel(features=ELO_STATE_FEATURES, name="logistic_elo", complexity=2),
        "poisson_elo": lambda: PoissonGoalsModel(features=ELO_STATE_FEATURES, name="poisson_elo", complexity=3),
        "logistic": LogisticOutcomeModel,
        "poisson": PoissonGoalsModel,
    }
    if include_catboost:
        if catboost_available():
            factories["catboost"] = lambda: CatBoostOutcomeModel(seed=seed)
        else:
            log.warning("catboost is not installed -> skipping the optional CatBoost model")
    return factories
