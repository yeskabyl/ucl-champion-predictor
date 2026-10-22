"""Chronological evaluation of probabilistic match-outcome models.

Rules enforced here (each one is a classic leakage / over-claiming trap):

* splits are by **season order**, never shuffled; train < validation < test, no overlap;
* every model is re-fitted from scratch on the training rows only (imputation and scaling live inside
  the model, see :mod:`src.models`);
* models are **selected on the validation period**; the test period is only reported, never used to choose;
* every metric is reported with its sample size, and sample-size caveats are returned as text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .models import OutcomeModel, normalize_probabilities
from .utils import CLASSES, get_logger

log = get_logger("ucl.eval")


class InsufficientDataError(ValueError):
    """Not enough completed seasons for a chronological train/validation/test split."""


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def _class_index(y_true: Sequence[str]) -> np.ndarray:
    lookup = {c: i for i, c in enumerate(CLASSES)}
    try:
        return np.array([lookup[y] for y in y_true], dtype=int)
    except KeyError as exc:
        raise ValueError(f"unknown outcome label {exc.args[0]!r}; expected one of {CLASSES}") from None


def per_match_log_loss(y_true: Sequence[str], proba, eps: float = 1e-15) -> np.ndarray:
    p = normalize_probabilities(proba, floor=eps)
    idx = _class_index(y_true)
    return -np.log(p[np.arange(len(idx)), idx])


def multiclass_log_loss(y_true: Sequence[str], proba) -> float:
    return float(per_match_log_loss(y_true, proba).mean())


def multiclass_brier(y_true: Sequence[str], proba) -> float:
    """Mean over matches of ``sum_k (p_k - 1[y=k])^2`` -- ranges from 0 (perfect) to 2."""
    p = normalize_probabilities(proba)
    onehot = np.eye(len(CLASSES))[_class_index(y_true)]
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def accuracy(y_true: Sequence[str], proba) -> float:
    p = normalize_probabilities(proba)
    return float((p.argmax(axis=1) == _class_index(y_true)).mean())


def calibration_table(y_true: Sequence[str], proba, n_bins: int = 5) -> pd.DataFrame:
    """Reliability table per class: equal-count bins of predicted probability vs observed frequency."""
    p = normalize_probabilities(proba)
    idx = _class_index(y_true)
    rows = []
    for k, cls in enumerate(CLASSES):
        order = np.argsort(p[:, k], kind="mergesort")
        for b, chunk in enumerate(np.array_split(order, min(n_bins, len(order)))):
            if len(chunk) == 0:
                continue
            rows.append({
                "class": cls, "bin": b + 1, "n": len(chunk),
                "mean_predicted": float(p[chunk, k].mean()),
                "observed_frequency": float((idx[chunk] == k).mean()),
            })
    return pd.DataFrame(rows)


def expected_calibration_error(y_true: Sequence[str], proba, n_bins: int = 5) -> float:
    """Class-wise ECE: weighted mean |observed - predicted| over equal-count bins, averaged over H/D/A."""
    t = calibration_table(y_true, proba, n_bins)
    t["gap"] = (t["observed_frequency"] - t["mean_predicted"]).abs() * t["n"]
    per_class = t.groupby("class")["gap"].sum() / t.groupby("class")["n"].sum()
    return float(per_class.mean())


def evaluate_predictions(y_true: Sequence[str], proba) -> dict:
    return {
        "n": int(len(y_true)),
        "log_loss": multiclass_log_loss(y_true, proba),
        "brier": multiclass_brier(y_true, proba),
        "accuracy": accuracy(y_true, proba),
        "ece": expected_calibration_error(y_true, proba),
    }


def paired_difference(losses_a: np.ndarray, losses_b: np.ndarray) -> tuple[float, float]:
    """Mean and standard error of per-match ``a - b`` (negative => ``a`` is better)."""
    d = np.asarray(losses_a) - np.asarray(losses_b)
    if len(d) < 2:
        return float(d.mean()), float("nan")
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


# --------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------
@dataclass
class ChronologicalSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_seasons: list[int]
    val_seasons: list[int]
    test_seasons: list[int]
    partial_seasons_excluded: list[int] = field(default_factory=list)

    def describe(self) -> pd.DataFrame:
        rows = []
        for name, df, seasons in (("train", self.train, self.train_seasons), ("validation", self.val, self.val_seasons),
                                  ("test", self.test, self.test_seasons)):
            rows.append({
                "split": name, "seasons": ", ".join(f"{s}/{str(s + 1)[-2:]}" for s in seasons),
                "start": df["utc_date"].min(), "end": df["utc_date"].max(), "n_matches": len(df),
            })
        return pd.DataFrame(rows)


def completed_before(features: pd.DataFrame, cutoff: pd.Timestamp | None) -> pd.DataFrame:
    df = features[features["is_completed"]]
    if cutoff is not None:
        df = df[df["utc_date"] < pd.Timestamp(cutoff)]
    return df


def make_chronological_split(features: pd.DataFrame, n_val_seasons: int = 2, n_test_seasons: int = 2,
                             min_season_matches: int = 100, min_train_seasons: int = 3,
                             cutoff: pd.Timestamp | None = None) -> ChronologicalSplit:
    """Last ``n_test`` full seasons -> test, the ``n_val`` before them -> validation, everything older -> train.

    A season with fewer than ``min_season_matches`` completed matches before ``cutoff`` is "partial"
    (the in-progress season). It is excluded from every split here -- it is *later* than the test
    seasons -- but is still used when the final model is fitted.
    """
    done = completed_before(features, cutoff)
    counts = done.groupby("season").size()
    full = sorted(int(s) for s in counts[counts >= min_season_matches].index)
    partial = sorted(int(s) for s in counts[counts < min_season_matches].index)
    need = n_val_seasons + n_test_seasons + min_train_seasons
    if len(full) < need:
        raise InsufficientDataError(
            f"need >= {need} complete seasons for a {min_train_seasons}+{n_val_seasons}+{n_test_seasons} "
            f"train/val/test split, found {len(full)} ({full})"
        )
    test_s = full[-n_test_seasons:]
    val_s = full[-(n_test_seasons + n_val_seasons):-n_test_seasons]
    train_s = [s for s in full if s < val_s[0]]
    pick = lambda seasons: done[done["season"].isin(seasons)]  # noqa: E731
    return ChronologicalSplit(pick(train_s), pick(val_s), pick(test_s), train_s, val_s, test_s,
                              [s for s in partial if s > test_s[-1]])


# --------------------------------------------------------------------------------------
# Model comparison
# --------------------------------------------------------------------------------------
ModelFactory = Callable[[], OutcomeModel]


HORIZON_ONE_STEP = "one_step_ahead"   # features computed just before each match (normal forecasting)
HORIZON_FROZEN = "frozen_state"       # features frozen at the start of the season (how the tournament simulator predicts)
HorizonFrames = Mapping[str, "pd.DataFrame | None"]  # horizon -> feature rows to predict from (None = ordinary features)


def _eval_rows(part: pd.DataFrame, frame: pd.DataFrame | None) -> pd.DataFrame:
    """The rows of ``frame`` for the same matches as ``part`` (same labels, different feature values)."""
    if frame is None:
        return part
    return frame.set_index("match_id").loc[part["match_id"]].reset_index()


@dataclass
class SplitEvaluation:
    metrics: pd.DataFrame                                  # one row per (horizon, model, split)
    val_losses: dict[str, dict[str, np.ndarray]]           # horizon -> model -> per-match validation log loss
    predictions: dict[str, dict[str, dict[str, np.ndarray]]]  # horizon -> model -> split -> probabilities
    fitted: dict[str, OutcomeModel]


def evaluate_models_on_split(split: ChronologicalSplit, factories: Mapping[str, ModelFactory],
                             horizons: HorizonFrames | None = None) -> SplitEvaluation:
    """Fit every model once on the training period, then score validation and test at each prediction horizon."""
    horizons = horizons or {HORIZON_ONE_STEP: None}
    rows, val_losses, preds, fitted = [], {h: {} for h in horizons}, {h: {} for h in horizons}, {}
    for name, make in factories.items():
        model = make().fit(split.train)
        fitted[name] = model
        for h, frame in horizons.items():
            preds[h][name] = {}
            parts = [("validation", split.val), ("test", split.test)]
            if frame is None:  # in-sample numbers only make sense with the ordinary features
                parts.insert(0, ("train", split.train))
            for label, part in parts:
                eval_part = _eval_rows(part, frame)
                proba = model.predict_proba(eval_part)
                preds[h][name][label] = proba
                m = evaluate_predictions(eval_part["result"].tolist(), proba)
                rows.append({"horizon": h, "model": name, "split": label, "start": part["utc_date"].min(),
                             "end": part["utc_date"].max(), **m})
                if label == "validation":
                    val_losses[h][name] = per_match_log_loss(eval_part["result"].tolist(), proba)
    return SplitEvaluation(pd.DataFrame(rows), val_losses, preds, fitted)


@dataclass
class RollingResult:
    folds: pd.DataFrame                    # one row per (model, test season)
    losses: dict[str, pd.DataFrame]        # model -> per-match log loss, probabilities and outcome with its season (aligned across models)

    def pooled_predictions(self, model: str, max_season: int | None = None) -> tuple[list[str], np.ndarray]:
        """Outcomes and out-of-sample probabilities of ``model`` pooled over the rolling-origin folds."""
        d = self.losses[model]
        if max_season is not None:
            d = d[d["season"] <= max_season]
        return d["y"].tolist(), d[["pH", "pD", "pA"]].to_numpy()


def rolling_origin_evaluation(features: pd.DataFrame, factories: Mapping[str, ModelFactory], *, min_train_seasons: int = 5,
                              min_season_matches: int = 100, cutoff: pd.Timestamp | None = None,
                              horizons: HorizonFrames | None = None) -> dict[str, RollingResult]:
    """Expanding-window evaluation: for each full season ``s``, train on all seasons before ``s``, test on ``s``.

    Models are always trained on ``features`` (ordinary one-step-ahead rows). For each prediction horizon the *test* rows
    come from ``horizons[h]`` (``None`` = the ordinary rows; the frozen-state rows from
    :func:`src.features.build_frozen_features` measure performance at the simulator's horizon).
    """
    horizons = horizons or {HORIZON_ONE_STEP: None}
    done = completed_before(features, cutoff)
    counts = done.groupby("season").size()
    full = sorted(int(s) for s in counts[counts >= min_season_matches].index)
    rows = {h: [] for h in horizons}
    losses = {h: {name: [] for name in factories} for h in horizons}
    for i, season in enumerate(full):
        if i < min_train_seasons:
            continue
        train = done[done["season"].isin(full[:i])]
        test = done[done["season"] == season]
        for name, make in factories.items():
            model = make().fit(train)
            for h, frame in horizons.items():
                eval_test = _eval_rows(test, frame)
                proba = model.predict_proba(eval_test)
                m = evaluate_predictions(eval_test["result"].tolist(), proba)
                rows[h].append({"model": name, "test_season": season, "n_train": len(train), **m})
                losses[h][name].append(pd.DataFrame({
                    "season": season, "loss": per_match_log_loss(eval_test["result"].tolist(), proba),
                    "pH": proba[:, 0], "pD": proba[:, 1], "pA": proba[:, 2], "y": eval_test["result"].to_numpy()}))
    return {h: RollingResult(pd.DataFrame(rows[h]), {k: pd.concat(v, ignore_index=True) for k, v in losses[h].items() if v})
            for h in horizons}


def pooled_losses(rolling: RollingResult, max_season: int) -> dict[str, np.ndarray]:
    """Per-match losses of every rolling-origin fold whose test season is <= ``max_season``.

    Used for model selection: the folds are all *earlier* than the final test seasons, each predicted by a
    model that never saw it, and together they hold ~4x more matches than a single validation window.
    """
    return {m: d.loc[d["season"] <= max_season, "loss"].to_numpy() for m, d in rolling.losses.items()}


def summarise_rolling(rolling: pd.DataFrame, max_season: int | None = None) -> pd.DataFrame:
    """Match-weighted pooled metrics per model across rolling-origin test seasons (optionally up to ``max_season``)."""
    if max_season is not None:
        rolling = rolling[rolling["test_season"] <= max_season]
    if rolling.empty:
        return rolling
    rows = []
    for model, g in rolling.groupby("model"):
        w = g["n"] / g["n"].sum()
        rows.append({"model": model, "n_test_matches": int(g["n"].sum()), "n_folds": len(g),
                     **{k: float((g[k] * w).sum()) for k in ("log_loss", "brier", "accuracy", "ece")}})
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


@dataclass
class Selection:
    chosen: str
    best_by_validation: str
    rule: str
    table: pd.DataFrame
    notes: list[str]


def select_model(losses: Mapping[str, np.ndarray], complexity: Mapping[str, int], candidates: Sequence[str],
                 rule: str = "one_se") -> Selection:
    """Pick a model from per-match log losses of pre-test data only (never the test period).

    ``rule='best'``   -> lowest mean validation log loss.
    ``rule='one_se'`` -> the simplest candidate whose paired validation loss is within one standard error of
                         the best (guards against crowning a complex model for a noise-level gain).
    """
    cands = [c for c in candidates if c in losses]
    if not cands:
        raise ValueError("no selectable candidate models were evaluated")
    means = {c: float(losses[c].mean()) for c in cands}
    best = min(cands, key=means.get)
    rows = []
    for c in cands:
        diff, se = paired_difference(losses[c], losses[best]) if c != best else (0.0, 0.0)
        rows.append({"model": c, "selection_log_loss": means[c], "diff_vs_best": diff, "se_of_diff": se,
                     "within_1se_of_best": bool(c == best or diff <= se), "complexity": complexity.get(c, 99)})
    table = pd.DataFrame(rows).sort_values("selection_log_loss").reset_index(drop=True)
    chosen = best
    if rule == "one_se":
        ok = table[table["within_1se_of_best"]].sort_values(["complexity", "selection_log_loss"])
        chosen = str(ok.iloc[0]["model"])
    elif rule != "best":
        raise ValueError(f"unknown selection rule {rule!r}")
    notes = []
    if chosen != best:
        notes.append(f"'{best}' has the lowest selection log loss, but '{chosen}' is within one standard error "
                     f"of it and simpler, so it was chosen (one-standard-error rule).")
    n_sel = len(losses[best])
    close = table[(table["model"] != best) & table["within_1se_of_best"]]
    if len(close):
        notes.append(f"With {n_sel} selection matches, {list(close['model'])} cannot be statistically "
                     f"distinguished from '{best}' (difference within one standard error).")
    return Selection(chosen, best, rule, table, notes)


def sample_size_caveats(split: ChronologicalSplit) -> list[str]:
    notes = [
        f"train={len(split.train)}, validation={len(split.val)}, test={len(split.test)} matches. A multiclass "
        "log-loss standard error on a few hundred matches is roughly 0.03-0.04, so model differences smaller "
        "than that are within noise."
    ]
    if len(split.test) < 500:
        notes.append("The test period has fewer than 500 matches: treat its metrics as a rough health check, "
                     "not a precise ranking of models.")
    notes.append("Calibration bins hold only a few dozen matches each; reliability curves are indicative only.")
    return notes
