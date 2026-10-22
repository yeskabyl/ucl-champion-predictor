"""Probability handling, models, metrics, chronological splits and model selection."""
import numpy as np
import pandas as pd
import pytest

from src import evaluation as ev
from src.features import build_features, build_frozen_features
from src.models import (
    CatBoostOutcomeModel, ClimatologyBaseline, EloBaseline, LogisticOutcomeModel, align_class_probabilities,
    catboost_available, make_model_factories, normalize_probabilities, outcome_probs_from_rates, rates_from_probabilities,
)
from src.utils import CLASSES

from .helpers import random_league


@pytest.fixture(scope="module")
def table():
    """8 synthetic seasons x 120 matches, with sequential and frozen-state features."""
    matches = random_league(n_teams=8, n_rounds=30, seed=21, seasons=8)
    feats, builder = build_features(matches)
    return feats, build_frozen_features(matches)


# ------------------------------------------------------------------ probability hygiene
def test_normalisation_makes_rows_sum_to_one_and_positive():
    p = normalize_probabilities([[2.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    assert p.sum(axis=1) == pytest.approx([1.0, 1.0]) and (p > 0).all()
    assert p[0] == pytest.approx([0.5, 0.25, 0.25])


def test_normalisation_accepts_a_single_row_and_refuses_garbage():
    assert normalize_probabilities([0.2, 0.3, 0.5]).shape == (1, 3)
    for bad in ([[np.nan, 0.5, 0.5]], [[np.inf, 0.0, 0.0]]):
        with pytest.raises(ValueError, match="NaN"):
            normalize_probabilities(bad)
    with pytest.raises(ValueError, match="shape"):
        normalize_probabilities([[0.5, 0.5]])


def test_class_alignment_reorders_alphabetical_sklearn_columns():
    out = align_class_probabilities(np.array([[0.2, 0.3, 0.5]]), ["A", "D", "H"])  # sklearn sorts classes: A, D, H
    assert out[0] == pytest.approx([0.5, 0.3, 0.2])  # -> H, D, A


def test_a_class_missing_from_training_gets_a_tiny_but_positive_probability():
    out = align_class_probabilities(np.array([[0.4, 0.6]]), ["A", "H"])
    assert out.shape == (1, 3) and out[0, CLASSES.index("D")] > 0 and out[0, CLASSES.index("D")] < 1e-3
    assert out.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unexpected class"):
        align_class_probabilities(np.array([[1.0]]), ["X"])


def test_logistic_regression_survives_training_data_without_draws(table):
    feats, _ = table
    no_draws = feats[(feats["result"] != "D") & feats["is_completed"]]
    model = LogisticOutcomeModel().fit(no_draws)
    p = model.predict_proba(feats[feats["is_completed"]].head(50))
    assert np.isfinite(p).all() and p.sum(axis=1) == pytest.approx(np.ones(50))
    assert (p[:, 1] > 0).all() and p[:, 1].max() < 0.01


# ------------------------------------------------------------------ individual models
def test_every_model_returns_valid_probabilities_and_rates(table):
    feats, _ = table
    train, test = feats[feats["season"] < 2023], feats[feats["season"] >= 2023]
    for name, make in make_model_factories(seed=1, include_catboost=True).items():
        model = make().fit(train)
        p = model.predict_proba(test)
        assert p.shape == (len(test), 3) and np.isfinite(p).all() and (p > 0).all(), name
        assert p.sum(axis=1) == pytest.approx(np.ones(len(test))), name
        lam_h, lam_a = model.predict_goal_rates(test)
        assert (lam_h > 0).all() and (lam_a > 0).all() and lam_h.shape == (len(test),), name


def test_models_refuse_misuse(table):
    feats, _ = table
    with pytest.raises(RuntimeError, match="not fitted"):
        LogisticOutcomeModel().predict_proba(feats)
    with pytest.raises(ValueError, match="no completed"):
        LogisticOutcomeModel().fit(feats.iloc[0:0])
    model = LogisticOutcomeModel().fit(feats[feats["is_completed"]])
    with pytest.raises(KeyError, match="missing feature columns"):
        model.predict_proba(feats.drop(columns=["elo_diff"]))


def test_climatology_predicts_smoothed_training_frequencies(table):
    feats, _ = table
    done = feats[feats["is_completed"]]
    p = ClimatologyBaseline().fit(done).predict_proba(done.head(5))
    freq = done["result"].value_counts().reindex(CLASSES).to_numpy(float)
    expected = (freq + 1) / (freq.sum() + 3)
    assert p[0] == pytest.approx(expected) and (p == p[0]).all()


def test_elo_baseline_properties(table):
    feats, _ = table
    done = feats[feats["is_completed"]]
    model = EloBaseline().fit(done)
    even = pd.DataFrame({"elo_expected_home": [0.5]})
    strong = pd.DataFrame({"elo_expected_home": [0.9]})
    pe, ps = model.predict_proba(even)[0], model.predict_proba(strong)[0]
    assert pe[0] == pytest.approx(pe[2]) and ps[0] > 0.8 and ps[1] < pe[1]  # draws are likeliest between equals
    avg_draw = model.predict_proba(done)[:, 1].mean()
    assert avg_draw == pytest.approx((done["result"] == "D").mean(), abs=0.01)  # draw constant matches the training rate


def test_preprocessing_is_learnt_from_training_rows_only(table):
    feats, _ = table
    train = feats[(feats["season"] < 2022) & feats["is_completed"]]
    model = LogisticOutcomeModel().fit(train)
    learnt = model.pipe_.named_steps["impute"].statistics_
    np.testing.assert_allclose(learnt, train[model.feature_columns].median().to_numpy(), equal_nan=False)
    # a NaN at prediction time is replaced by the TRAINING median, not by anything from the new data
    row = feats[feats["season"] == 2024].head(1).copy()
    blank, filled = row.copy(), row.copy()
    blank["rest_diff"] = np.nan
    filled["rest_diff"] = train["rest_diff"].median()
    np.testing.assert_allclose(model.predict_proba(blank), model.predict_proba(filled))


@pytest.mark.skipif(not catboost_available(), reason="catboost not installed")
def test_catboost_is_reproducible_for_a_given_seed(table):
    feats, _ = table
    train, test = feats[feats["season"] < 2023], feats[feats["season"] >= 2023]
    a = CatBoostOutcomeModel(iterations=40, seed=7).fit(train).predict_proba(test)
    b = CatBoostOutcomeModel(iterations=40, seed=7).fit(train).predict_proba(test)
    np.testing.assert_allclose(a, b)


def test_goal_rates_from_probabilities_reproduce_the_supremacy_and_total():
    p = np.array([[0.55, 0.25, 0.20], [0.25, 0.25, 0.50], [0.80, 0.12, 0.08], [0.34, 0.30, 0.36]])
    lam_h, lam_a = rates_from_probabilities(p, total_goals=2.8)
    implied = outcome_probs_from_rates(lam_h, lam_a, max_goals=12)
    assert (implied[:, 0] - implied[:, 2]) == pytest.approx(p[:, 0] - p[:, 2], abs=1e-5)
    assert lam_h + lam_a == pytest.approx(np.full(4, 2.8), abs=1e-6)
    assert lam_h[2] > lam_a[2] and lam_h[1] < lam_a[1]


def test_equal_win_probabilities_give_equal_rates_and_outcome_probs_are_symmetric():
    lam_h, lam_a = rates_from_probabilities(np.array([[0.4, 0.2, 0.4]]))
    assert lam_h[0] == pytest.approx(lam_a[0], rel=1e-6)
    a = outcome_probs_from_rates(np.array([1.8]), np.array([0.9]))
    b = outcome_probs_from_rates(np.array([0.9]), np.array([1.8]))
    assert a[0] == pytest.approx(b[0][::-1]) and a.sum() == pytest.approx(1.0)


# ------------------------------------------------------------------ metrics
Y = ["H", "D", "A"]
P = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])


def test_log_loss_by_hand():
    assert ev.multiclass_log_loss(Y, P) == pytest.approx(np.mean([-np.log(0.5), -np.log(0.5), -np.log(0.8)]))


def test_brier_score_by_hand():
    expected = np.mean([(0.5 - 1) ** 2 + 0.3 ** 2 + 0.2 ** 2, 0.2 ** 2 + (0.5 - 1) ** 2 + 0.3 ** 2, 0.1 ** 2 + 0.1 ** 2 + (0.8 - 1) ** 2])
    assert ev.multiclass_brier(Y, P) == pytest.approx(expected)


def test_uniform_prediction_scores_the_textbook_values():
    u = np.full((3, 3), 1 / 3)
    assert ev.multiclass_log_loss(Y, u) == pytest.approx(np.log(3)) and ev.multiclass_brier(Y, u) == pytest.approx(2 / 3)


def test_accuracy_and_perfect_prediction():
    assert ev.accuracy(Y, P) == pytest.approx(1.0)
    wrong = P[::-1]
    assert ev.accuracy(Y, wrong) < 1.0
    perfect = np.eye(3)
    assert ev.multiclass_log_loss(Y, perfect) < 1e-3 and ev.multiclass_brier(Y, perfect) < 1e-6


def test_unknown_labels_are_rejected():
    with pytest.raises(ValueError, match="unknown outcome label"):
        ev.multiclass_log_loss(["H", "X"], P[:2])


def test_calibration_table_of_well_calibrated_forecasts_has_small_gaps():
    rng = np.random.default_rng(0)
    p = rng.dirichlet([4, 2, 3], size=30_000)
    y = [CLASSES[i] for i in (rng.random(len(p))[:, None] > p.cumsum(1)).sum(1)]
    table = ev.calibration_table(y, p, n_bins=5)
    assert (table["observed_frequency"] - table["mean_predicted"]).abs().max() < 0.03
    assert table.groupby("class")["n"].sum().tolist() == [30_000] * 3
    assert ev.expected_calibration_error(y, p) < 0.01


def test_calibration_error_flags_overconfidence():
    y = ["H"] * 50 + ["A"] * 50
    over = np.tile([0.9, 0.05, 0.05], (100, 1))  # claims 90% home wins; only 50% happen
    assert ev.expected_calibration_error(y, over) > 0.2


def test_paired_difference():
    d, se = ev.paired_difference(np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.5, 3.5]))
    assert d == pytest.approx(-0.5) and se == pytest.approx(0.0)


# ------------------------------------------------------------------ chronological splits
def test_split_is_strictly_chronological_and_disjoint(table):
    feats, _ = table
    split = ev.make_chronological_split(feats)
    assert split.train_seasons == [2018, 2019, 2020, 2021]
    assert split.val_seasons == [2022, 2023] and split.test_seasons == [2024, 2025]
    assert split.train["utc_date"].max() < split.val["utc_date"].min() <= split.val["utc_date"].max() < split.test["utc_date"].min()
    ids = [set(x["match_id"]) for x in (split.train, split.val, split.test)]
    assert not (ids[0] & ids[1]) and not (ids[1] & ids[2]) and not (ids[0] & ids[2])
    assert split.train["season"].max() < split.val["season"].min() and split.val["season"].max() < split.test["season"].min()
    desc = split.describe()
    assert desc["n_matches"].tolist() == [len(split.train), len(split.val), len(split.test)]


def test_split_sizes_follow_the_requested_season_counts(table):
    feats, _ = table
    split = ev.make_chronological_split(feats, n_val_seasons=1, n_test_seasons=3, min_train_seasons=2)
    assert split.test_seasons == [2023, 2024, 2025] and split.val_seasons == [2022] and split.train_seasons == [2018, 2019, 2020, 2021]


def test_in_progress_season_is_excluded_from_every_split_but_reported(table):
    feats, _ = table
    completed_seasons = feats[feats["season"] <= 2024]
    short = feats[feats["season"] == 2024].head(30).copy()  # an in-progress season: only 30 matches so far
    short["season"] = 2025
    short["match_id"] = "late-" + short["match_id"]
    short["utc_date"] = short["utc_date"] + pd.Timedelta(days=400)
    split = ev.make_chronological_split(pd.concat([completed_seasons, short], ignore_index=True))
    assert split.partial_seasons_excluded == [2025]
    assert split.test_seasons == [2023, 2024]
    assert 2025 not in set(split.train["season"]) | set(split.val["season"]) | set(split.test["season"])


def test_cutoff_removes_everything_after_it(table):
    feats, _ = table
    cutoff = feats[feats["season"] == 2023]["utc_date"].min()
    split = ev.make_chronological_split(feats, n_val_seasons=1, n_test_seasons=1, cutoff=cutoff)
    for part in (split.train, split.val, split.test):
        assert (part["utc_date"] < cutoff).all()
    assert split.test_seasons == [2022] and split.val_seasons == [2021]


def test_too_little_history_is_reported_explicitly():
    feats, _ = build_features(random_league(n_teams=8, n_rounds=30, seed=1, seasons=4))
    with pytest.raises(ev.InsufficientDataError, match="need >= 7 complete seasons"):
        ev.make_chronological_split(feats)


class SpyModel(ClimatologyBaseline):
    name = "spy"
    seen_train_seasons: list = []

    def _fit(self, train):
        SpyModel.seen_train_seasons.append((int(train["season"].min()), int(train["season"].max())))
        super()._fit(train)


def test_rolling_origin_never_trains_on_the_season_it_tests(table):
    feats, frozen = table
    SpyModel.seen_train_seasons = []
    res = ev.rolling_origin_evaluation(feats, {"spy": SpyModel}, min_train_seasons=3,
                                       horizons={ev.HORIZON_ONE_STEP: None, ev.HORIZON_FROZEN: frozen})
    folds = res[ev.HORIZON_ONE_STEP].folds
    assert folds["test_season"].tolist() == [2021, 2022, 2023, 2024, 2025]
    assert len(SpyModel.seen_train_seasons) == len(folds)  # one fit per fold, shared by both horizons
    for (lo, hi), test_season in zip(SpyModel.seen_train_seasons, folds["test_season"]):
        assert lo == 2018 and hi < test_season  # strictly older data only, expanding window
    assert set(res) == {ev.HORIZON_ONE_STEP, ev.HORIZON_FROZEN}


def test_cutoff_is_respected_by_rolling_origin(table):
    feats, _ = table
    cutoff = feats[feats["season"] == 2024]["utc_date"].min()
    res = ev.rolling_origin_evaluation(feats, {"clim": ClimatologyBaseline}, min_train_seasons=3, cutoff=cutoff)
    assert res[ev.HORIZON_ONE_STEP].folds["test_season"].max() == 2023


def test_evaluating_on_a_split_reports_every_model_horizon_and_split(table):
    feats, frozen = table
    split = ev.make_chronological_split(feats)
    out = ev.evaluate_models_on_split(split, {"clim": ClimatologyBaseline, "elo": EloBaseline},
                                      {ev.HORIZON_ONE_STEP: None, ev.HORIZON_FROZEN: frozen})
    m = out.metrics
    assert set(m["horizon"]) == {ev.HORIZON_ONE_STEP, ev.HORIZON_FROZEN}
    assert set(m[m["horizon"] == ev.HORIZON_ONE_STEP]["split"]) == {"train", "validation", "test"}
    assert set(m[m["horizon"] == ev.HORIZON_FROZEN]["split"]) == {"validation", "test"}  # no meaningless in-sample numbers
    assert (m["n"] > 0).all() and m["log_loss"].between(0, 5).all()
    assert out.predictions[ev.HORIZON_FROZEN]["elo"]["test"].shape == (len(split.test), 3)


def test_frozen_horizon_scores_the_same_matches_with_different_features(table):
    feats, frozen = table
    part = feats[feats["season"] == 2024]
    rows = ev._eval_rows(part, frozen)
    assert rows["match_id"].tolist() == part["match_id"].tolist()
    assert (rows["result"].tolist() == part["result"].tolist())
    assert not np.allclose(rows["elo_diff"], part["elo_diff"])


def test_pooled_summary_is_match_weighted():
    folds = pd.DataFrame({"model": ["m", "m"], "test_season": [2020, 2021], "n": [100, 300], "log_loss": [1.0, 0.8],
                          "brier": [0.6, 0.5], "accuracy": [0.4, 0.5], "ece": [0.1, 0.05]})
    pooled = ev.summarise_rolling(folds)
    assert pooled.loc[0, "log_loss"] == pytest.approx(0.85) and pooled.loc[0, "n_test_matches"] == 400
    assert ev.summarise_rolling(folds, max_season=2020).loc[0, "n_test_matches"] == 100


# ------------------------------------------------------------------ selection
def _losses(seed=0, n=400):
    """'complex' is nominally better than 'simple' by 0.002 -- a gap smaller than its standard error (~0.0025)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(1.0, 0.5, n)
    d = rng.normal(0.0, 0.05, n)
    d = d - d.mean() + 0.002  # exact mean gap, so the test does not depend on luck
    return {"simple": base + d, "complex": base.copy(), "bad": base + 0.2 + rng.normal(0, 0.05, n)}


def test_one_standard_error_rule_prefers_the_simpler_model_when_gains_are_noise():
    losses = _losses()
    cx = {"simple": 1, "complex": 5, "bad": 3}
    assert losses["complex"].mean() < losses["simple"].mean()  # complex is nominally better...
    best = ev.select_model(losses, cx, ["simple", "complex", "bad"], rule="best")
    assert best.chosen == "complex"
    one_se = ev.select_model(losses, cx, ["simple", "complex", "bad"], rule="one_se")
    assert one_se.chosen == "simple" and any("simpler" in n for n in one_se.notes)  # ...but not distinguishably so
    assert "bad" not in one_se.table.loc[one_se.table["within_1se_of_best"], "model"].tolist()


def test_selection_does_not_pick_a_clearly_worse_simple_model():
    losses = _losses()
    losses["simple"] = losses["simple"] + 0.5
    assert ev.select_model(losses, {"simple": 1, "complex": 5, "bad": 3}, ["simple", "complex"], "one_se").chosen == "complex"


def test_selection_input_validation():
    with pytest.raises(ValueError, match="no selectable"):
        ev.select_model({"a": np.ones(3)}, {"a": 1}, ["zzz"])
    with pytest.raises(ValueError, match="unknown selection rule"):
        ev.select_model({"a": np.ones(3)}, {"a": 1}, ["a"], rule="vibes")


def test_climatology_is_excluded_from_candidates_by_the_pipeline_but_is_the_yardstick(table):
    feats, frozen = table
    out = ev.rolling_origin_evaluation(feats, make_model_factories(include_catboost=False), min_train_seasons=3,
                                       horizons={ev.HORIZON_FROZEN: frozen})[ev.HORIZON_FROZEN]
    pooled = ev.summarise_rolling(out.folds)
    assert {"climatology", "elo", "logistic", "poisson"} <= set(pooled["model"])
    assert np.isfinite(pooled["log_loss"]).all()
