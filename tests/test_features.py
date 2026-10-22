"""Chronology, leakage, missing/duplicated data and identity handling for the feature pipeline."""
import numpy as np
import pandas as pd
import pytest

from src.elo import EloConfig
from src.features import FEATURE_COLUMNS, FeatureBuilder, build_features, build_frozen_features
from src.utils import canonical_team_key, find_possible_duplicate_teams, strip_country_suffix, team_id_from_name

from .helpers import make_matches, random_league


def _line(goals):  # one team 'a' playing a different opponent each week, scoring/conceding as given
    rows = []
    for i, (gf, ga) in enumerate(goals):
        rows.append({"home": "a", "away": f"opp{i}", "hg": gf, "ag": ga, "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=7 * i)})
    return make_matches(rows)


# ---------------------------------------------------------------- chronological generation
def test_first_match_has_no_history():
    feats, _ = build_features(_line([(2, 0), (1, 1)]))
    first = feats.iloc[0]
    assert np.isnan(first["gf5_home"]) and np.isnan(first["ga10_home"]) and np.isnan(first["rest_days_home"])
    assert first["n_prev_home"] == 0 and first["elo_home"] == 1500.0


def test_rolling_means_use_only_strictly_previous_matches():
    goals = [(1, 0), (2, 1), (3, 2), (0, 0), (4, 1), (2, 2), (1, 3)]
    feats, _ = build_features(_line(goals))
    for k in range(1, len(goals)):
        prev = goals[max(0, k - 5):k]
        assert feats.loc[k, "gf5_home"] == pytest.approx(np.mean([g[0] for g in prev]))
        assert feats.loc[k, "ga5_home"] == pytest.approx(np.mean([g[1] for g in prev]))
        prev10 = goals[max(0, k - 10):k]
        assert feats.loc[k, "gf10_home"] == pytest.approx(np.mean([g[0] for g in prev10]))
    # the current match must not be in its own window:
    assert feats.loc[6, "gf5_home"] != pytest.approx(np.mean([g[0] for g in goals[2:7]]))


def test_points_form_is_three_one_nil():
    feats, _ = build_features(_line([(2, 0), (1, 1), (0, 1), (5, 0)]))
    assert feats.loc[3, "pts5_home"] == pytest.approx((3 + 1 + 0) / 3)


def test_rest_days_and_cap():
    m = make_matches([
        {"home": "a", "away": "b", "hg": 1, "ag": 0, "date": "2020-01-01"},
        {"home": "a", "away": "c", "hg": 1, "ag": 0, "date": "2020-01-04"},
        {"home": "a", "away": "d", "hg": 1, "ag": 0, "date": "2020-06-01"},
    ])
    feats, _ = build_features(m)
    assert feats.loc[1, "rest_days_home"] == pytest.approx(3.0)
    assert feats.loc[2, "rest_days_home"] == 30.0  # capped
    assert np.isnan(feats.loc[1, "rest_days_away"])  # 'c' is new


def test_opponent_strength_is_the_opponents_rating_at_that_time():
    m = make_matches([
        {"home": "b", "away": "x", "hg": 4, "ag": 0, "date": "2020-01-01"},   # b becomes strong
        {"home": "a", "away": "b", "hg": 0, "ag": 0, "date": "2020-01-08"},   # a meets strong b
        {"home": "a", "away": "c", "hg": 1, "ag": 0, "date": "2020-01-15"},
    ])
    feats, builder = build_features(m, EloConfig(home_advantage=0))
    b_rating_when_met = feats.loc[1, "elo_away"]
    assert b_rating_when_met > 1500
    assert feats.loc[2, "opp_elo5_home"] == pytest.approx(b_rating_when_met)  # not b's *later* rating


def test_elo_features_are_recorded_before_the_update():
    m = _line([(3, 0), (0, 2)])
    feats, builder = build_features(m, EloConfig(home_advantage=0, season_regression=0.0))
    assert feats.loc[0, "elo_home"] == 1500.0 and feats.loc[0, "elo_away"] == 1500.0
    assert feats.loc[1, "elo_home"] > 1500.0       # carries match 0, but not match 1's own result
    assert builder.elo.rating("a") < feats.loc[1, "elo_home"]  # final state includes match 1 (a lost)


def test_labels_and_flags():
    m = make_matches([
        {"home": "a", "away": "b", "hg": 2, "ag": 1, "date": "2020-01-01"},
        {"home": "b", "away": "a", "hg": 0, "ag": 0, "date": "2020-01-08", "stage": "LAST_16"},
        {"home": "a", "away": "b", "hg": 0, "ag": 3, "date": "2020-06-01", "stage": "FINAL"},
    ])
    feats, _ = build_features(m)
    assert feats["result"].tolist() == ["H", "D", "A"]
    assert feats["neutral"].tolist() == [0.0, 0.0, 1.0]
    assert feats["is_knockout"].tolist() == [0.0, 1.0, 1.0]
    assert feats.loc[2, "elo_expected_home"] == pytest.approx(1 / (1 + 10 ** ((feats.loc[2, "elo_away"] - feats.loc[2, "elo_home"]) / 400)))


# ---------------------------------------------------------------- leakage
def test_changing_a_matchs_own_result_never_changes_its_own_features():
    base = random_league(seed=11)
    feats0, _ = build_features(base)
    for i in (0, 5, 17, len(base) - 1):
        tweaked = base.copy()
        tweaked.loc[i, ["home_goals_90", "home_goals_aet"]] = base.loc[i, "home_goals_90"] + 4
        feats1, _ = build_features(tweaked)
        pd.testing.assert_series_equal(feats0.loc[i, FEATURE_COLUMNS], feats1.loc[i, FEATURE_COLUMNS], check_names=False)


def test_future_results_cannot_leak_into_earlier_rows():
    base = random_league(n_teams=8, n_rounds=25, seed=5)
    cut = len(base) // 2
    feats0, _ = build_features(base)
    tweaked = base.copy()
    tweaked.loc[cut:, ["home_goals_90", "away_goals_90", "home_goals_aet", "away_goals_aet"]] = 9  # rewrite the whole future
    feats1, _ = build_features(tweaked)
    pd.testing.assert_frame_equal(feats0.iloc[:cut][FEATURE_COLUMNS], feats1.iloc[:cut][FEATURE_COLUMNS])
    assert not feats0.iloc[cut:][FEATURE_COLUMNS].equals(feats1.iloc[cut:][FEATURE_COLUMNS])  # ...and does change later rows


def test_truncating_history_gives_identical_features_for_the_kept_rows():
    base = random_league(n_teams=6, n_rounds=20, seed=2)
    full, _ = build_features(base)
    part, _ = build_features(base.iloc[:30])
    pd.testing.assert_frame_equal(full.iloc[:30][FEATURE_COLUMNS], part[FEATURE_COLUMNS])


def test_row_order_of_the_input_does_not_matter():
    base = random_league(seed=8)
    shuffled = base.sample(frac=1.0, random_state=1)
    a, _ = build_features(base)
    b, _ = build_features(shuffled)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_unplayed_fixtures_get_features_but_never_change_state():
    played = random_league(seed=4)
    fixture = make_matches([{"home": "t0", "away": "t1", "hg": None, "ag": None, "date": "2018-09-02", "completed": False,
                             "match_id": "fx"}])
    mixed = pd.concat([played, fixture], ignore_index=True)
    f_mixed, b_mixed = build_features(mixed)
    f_played, b_played = build_features(played)
    assert b_mixed.elo.snapshot() == b_played.elo.snapshot()
    row = f_mixed[f_mixed["match_id"] == "fx"].iloc[0]
    assert not row["is_completed"] and pd.isna(row["result"]) and np.isnan(row["home_goals_90"])
    assert np.isfinite(row["elo_diff"])
    others = f_mixed[f_mixed["match_id"] != "fx"].reset_index(drop=True)
    pd.testing.assert_frame_equal(others[FEATURE_COLUMNS], f_played[FEATURE_COLUMNS])  # fixture did not disturb anyone


def test_same_timestamp_matches_are_ordered_deterministically():
    rows = [{"home": "a", "away": "b", "hg": 1, "ag": 0, "date": "2020-01-01 20:00", "match_id": "m2"},
            {"home": "c", "away": "d", "hg": 0, "ag": 2, "date": "2020-01-01 20:00", "match_id": "m1"}]
    f1, _ = build_features(make_matches(rows))
    f2, _ = build_features(make_matches(rows[::-1]))
    assert f1["match_id"].tolist() == f2["match_id"].tolist() == ["m1", "m2"]


# ---------------------------------------------------------------- frozen-state features
def test_frozen_features_hold_the_season_start_state_for_the_whole_season():
    base = random_league(n_teams=6, n_rounds=8, seed=6, seasons=2)
    seq, _ = build_features(base)
    frozen = build_frozen_features(base)
    s1 = frozen[frozen["season"] == 2019].reset_index(drop=True)
    s0_end_state = build_features(base[base["season"] == 2018])[1]
    first = s1.iloc[0]
    for _, row in s1.iterrows():  # same team => same frozen rating for every match of the season
        for side, col in (("home_team_id", "elo_home"), ("away_team_id", "elo_away")):
            expected = s0_end_state.elo.rating_for_season(row[side], 2019)
            assert row[col] == pytest.approx(expected)
    # ...whereas the ordinary features of later matches have moved on
    seq1 = seq[seq["season"] == 2019].reset_index(drop=True)
    assert not np.allclose(seq1["elo_home"], s1["elo_home"])
    assert first["n_prev_home"] >= 0 and s1["rest_days_home"].isna().all()


def test_frozen_features_never_use_the_seasons_own_results():
    base = random_league(n_teams=6, n_rounds=8, seed=7, seasons=2)
    tweaked = base.copy()
    mask = tweaked["season"] == 2019
    tweaked.loc[mask, ["home_goals_90", "away_goals_90", "home_goals_aet", "away_goals_aet"]] = 7
    a = build_frozen_features(base)
    b = build_frozen_features(tweaked)
    pd.testing.assert_frame_equal(a[a["season"] == 2019][FEATURE_COLUMNS], b[b["season"] == 2019][FEATURE_COLUMNS])


# ---------------------------------------------------------------- missing / duplicated data
def test_duplicate_match_ids_are_rejected():
    m = make_matches([{"home": "a", "away": "b", "hg": 1, "ag": 0, "date": "2020-01-01", "match_id": "dup"},
                      {"home": "c", "away": "d", "hg": 1, "ag": 0, "date": "2020-01-02", "match_id": "dup"}])
    with pytest.raises(ValueError, match="duplicate"):
        build_features(m)


def test_rows_with_an_undetermined_team_are_skipped_not_guessed():
    m = make_matches([{"home": "a", "away": "b", "hg": 1, "ag": 0, "date": "2020-01-01"},
                      {"home": None, "away": "b", "hg": None, "ag": None, "date": "2020-02-01", "completed": False,
                       "stage": "LAST_16"}])
    feats, _ = build_features(m)
    assert len(feats) == 1


def test_a_finished_match_without_a_score_is_not_treated_as_played():
    m = make_matches([{"home": "a", "away": "b", "hg": 1, "ag": 0, "date": "2020-01-01"},
                      {"home": "a", "away": "c", "hg": 2, "ag": 2, "date": "2020-01-08"}])
    m.loc[1, "away_goals_90"] = np.nan  # claims completed but the score is missing
    feats, builder = build_features(m)
    assert bool(feats.loc[1, "is_completed"]) is False and pd.isna(feats.loc[1, "result"])
    assert builder.n_updates == 1


def test_builder_features_for_is_side_effect_free():
    builder = FeatureBuilder()
    builder.update("a", "b", 2, 0, pd.Timestamp("2020-01-01"), season=2020)
    before = builder.elo.snapshot(), builder.n_updates
    builder.features_for("a", "z", pd.Timestamp("2020-02-01"), season=2021)
    assert (builder.elo.snapshot(), builder.n_updates) == before


# ---------------------------------------------------------------- team identity
@pytest.mark.parametrize("a,b", [
    ("Inter", "FC Internazionale Milano"), ("Bayern München", "FC Bayern München"),
    ("Atlético Madrid", "Club Atlético de Madrid"), ("Qarabağ FK", "Qarabağ Ağdam FK"),
    ("Manchester City", "Manchester City FC (ENG)"), ("RB Salzburg", "FC Red Bull Salzburg"),
    ("SL Benfica", "Sport Lisboa e Benfica"), ("Olympiakos Piraeus", "PAE Olympiakos SFP"),
    ("Bayer Leverkusen", "Bayer 04 Leverkusen"), ("FC Nordsjælland", "Nordsjaelland"),
])
def test_name_variants_share_one_identity(a, b):
    assert team_id_from_name(a) == team_id_from_name(b)


@pytest.mark.parametrize("a,b", [("Dinamo Zagreb", "Dynamo Kyiv"), ("Union Berlin", "Union Saint-Gilloise"),
                                 ("Sparta Praha", "Slavia Praha"), ("Real Madrid", "Real Sociedad")])
def test_different_clubs_stay_different(a, b):
    assert team_id_from_name(a) != team_id_from_name(b)


def test_identity_helpers_edge_cases():
    assert strip_country_suffix("Arsenal FC (ENG)") == "Arsenal FC"
    assert team_id_from_name("  Arsenal FC  ") == "arsenal"
    assert canonical_team_key("FC") == "fc"  # only stop-words: falls back to the raw text instead of an empty key
    with pytest.raises(ValueError):
        team_id_from_name("   ")
    assert find_possible_duplicate_teams(["bayern-munchen", "bayern-munich", "arsenal"])[0][:2] == ("bayern-munchen", "bayern-munich")
