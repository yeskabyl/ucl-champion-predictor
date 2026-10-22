import numpy as np
import pytest

from src.elo import EloConfig, EloRatingSystem, actual_score, goal_difference_multiplier, run_elo

from .helpers import make_matches, random_league


def test_equal_ratings_neutral_is_a_coin_flip():
    elo = EloRatingSystem()
    assert elo.expected_home(1500, 1500, neutral=True) == pytest.approx(0.5)


def test_home_advantage_favours_the_home_side_only_when_not_neutral():
    elo = EloRatingSystem(EloConfig(home_advantage=65))
    assert elo.expected_home(1500, 1500, neutral=False) > 0.5
    assert elo.expected_home(1500, 1500, neutral=True) == pytest.approx(0.5)


def test_expectation_is_monotonic_and_complementary():
    elo = EloRatingSystem(EloConfig(home_advantage=0))
    e = [elo.expected_home(1500 + d, 1500) for d in (-200, -50, 0, 50, 200)]
    assert e == sorted(e)
    assert elo.expected_home(1600, 1500) + elo.expected_home(1500, 1600) == pytest.approx(1.0)


def test_400_points_is_ten_to_one_odds():
    elo = EloRatingSystem(EloConfig(home_advantage=0))
    assert elo.expected_home(1900, 1500) == pytest.approx(10 / 11)


@pytest.mark.parametrize("gd,expected", [(0, 1.0), (1, 1.0), (2, 1.5), (3, 1.75), (4, 15 / 8), (-3, 1.75)])
def test_goal_difference_multiplier(gd, expected):
    assert goal_difference_multiplier(gd) == pytest.approx(expected)


def test_actual_score():
    assert (actual_score(2, 1), actual_score(1, 1), actual_score(0, 3)) == (1.0, 0.5, 0.0)


def test_update_is_zero_sum_and_moves_the_right_way():
    elo = EloRatingSystem()
    upd = elo.update("a", "b", 2, 0)
    assert upd.delta_home > 0
    assert elo.rating("a") + elo.rating("b") == pytest.approx(3000.0)
    assert elo.rating("a") > 1500 > elo.rating("b")


def test_bigger_wins_move_ratings_more():
    small, big = EloRatingSystem(), EloRatingSystem()
    small.update("a", "b", 1, 0)
    big.update("a", "b", 4, 0)
    assert big.rating("a") - 1500 > small.rating("a") - 1500


def test_goal_difference_can_be_switched_off():
    elo = EloRatingSystem(EloConfig(use_goal_difference=False))
    a, b = EloRatingSystem(EloConfig(use_goal_difference=False)), EloRatingSystem(EloConfig(use_goal_difference=False))
    a.update("x", "y", 1, 0)
    b.update("x", "y", 5, 0)
    assert a.rating("x") == pytest.approx(b.rating("x"))
    assert elo.rating("x") == 1500


def test_neutral_draw_between_equals_changes_nothing_but_home_draw_costs_the_host():
    neutral = EloRatingSystem()
    neutral.update("a", "b", 1, 1, neutral=True)
    assert neutral.rating("a") == pytest.approx(1500.0)
    home = EloRatingSystem()
    home.update("a", "b", 1, 1, neutral=False)
    assert home.rating("a") < 1500 < home.rating("b")  # the host was expected to do better than a draw


def test_unseen_teams_start_at_the_initial_rating_and_reads_do_not_create_state():
    elo = EloRatingSystem(EloConfig(initial_rating=1400))
    assert elo.rating("never-seen") == 1400
    assert "never-seen" not in elo.ratings
    elo.update("a", "b", 1, 0)
    assert elo.rating("c") == 1400  # untouched teams are untouched


def test_update_rejects_bad_input():
    elo = EloRatingSystem()
    with pytest.raises(ValueError):
        elo.update("a", "a", 1, 0)
    with pytest.raises(ValueError):
        elo.update("a", "b", float("nan"), 0)
    with pytest.raises(ValueError):
        elo.update("a", "b", None, 0)


def test_season_regression_pulls_toward_the_mean_and_peek_matches_commit():
    cfg = EloConfig(season_regression=0.10)
    elo = EloRatingSystem(cfg, ratings={"strong": 1700.0, "weak": 1300.0}, season=2019)
    peek_strong = elo.rating_for_season("strong", 2020)
    assert elo.rating("strong") == 1700.0  # peeking does not mutate
    elo.advance_season(2020)
    assert elo.rating("strong") == pytest.approx(1680.0) == pytest.approx(peek_strong)
    assert elo.rating("weak") == pytest.approx(1320.0)
    elo.advance_season(2022)  # two boundaries at once
    assert elo.rating("strong") == pytest.approx(1500 + 200 * 0.9 ** 3)  # 1 boundary, then 2 more


def test_same_season_does_not_regress_and_going_back_in_time_is_an_error():
    elo = EloRatingSystem(ratings={"a": 1600.0}, season=2020)
    elo.advance_season(2020)
    assert elo.rating("a") == 1600.0
    with pytest.raises(ValueError):
        elo.advance_season(2019)


def test_pre_match_ratings_are_recorded_before_the_update():
    m = make_matches([
        {"home": "a", "away": "b", "hg": 3, "ag": 0, "date": "2020-01-01"},
        {"home": "a", "away": "c", "hg": 0, "ag": 0, "date": "2020-01-08"},
    ])
    rec, system = run_elo(m, EloConfig(home_advantage=0))
    assert rec.loc[0, "elo_home_pre"] == 1500.0 and rec.loc[0, "elo_away_pre"] == 1500.0
    assert rec.loc[1, "elo_home_pre"] > 1500.0  # 'a' carries the first win into match 2, 'c' starts fresh
    assert rec.loc[1, "elo_away_pre"] == 1500.0
    assert system.rating("a") != rec.loc[1, "elo_home_pre"]  # the second match has since changed it


def test_run_elo_ignores_matches_that_have_not_been_played():
    m = make_matches([
        {"home": "a", "away": "b", "hg": 1, "ag": 0, "date": "2020-01-01"},
        {"home": "a", "away": "b", "hg": None, "ag": None, "date": "2020-02-01", "completed": False},
    ])
    rec, system = run_elo(m)
    one_match, reference = run_elo(m.iloc[:1])
    assert len(rec) == 1
    assert system.snapshot() == reference.snapshot()  # the unplayed fixture changed nothing


def test_ratings_are_conserved_over_a_whole_synthetic_league():
    m = random_league(n_teams=8, n_rounds=30, seed=3)
    _, system = run_elo(m, EloConfig(season_regression=0.0))
    assert sum(system.ratings.values()) == pytest.approx(1500.0 * len(system.ratings))
    assert np.isfinite(list(system.ratings.values())).all()
