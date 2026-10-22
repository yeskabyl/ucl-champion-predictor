"""End-to-end runs of the pipeline, fully offline.

The snapshot cache is pre-seeded with the real openfootball season files from ``tests/fixtures`` so no network is touched.
The *live-mode* test needs a season that looks like a season in progress; provider state for it is MOCKED by truncating the
real 2025/26 file (results removed after matchday 4, knockout rounds removed -- which is what a provider returns as TBD) and
filing it under season 2026. That mock exists only inside this test; nothing like it is ever produced by the pipeline.
"""
import json
import re

import pandas as pd
import pytest
import requests

from src import pipeline as pl
from src.data_loader import DataSourceError, OfficialDataUnavailable, RawSnapshotStore
from src.evaluation import InsufficientDataError
from src.simulator import MODE_LIVE, MODE_REPLAY, MODE_SCENARIO
from src.utils import ProjectPaths, Stage

from .helpers import FIXTURES
from .test_data_loader import FakeResponse


def block_network(mp):
    """Any HTTP request made while this is active gets a 404 -- and is recorded, so tests can prove nothing was fetched."""
    calls = []

    def fake_get(self, url, **kwargs):
        calls.append(url)
        return FakeResponse(404, {"message": "offline test"})

    mp.setattr(requests.Session, "get", fake_get)
    return calls


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    block_network(monkeypatch)


def seed_cache(root, extra_season_text=None):
    paths = ProjectPaths(root).ensure()
    store = RawSnapshotStore(paths.data_raw)
    for season in range(2011, 2026):
        text = (FIXTURES / f"cl_{season}-{str(season + 1)[-2:]}.txt").read_text(encoding="utf-8")
        store.save("openfootball", f"cl_{season}-{str(season + 1)[-2:]}", text, f"fixture://{season}", ext="txt",
                   provider="openfootball (CC0)")
    if extra_season_text is not None:
        store.save("openfootball", "cl_2026-27", extra_season_text, "fixture://mock-2026", ext="txt", provider="MOCK (test)")
    return paths


def mock_season_in_progress() -> str:
    """Real 2025/26 league phase, dates moved one year on (so it follows the real 2025/26 season), results removed after
    matchday 4 and no knockout rounds (a provider lists those as TBD)."""
    text = (FIXTURES / "cl_2025-26.txt").read_text(encoding="utf-8")
    keep = text.split("▪ Playoffs, Matchday 1")[0]
    keep = re.sub(r"\b(2025|2026)\b", lambda m: str(int(m.group(1)) + 1), keep)
    head, _, rest = keep.partition("▪ League, Matchday 5")
    score = re.compile(r"(\)\s+)\d+-\d+(?:\s+\([^)]*\))?\s*$", re.M)
    return head + "▪ League, Matchday 5" + score.sub(r"\1", rest)


def cfg(paths, **kw):
    base = dict(paths=paths, n_sims=400, seed=3, include_catboost=False, run_sensitivity=False, use_football_data=False,
                target_season=2025)
    base.update(kw)
    return pl.RunConfig(**base)


@pytest.fixture(scope="module")
def replay_run(tmp_path_factory):
    paths = seed_cache(tmp_path_factory.mktemp("replay"))
    with pytest.MonkeyPatch.context() as mp:
        calls = block_network(mp)
        result = pl.run_pipeline(cfg(paths, mode="replay", n_sims=1000))
    assert calls == []  # everything came from the pre-seeded snapshot cache: no HTTP request was even attempted
    return result


# ------------------------------------------------------------------ replay mode, full run
def test_replay_run_produces_every_required_output(replay_run):
    r, paths = replay_run, replay_run.config.paths
    expected = ["data/processed/matches.csv", "outputs/model_metrics.csv", "outputs/champion_probabilities.csv",
                "outputs/qualification_probabilities.csv", "outputs/final_standings_probabilities.csv",
                "outputs/champion_probabilities.png", "outputs/run_metadata.json"]
    for rel in expected:
        f = paths.root / rel
        assert f.is_file() and f.stat().st_size > 0, rel
    assert r.mode == MODE_REPLAY and "NOT a forecast" in r.banner


def test_champion_table_has_the_required_columns_and_sums_to_one(replay_run):
    df = pd.read_csv(replay_run.config.paths.outputs / "champion_probabilities.csv")
    for col in ("team_name", "simulated_championships", "champion_probability", "ci95_low", "ci95_high", "mc_std_error",
                "forecast_mode"):
        assert col in df.columns
    assert len(df) == 36 and df["simulated_championships"].sum() == 1000
    assert df["champion_probability"].sum() == pytest.approx(1.0, abs=1e-9)
    assert df["forecast_mode"].eq(MODE_REPLAY).all()
    assert (df["ci95_low"] <= df["champion_probability"]).all() and (df["champion_probability"] <= df["ci95_high"]).all()


def test_qualification_and_standing_tables_are_consistent(replay_run):
    q = pd.read_csv(replay_run.config.paths.outputs / "qualification_probabilities.csv")
    assert q["p_top8_direct_r16"].sum() == pytest.approx(8) and q["p_playoff"].sum() == pytest.approx(16)
    assert q["p_eliminated"].sum() == pytest.approx(12) and q["p_champion"].sum() == pytest.approx(1.0)
    fs = pd.read_csv(replay_run.config.paths.outputs / "final_standings_probabilities.csv")
    assert fs[[f"p_pos_{i}" for i in range(1, 37)]].sum(axis=1).tolist() == pytest.approx([1.0] * 36)


def test_replay_uses_only_information_from_before_its_cutoff(replay_run):
    cutoff = pd.Timestamp("2025-09-01")
    visible = replay_run.matches
    assert (visible.loc[visible["is_completed"], "utc_date"] < cutoff).all()
    assert replay_run.features.loc[replay_run.features["is_completed"], "utc_date"].max() < cutoff
    assert not visible[(visible["season"] == 2025)]["is_completed"].any()
    # no knockout pairing of the replayed season is visible yet
    ko = visible[(visible["season"] == 2025) & visible["stage"].isin(Stage.KNOCKOUT)]
    assert ko["home_team_id"].isna().all()
    # every evaluation window ends before the cutoff as well
    split = replay_run.evaluation["split"]
    assert split.test["utc_date"].max() < cutoff
    assert split.train["utc_date"].max() < split.val["utc_date"].min() < split.test["utc_date"].min()


def test_metadata_is_honest_about_what_this_is(replay_run):
    meta = json.loads((replay_run.config.paths.outputs / "run_metadata.json").read_text(encoding="utf-8"))
    f = meta["forecast"]
    assert f["mode"] == MODE_REPLAY and f["is_live_forecast"] is False and f["target_season"] == 2025
    assert f["league_matches_known"] == 0 and f["league_matches_simulated"] == 144
    assert "Monte Carlo" in f["ci_interpretation"] or "simulations" in f["ci_interpretation"]
    assert any("frozen" in a.lower() for a in f["assumptions"])
    assert f["replay_check"]["caveat"] and "anecdote" in f["replay_check"]["caveat"]
    ev = meta["evaluation"]
    assert ev["selected_model"] == replay_run.model_name and ev["selection_notes"] and ev["caveats"]
    assert {s["split"] for s in ev["split"]} == {"train", "validation", "test"}
    assert all(s["n_matches"] > 0 and s["start"] < s["end"] for s in ev["split"])
    assert meta["data"]["provenance"]["season_source"]["2024"] == "openfootball"
    assert meta["data"]["provenance"]["token_configured"] in (True, False)
    assert set(meta["outputs_sha256"]) >= {"champion_probabilities", "model_metrics", "matches"}
    assert "token" not in json.dumps(meta).lower().replace("token_configured", "")  # no credential material


def test_metrics_table_reports_every_split_with_its_size_and_dates(replay_run):
    m = pd.read_csv(replay_run.config.paths.outputs / "model_metrics.csv")
    fixed = m[(m["design"] == "fixed_split") & (m["horizon"] == "frozen_state")]
    assert set(fixed["scope"]) == {"validation", "test"} and (fixed["n"] > 0).all()
    assert fixed["start"].notna().all() and fixed["end"].notna().all()
    assert {"climatology", "elo", "logistic", "poisson"} <= set(m["model"])
    assert m[m["selected_model"]]["model"].nunique() == 1
    assert {"log_loss", "brier", "accuracy", "ece"} <= set(m.columns)
    clim = m[(m["design"] == "rolling_origin_pooled") & (m["model"] == "climatology")]["log_loss"].iloc[0]
    best = m[(m["design"] == "rolling_origin_pooled") & (m["model"] != "climatology")]["log_loss"].min()
    assert best < clim  # the models beat the no-skill baseline out of sample


def test_the_run_is_reproducible(replay_run, tmp_path):
    again = pl.run_pipeline(cfg(seed_cache(tmp_path), mode="replay", n_sims=1000))
    a = replay_run.simulation.champion_table()
    b = again.simulation.champion_table()
    pd.testing.assert_frame_equal(a, b)
    assert again.model_name == replay_run.model_name


def test_replay_check_is_reported_after_the_fact_only(replay_run):
    rc = replay_run.replay_check
    assert rc["actual_champion"] == "Paris Saint-Germain FC"
    assert 0 < rc["assigned_probability"] < 1 and rc["rank_in_forecast"] >= 1


# ------------------------------------------------------------------ mode resolution
def test_auto_mode_falls_back_to_a_labelled_replay_when_the_target_season_is_unavailable(tmp_path):
    paths = seed_cache(tmp_path)
    r = pl.run_pipeline(cfg(paths, mode="auto", n_sims=200, target_season=2026), save=False)
    assert r.mode == MODE_REPLAY and any("Falling back" in n for n in r.notes)
    assert "2026/27" not in r.label  # it must never be presented as a 2026/27 forecast


def test_live_mode_refuses_to_run_without_official_data(tmp_path):
    paths = seed_cache(tmp_path)
    with pytest.raises(OfficialDataUnavailable, match="league-phase fixtures"):
        pl.run_pipeline(cfg(paths, mode="live", target_season=2026), save=False)


def test_unrecognised_stage_names_are_explained_when_fixtures_seem_missing():
    from .helpers import random_league

    df = random_league(seed=1)
    df.attrs["dropped_other_stage_rows"] = {2026: 189}  # what load_data records when a provider uses an unmapped stage name
    with pytest.raises(OfficialDataUnavailable, match="unrecognised stage name"):
        pl.resolve_mode(pl.RunConfig(mode="live", target_season=2026), df)
    df.attrs["dropped_other_stage_rows"] = {}
    with pytest.raises(OfficialDataUnavailable) as exc:
        pl.resolve_mode(pl.RunConfig(mode="live", target_season=2026), df)
    assert "unrecognised stage name" not in str(exc.value)


def test_replay_of_a_season_without_the_36_team_format_is_refused(tmp_path):
    paths = seed_cache(tmp_path)
    with pytest.raises(OfficialDataUnavailable, match="36-team"):
        pl.run_pipeline(cfg(paths, mode="replay", replay_season=2022, replay_cutoff="2022-09-01"), save=False)


def test_scenario_mode_needs_participants(tmp_path):
    with pytest.raises(DataSourceError, match="scenario_pots"):
        pl.run_pipeline(cfg(seed_cache(tmp_path), mode="scenario"), save=False)


def test_scenario_mode_end_to_end_is_labelled_as_an_assumption(tmp_path):
    paths = seed_cache(tmp_path)
    full, _ = pl.load_data(cfg(paths, target_season=2025))
    teams = (full[full["season"] == 2025].drop_duplicates("home_team_id").sort_values("home_team_id")["home_team"].tolist())
    assert len(teams) >= 36
    pots = [teams[i * 9:(i + 1) * 9] for i in range(4)]
    r = pl.run_pipeline(cfg(paths, mode="scenario", scenario_pots=pots, n_sims=300, target_season=2026))
    assert r.mode == MODE_SCENARIO and "SCENARIO" in r.banner and "not an official draw" in r.banner
    meta = json.loads((paths.outputs / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["forecast"]["is_live_forecast"] is False
    assert any("not official" in a for a in meta["forecast"]["assumptions"])
    assert pd.read_csv(paths.outputs / "champion_probabilities.csv")["champion_probability"].sum() == pytest.approx(1.0)


def test_too_little_history_stops_the_run_with_a_clear_message(tmp_path):
    paths = ProjectPaths(tmp_path).ensure()
    store = RawSnapshotStore(paths.data_raw)
    for season in (2024, 2025):
        text = (FIXTURES / f"cl_{season}-{str(season + 1)[-2:]}.txt").read_text(encoding="utf-8")
        store.save("openfootball", f"cl_{season}-{str(season + 1)[-2:]}", text, "fixture://", ext="txt")
    with pytest.raises(InsufficientDataError, match="complete seasons"):
        pl.run_pipeline(cfg(paths, mode="replay", first_season=2024), save=False)


# ------------------------------------------------------------------ live mode mechanics (provider state is MOCKED)
def test_live_mode_with_a_mocked_season_in_progress(tmp_path):
    paths = seed_cache(tmp_path, extra_season_text=mock_season_in_progress())
    r = pl.run_pipeline(cfg(paths, mode="live", n_sims=500, target_season=2026, live_cutoff="2026-11-15"))
    assert r.mode == MODE_LIVE and r.banner.startswith("LIVE FORECAST 2026/27") and "2026-11-15" in r.banner
    sim = r.simulation
    assert 0 < sim.notes["league_matches_known"] < 144
    assert sim.notes["league_matches_known"] + sim.notes["league_matches_simulated"] == 144
    assert sim.notes["known_knockout_legs"] == 0 and sim.n_draw_outcomes == 4096
    played = r.matches[(r.matches["season"] == 2026) & r.matches["is_completed"]]
    assert len(played) == sim.notes["league_matches_known"]
    # the in-progress season is not used as an evaluation season but IS used to fit the final model
    split = r.evaluation["split"]
    assert 2026 not in set(split.train["season"]) | set(split.val["season"]) | set(split.test["season"])
    assert r.features[r.features["season"] == 2026]["is_completed"].sum() == len(played)
    meta = json.loads((paths.outputs / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["forecast"]["is_live_forecast"] is True and meta["forecast"]["replay_check"] is None
    assert meta["data"]["provenance"]["season_source"]["2026"] == "openfootball"  # (mock) source recorded per season
    tab = pd.read_csv(paths.outputs / "champion_probabilities.csv")
    assert tab["forecast_mode"].eq(MODE_LIVE).all() and tab["champion_probability"].sum() == pytest.approx(1.0)


def test_sensitivity_table_shows_how_much_the_model_choice_matters(tmp_path):
    paths = seed_cache(tmp_path)
    r = pl.run_pipeline(cfg(paths, mode="replay", n_sims=300, run_sensitivity=True))
    s = pd.read_csv(paths.outputs / "champion_probabilities_by_model.csv")
    assert {"p_elo", "p_logistic", "p_poisson", "range_across_models"} <= set(s.columns) and len(s) == 36
    for col in [c for c in s.columns if c.startswith("p_")]:
        assert s[col].sum() == pytest.approx(1.0)
    assert r.sensitivity is not None
