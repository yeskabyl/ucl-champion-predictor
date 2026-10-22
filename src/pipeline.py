"""End-to-end orchestration. Every step is a plain function so the notebook can call (and show) them one by one;
``python -m src.pipeline`` runs them all.

Forecast modes (always written into every output, never mixed up):

``LIVE_FORECAST``      official 2026/27 participants and fixtures are in the data (needs a football-data.org token).
``HISTORICAL_REPLAY``  a past season replayed as of an earlier cutoff, using only data that existed then. NOT a forecast
                       of the current season.
``SCENARIO``           user-assumed participants / schedule.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from .data_loader import (
    DataSourceError, FootballDataSource, OfficialDataUnavailable, OpenFootballSource, as_of, build_team_directory,
    get_api_token, load_matches, save_processed,
)
from .evaluation import (
    HORIZON_FROZEN, HORIZON_ONE_STEP, calibration_table, evaluate_models_on_split, make_chronological_split,
    pooled_losses, rolling_origin_evaluation, sample_size_caveats, select_model, summarise_rolling,
)
from .features import FeatureBuilder, build_features, build_frozen_features
from .models import make_model_factories
from .plots import plot_calibration, plot_champion_probabilities
from .simulator import (
    MODE_LIVE, MODE_REPLAY, MODE_SCENARIO, N_LEAGUE_MATCHES, SimConfig, SimulationResult, TournamentSetup,
    build_rate_tables, build_scenario_setup, build_setup_from_matches, simulate_tournament,
)
from .utils import ProjectPaths, Stage, get_logger, season_label, set_global_seed, sha256_file, utc_now_iso, write_json

log = get_logger("ucl.pipeline")

CI_INTERPRETATION = (
    "ci95_low/ci95_high is a 95% Wilson score interval for the share of simulated tournaments won. It measures only the "
    "random error from using a finite number of simulations (it shrinks as n_sims grows). It does NOT include "
    "uncertainty about the match model, the ratings, or the unplayed draw, so the true forecast uncertainty is larger."
)


@dataclass
class RunConfig:
    mode: str = "auto"                   # auto | live | replay | scenario
    target_season: int = 2026            # 2026 == 2026/27
    first_season: int = 2011
    replay_season: int = 2025            # 2025 == 2025/26
    replay_cutoff: str = "2025-09-01"
    n_sims: int = 10_000
    seed: int = 42
    selection_rule: str = "one_se"
    include_catboost: bool = True
    refresh: bool = False
    use_football_data: bool = True
    live_cutoff: str | None = None       # pin "now" for LIVE mode (reproducible re-runs); default: the current UTC time
    run_sensitivity: bool = True         # re-run the simulation under every candidate model (model-choice uncertainty)
    scenario_pots: list[list[str]] | None = None
    paths: ProjectPaths = field(default_factory=ProjectPaths.default)


@dataclass
class PipelineResult:
    config: RunConfig
    matches: pd.DataFrame              # normalised, as visible at the cutoff
    provenance: dict
    features: pd.DataFrame
    evaluation: dict
    model_name: str
    setup: TournamentSetup
    simulation: SimulationResult
    mode: str
    label: str
    banner: str
    cutoff: pd.Timestamp
    notes: list[str]
    replay_check: dict | None = None
    sensitivity: pd.DataFrame | None = None
    model: object | None = None
    builder: FeatureBuilder | None = None
    outputs: dict[str, Path] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# 1. data
# --------------------------------------------------------------------------------------
def load_data(cfg: RunConfig) -> tuple[pd.DataFrame, dict]:
    """football-data.org first (when a token is configured), openfootball for every season it cannot supply."""
    seasons = list(range(cfg.first_season, cfg.target_season + 1))
    sources = []
    token = get_api_token()
    if cfg.use_football_data and token:
        from .data_loader import FootballDataClient
        sources.append(FootballDataSource(FootballDataClient(token), paths=cfg.paths))
    elif cfg.use_football_data:
        log.warning("no football-data.org token configured -> using the openfootball public dataset only. "
                    "The target-season (live) data therefore cannot be loaded.")
    sources.append(OpenFootballSource(paths=cfg.paths))
    matches, prov = load_matches(sources, seasons, refresh=cfg.refresh, paths=cfg.paths)
    in_scope = matches["stage"].isin(Stage.ORDER)
    dropped = {int(k): int(v) for k, v in matches[~in_scope].groupby("season").size().items()}
    matches = matches[in_scope].reset_index(drop=True)  # drop qualifying rounds and any unmapped stage names
    matches.attrs["dropped_other_stage_rows"] = dropped  # kept so a failure to find fixtures can be explained
    prov["dropped_other_stage_rows"] = dropped
    prov["token_configured"] = bool(token)
    return matches, prov


# --------------------------------------------------------------------------------------
# 2. mode
# --------------------------------------------------------------------------------------
def _league_fixture_count(matches: pd.DataFrame, season: int) -> int:
    sm = matches[(matches["season"] == season) & (matches["stage"] == Stage.LEAGUE)]
    return int(sm.dropna(subset=["home_team_id", "away_team_id"]).shape[0])


def resolve_mode(cfg: RunConfig, matches: pd.DataFrame) -> tuple[str, pd.Timestamp, int, list[str]]:
    """Decide live / replay / scenario honestly. Returns (mode, cutoff, season, notes)."""
    notes: list[str] = []
    want_live = cfg.mode in ("auto", "live")
    if want_live:
        n = _league_fixture_count(matches, cfg.target_season)
        if n == N_LEAGUE_MATCHES:
            cutoff = pd.Timestamp(cfg.live_cutoff) if cfg.live_cutoff else pd.Timestamp.now(tz="UTC").tz_localize(None)
            return MODE_LIVE, cutoff, cfg.target_season, notes
        msg = (f"{season_label(cfg.target_season)}: only {n} of {N_LEAGUE_MATCHES} league-phase fixtures found in the data "
               f"(football-data.org token present: {bool(get_api_token())}). Official participants/fixtures unavailable.")
        unmapped = matches.attrs.get("dropped_other_stage_rows", {}).get(cfg.target_season, 0)
        if unmapped and n < N_LEAGUE_MATCHES:
            msg += (f" NOTE: {unmapped} rows of this season carry an unrecognised stage name and were excluded - if they are "
                    f"league-phase matches, extend data_loader._FD_STAGE_MAP (the notebook's schema probe lists the stage names).")
        if cfg.mode == "live":
            raise OfficialDataUnavailable(msg + " Set FOOTBALL_DATA_API_TOKEN, or use mode='replay' / 'scenario'.")
        notes.append(msg + " Falling back to HISTORICAL_REPLAY (clearly labelled; not a forecast of the current season).")
        log.warning(notes[-1])
    if cfg.mode == "scenario":
        if not cfg.scenario_pots:
            raise DataSourceError("scenario mode needs RunConfig.scenario_pots (four lists of nine club names)")
        last_done = matches.loc[matches["is_completed"], "utc_date"].max()
        return MODE_SCENARIO, last_done + pd.Timedelta(seconds=1), cfg.target_season, notes
    cutoff = pd.Timestamp(cfg.replay_cutoff)
    n = _league_fixture_count(matches, cfg.replay_season)
    if n != N_LEAGUE_MATCHES:
        raise OfficialDataUnavailable(
            f"cannot replay {season_label(cfg.replay_season)}: found {n} league-phase fixtures, need {N_LEAGUE_MATCHES}. "
            f"Historical replay only works for 36-team-format seasons (2024/25 onwards).")
    return MODE_REPLAY, cutoff, cfg.replay_season, notes


def describe_mode(mode: str, season: int, cutoff: pd.Timestamp, n_played: int | None = None) -> tuple[str, str]:
    """(label, banner)"""
    s = season_label(season)
    if mode == MODE_LIVE:
        played = f" ({n_played} of {N_LEAGUE_MATCHES} league-phase matches played)" if n_played is not None else ""
        return (f"LIVE FORECAST {s} as of {cutoff:%Y-%m-%d %H:%M} UTC",
                f"LIVE FORECAST {s} - as of {cutoff:%Y-%m-%d}{played}")
    if mode == MODE_REPLAY:
        return (f"HISTORICAL REPLAY of {s} as of {cutoff:%Y-%m-%d} (uses only data before that date)",
                f"HISTORICAL REPLAY of {s} as of {cutoff:%Y-%m-%d} - NOT a forecast of the current season")
    return (f"SCENARIO {s} (assumed participants and schedule)", f"SCENARIO {s} - assumed participants, not an official draw")


# --------------------------------------------------------------------------------------
# 3-4. features + evaluation
# --------------------------------------------------------------------------------------
def run_evaluation(features: pd.DataFrame, frozen: pd.DataFrame, cfg: RunConfig, cutoff: pd.Timestamp) -> dict:
    """Fixed chronological split + rolling-origin folds, each scored at two horizons:

    * ``one_step_ahead`` -- ordinary features, as when forecasting the next match;
    * ``frozen_state``   -- features frozen at the season start, which is how the tournament simulator predicts.

    The model for the simulation is chosen on the *frozen_state* folds that end before the test seasons.
    """
    factories = make_model_factories(cfg.seed, cfg.include_catboost)
    horizons = {HORIZON_ONE_STEP: None, HORIZON_FROZEN: frozen}
    split = make_chronological_split(features, cutoff=cutoff)
    fixed = evaluate_models_on_split(split, factories, horizons)
    rolling = rolling_origin_evaluation(features, factories, cutoff=cutoff, horizons=horizons)
    candidates = [m for m in factories if m != "climatology"]
    complexity = {name: make().complexity for name, make in factories.items()}
    last_val = max(split.val_seasons)
    if len(rolling[HORIZON_FROZEN].folds):
        losses = pooled_losses(rolling[HORIZON_FROZEN], last_val)
        basis = f"horizon-matched (frozen-state) rolling-origin folds for seasons <= {season_label(last_val)}"
    else:
        losses, basis = fixed.val_losses[HORIZON_FROZEN], "frozen-state validation window"
    selection = select_model(losses, complexity, candidates, cfg.selection_rule)
    selection.notes.insert(0, f"Selection basis: {basis}, {len(next(iter(losses.values())))} matches. The simulator freezes team "
                              f"state at the cutoff, so models are compared in exactly that setting; the test seasons were "
                              f"not used to choose.")
    return {"factories": factories, "split": split, "fixed": fixed, "rolling": rolling, "selection": selection,
            "caveats": sample_size_caveats(split), "last_val_season": last_val}


def metrics_table(ev: dict) -> pd.DataFrame:
    """One tidy table for outputs/model_metrics.csv (every row says which horizon and design it comes from)."""
    parts = [ev["fixed"].metrics.rename(columns={"split": "scope"}).assign(design="fixed_split")]
    for horizon, rr in ev["rolling"].items():
        if not len(rr.folds):
            continue
        parts.append(rr.folds.assign(design="rolling_origin_fold", horizon=horizon, scope=rr.folds["test_season"].map(season_label))
                     .drop(columns=["test_season", "n_train"]))
        for scope, upto in ((f"folds <= {season_label(ev['last_val_season'])} (selection basis)", ev["last_val_season"]),
                            ("all folds", None)):
            pooled = summarise_rolling(rr.folds, upto)
            if len(pooled):
                parts.append(pooled.assign(design="rolling_origin_pooled", horizon=horizon, scope=scope)
                             .rename(columns={"n_test_matches": "n"}).drop(columns=["n_folds"]))
    out = pd.concat(parts, ignore_index=True)
    out["selected_model"] = out["model"] == ev["selection"].chosen
    cols = ["design", "horizon", "scope", "model", "n", "start", "end", "log_loss", "brier", "accuracy", "ece", "selected_model"]
    return out[[c for c in cols if c in out.columns]]


# --------------------------------------------------------------------------------------
# 5-6. final model + simulation
# --------------------------------------------------------------------------------------
def fit_final_model(features: pd.DataFrame, factories: dict, name: str):
    done = features[features["is_completed"]]
    return factories[name]().fit(done)


def model_sensitivity(setup: TournamentSetup, builder, features: pd.DataFrame, factories: dict, candidates: list[str],
                      chosen: str, cfg: RunConfig, chosen_sim: SimulationResult) -> pd.DataFrame:
    """Championship probability under every candidate model: how much does the model choice move the answer?"""
    done = features[features["is_completed"]]
    table = chosen_sim.champion_table()[["team_id", "team_name", "champion_probability"]].rename(
        columns={"champion_probability": f"p_{chosen}"})
    for name in candidates:
        if name == chosen:
            continue
        model = factories[name]().fit(done)
        alt = replace(setup, rates=build_rate_tables(model, builder, setup.team_ids, setup.season))
        sim = simulate_tournament(alt, SimConfig(n_sims=cfg.n_sims, seed=cfg.seed))
        col = sim.champion_table()[["team_id", "champion_probability"]].rename(columns={"champion_probability": f"p_{name}"})
        table = table.merge(col, on="team_id", how="left")
    cols = [c for c in table.columns if c.startswith("p_")]
    table["range_across_models"] = table[cols].max(axis=1) - table[cols].min(axis=1)
    table.insert(0, "selected_model", chosen)
    return table.sort_values(f"p_{chosen}", ascending=False).reset_index(drop=True)


def replay_check(full_matches: pd.DataFrame, season: int, sim: SimulationResult) -> dict | None:
    """After the fact only: how much probability had the replay given to the team that actually won?"""
    final = full_matches[(full_matches["season"] == season) & (full_matches["stage"] == Stage.FINAL) & full_matches["is_completed"]]
    if final.empty:
        return None
    r = final.iloc[0]
    winner_side = r["winner"]
    team_id = r["home_team_id"] if winner_side == "HOME_TEAM" else r["away_team_id"] if winner_side == "AWAY_TEAM" else None
    if team_id is None:
        return None
    tab = sim.champion_table()
    row = tab[tab["team_id"] == team_id].iloc[0]
    return {"actual_champion": row["team_name"], "assigned_probability": float(row["champion_probability"]),
            "rank_in_forecast": int(row["rank"]),
            "caveat": "A single season cannot validate calibration; this is an anecdote, not evidence of accuracy."}


# --------------------------------------------------------------------------------------
# full run
# --------------------------------------------------------------------------------------
def run_pipeline(cfg: RunConfig, *, save: bool = True) -> PipelineResult:
    started = utc_now_iso()
    set_global_seed(cfg.seed)
    cfg.paths.ensure()

    full, provenance = load_data(cfg)
    mode, cutoff, season, notes = resolve_mode(cfg, full)
    visible = as_of(full, cutoff, mask_undetermined_knockout=(mode != MODE_LIVE))
    features, builder = build_features(visible)
    frozen = build_frozen_features(visible)
    log.info("mode=%s season=%s cutoff=%s | %d matches visible, %d completed", mode, season, cutoff,
             len(visible), int(visible["is_completed"].sum()))

    ev = run_evaluation(features, frozen, cfg, cutoff)
    chosen = ev["selection"].chosen
    model = fit_final_model(features, ev["factories"], chosen)

    n_played = int(visible[(visible["season"] == season) & (visible["stage"] == Stage.LEAGUE) & visible["is_completed"]].shape[0])
    label, banner = describe_mode(mode, season, cutoff, n_played if mode != MODE_SCENARIO else None)
    if mode == MODE_SCENARIO:
        setup = build_scenario_setup(cfg.scenario_pots, model, builder, season=season, seed=cfg.seed, label=label)
    else:
        setup = build_setup_from_matches(visible, season, model, builder, mode=mode, label=label, cutoff=cutoff)
    sim = simulate_tournament(setup, SimConfig(n_sims=cfg.n_sims, seed=cfg.seed))

    check = replay_check(full, season, sim) if mode == MODE_REPLAY else None
    sens = None
    if cfg.run_sensitivity:
        candidates = [m for m in ev["factories"] if m != "climatology"]
        sens = model_sensitivity(setup, builder, features, ev["factories"], candidates, chosen, cfg, sim)
    result = PipelineResult(cfg, visible, provenance, features, ev, chosen, setup, sim, mode, label, banner, cutoff,
                            notes, check, sens, model, builder)
    if save:
        save_outputs(result, started)
    return result


# --------------------------------------------------------------------------------------
# 7. outputs
# --------------------------------------------------------------------------------------
def _versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "pandas", "sklearn", "scipy", "catboost", "matplotlib", "requests"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = None
    return out


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
                              timeout=5, check=True).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def save_model_artifacts(res: PipelineResult) -> dict[str, Path]:
    """The fitted model (joblib) and the team ratings it uses, both as of the cutoff."""
    import joblib

    paths = res.config.paths.ensure()
    directory = build_team_directory(res.matches).set_index("team_id")
    ratings = pd.DataFrame({"team_id": list(res.builder.elo.ratings), "elo": list(res.builder.elo.ratings.values())})
    ratings["team_name"] = ratings["team_id"].map(directory["team_name"])
    ratings = ratings.sort_values("elo", ascending=False).reset_index(drop=True)
    ratings.insert(0, "as_of", str(res.cutoff))
    out = {"elo_ratings": paths.outputs / "elo_ratings_at_cutoff.csv", "model_artifact": paths.outputs / "model.joblib"}
    ratings.to_csv(out["elo_ratings"], index=False)
    joblib.dump({"model": res.model, "model_name": res.model_name, "elo_ratings": res.builder.elo.snapshot(),
                 "elo_config": res.builder.elo.config, "cutoff": str(res.cutoff), "forecast_mode": res.mode,
                 "created_at": utc_now_iso(), "note": "load with joblib; unpickling needs the `src` package on sys.path"},
                out["model_artifact"])
    return out


def save_outputs(res: PipelineResult, started_at: str | None = None) -> dict[str, Path]:
    paths, sim, ev = res.config.paths.ensure(), res.simulation, res.evaluation
    out: dict[str, Path] = {}
    out["matches"] = save_processed(res.matches, paths.data_processed / "matches.csv")

    mt = metrics_table(ev)
    out["model_metrics"] = paths.outputs / "model_metrics.csv"
    mt.to_csv(out["model_metrics"], index=False)

    champ = sim.champion_table()
    total = float(champ["champion_probability"].sum())
    if abs(total - 1.0) > 1e-9:
        raise AssertionError(f"champion probabilities sum to {total}, expected 1")
    out["champion_probabilities"] = paths.outputs / "champion_probabilities.csv"
    champ.to_csv(out["champion_probabilities"], index=False)
    out["qualification_probabilities"] = paths.outputs / "qualification_probabilities.csv"
    sim.qualification_table().to_csv(out["qualification_probabilities"], index=False)
    out["final_standings_probabilities"] = paths.outputs / "final_standings_probabilities.csv"
    sim.final_standings_table().to_csv(out["final_standings_probabilities"], index=False)

    out["champion_chart"] = plot_champion_probabilities(
        champ, paths.outputs / "champion_probabilities.png", title="Champions League winner probabilities",
        banner=res.banner, n_sims=sim.n_sims)

    fixed = ev["fixed"]
    test = ev["split"].test
    cal_frames = []
    for horizon in (HORIZON_ONE_STEP, HORIZON_FROZEN):
        proba = fixed.predictions[horizon][res.model_name]["test"]
        cal_frames.append(calibration_table(test["result"].tolist(), proba).assign(
            model=res.model_name, horizon=horizon, scope="test seasons"))
        rr = ev["rolling"][horizon]
        if len(rr.folds):
            y_pool, p_pool = rr.pooled_predictions(res.model_name)
            cal_frames.append(calibration_table(y_pool, p_pool, n_bins=6).assign(
                model=res.model_name, horizon=horizon, scope="all rolling-origin folds (out-of-sample)"))
    cal_all = pd.concat(cal_frames, ignore_index=True)
    out["calibration_table"] = paths.outputs / "calibration_table.csv"
    cal_all.to_csv(out["calibration_table"], index=False)
    plot_df = cal_all[(cal_all["horizon"] == HORIZON_FROZEN) & cal_all["scope"].str.startswith("all rolling")]
    plot_scope = "pooled rolling-origin folds"
    if plot_df.empty:
        plot_df, plot_scope = cal_all[(cal_all["horizon"] == HORIZON_FROZEN) & (cal_all["scope"] == "test seasons")], "test seasons"
    n_plot = int(plot_df[plot_df["class"] == "H"]["n"].sum())
    out["calibration_chart"] = plot_calibration(
        plot_df, paths.outputs / "calibration.png", title=f"Calibration of the '{res.model_name}' model",
        subtitle=f"Simulator horizon, out-of-sample ({plot_scope}, {n_plot} matches); equal-count bins")
    if res.model is not None and res.builder is not None:
        out.update(save_model_artifacts(res))
    if res.sensitivity is not None:
        out["champion_probabilities_by_model"] = paths.outputs / "champion_probabilities_by_model.csv"
        res.sensitivity.to_csv(out["champion_probabilities_by_model"], index=False)

    directory = build_team_directory(res.matches)
    meta = {
        "run": {"started_at": started_at, "finished_at": utc_now_iso(), "git_commit": _git_commit(paths.root),
                "seed": res.config.seed, "n_sims": sim.n_sims},
        "forecast": {"mode": res.mode, "label": res.label, "banner": res.banner, "cutoff": str(res.cutoff),
                     "target_season": res.setup.season, "is_live_forecast": res.mode == MODE_LIVE,
                     "league_matches_known": sim.notes["league_matches_known"],
                     "league_matches_simulated": sim.notes["league_matches_simulated"],
                     "known_knockout_legs": sim.notes["known_knockout_legs"],
                     "knockout_draw_outcomes_consistent_with_data": sim.n_draw_outcomes,
                     "team_notes": sim.notes["team_notes"], "mode_notes": res.notes,
                     "replay_check": res.replay_check, "ci_interpretation": CI_INTERPRETATION,
                     "assumptions": sim.assumptions},
        "data": {"provenance": res.provenance, "n_matches_visible": int(len(res.matches)),
                 "n_completed": int(res.matches["is_completed"].sum()), "n_teams": int(len(directory)),
                 "date_range": [str(res.matches["utc_date"].min()), str(res.matches["utc_date"].max())]},
        "evaluation": {
            "split": ev["split"].describe().assign(start=lambda d: d["start"].astype(str), end=lambda d: d["end"].astype(str)).to_dict("records"),
            "selected_model": res.model_name, "selection_rule": ev["selection"].rule,
            "selection_table": ev["selection"].table.to_dict("records"), "selection_notes": ev["selection"].notes,
            "caveats": ev["caveats"],
            "reference_model_is_climatology": "'climatology' (class frequencies only) is the no-skill baseline in model_metrics.csv",
            "horizons": {HORIZON_ONE_STEP: "features computed just before each match (normal match forecasting)",
                         HORIZON_FROZEN: "features frozen at the start of the season = how the tournament simulator predicts; "
                                         "the model is selected on this horizon"},
            "calibration_note": "Calibration was measured on the test seasons only (outputs/calibration_table.csv); "
                                "see ece in model_metrics.csv. With a few hundred matches per split these estimates are noisy.",
        },
        "versions": _versions(),
    }
    out["run_metadata"] = write_json(paths.outputs / "run_metadata.json", meta)
    meta["outputs_sha256"] = {k: sha256_file(v) for k, v in out.items() if k != "run_metadata"}
    write_json(out["run_metadata"], meta)
    res.outputs, res.metadata = out, meta
    return out


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="UEFA Champions League champion probability pipeline")
    ap.add_argument("--mode", choices=["auto", "live", "replay", "scenario"], default="auto")
    ap.add_argument("--target-season", type=int, default=2026, help="start year (2026 = 2026/27)")
    ap.add_argument("--replay-season", type=int, default=2025)
    ap.add_argument("--replay-cutoff", default="2025-09-01")
    ap.add_argument("--n-sims", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selection-rule", choices=["one_se", "best"], default="one_se")
    ap.add_argument("--no-catboost", action="store_true")
    ap.add_argument("--no-football-data", action="store_true", help="skip the API even if a token is configured")
    ap.add_argument("--refresh", action="store_true", help="ignore cached raw snapshots")
    ap.add_argument("--scenario-json", help="JSON file: list of four lists of nine club names")
    args = ap.parse_args(argv)
    pots = json.loads(Path(args.scenario_json).read_text(encoding="utf-8")) if args.scenario_json else None
    cfg = RunConfig(mode=args.mode, target_season=args.target_season, replay_season=args.replay_season,
                    replay_cutoff=args.replay_cutoff, n_sims=args.n_sims, seed=args.seed,
                    selection_rule=args.selection_rule, include_catboost=not args.no_catboost, refresh=args.refresh,
                    use_football_data=not args.no_football_data, scenario_pots=pots)
    res = run_pipeline(cfg)
    print(f"\n{res.banner}\nselected model: {res.model_name}\n")
    print(res.simulation.champion_table().head(12)[["rank", "team_name", "simulated_championships", "champion_probability",
                                                    "ci95_low", "ci95_high"]].to_string(index=False))
    print("\nwritten:", ", ".join(p.name for p in res.outputs.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
