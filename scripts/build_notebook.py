"""Generate notebooks/UCL_Champion_Predictor.ipynb (run: ``python scripts/build_notebook.py [--execute]``).

The notebook is generated rather than hand-edited so it stays in sync with ``src/`` and is easy to review as code.
With ``--execute`` it is also run top to bottom (needs jupyter's nbclient + an ipykernel) and saved with its outputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "UCL_Champion_Predictor.ipynb"

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# ----------------------------------------------------------------------------------------------------------------
md("""
# UEFA Champions League 2026/27 - champion probability model

A reproducible pipeline that (1) collects historical Champions League results, (2) estimates team strength and match
outcome probabilities with leakage-safe features, (3) evaluates models **chronologically**, and (4) runs a Monte Carlo
simulation of the 36-team league phase and knockout bracket to estimate each club's chance of winning the trophy.

> **Read this first.** The output is a *probability table*, not a claim to know the winner. Every table and chart is
> stamped with the **forecast mode** that produced it:
> `LIVE_FORECAST` (official 2026/27 participants and fixtures were available), `HISTORICAL_REPLAY` (a past season replayed
> using only data from before its cutoff - **not** a forecast of the current season) or `SCENARIO` (assumed participants).
> If the official 2026/27 fixtures cannot be loaded, the notebook says so and falls back to a clearly labelled replay -
> it never invents data.

**Run it in Google Colab:** upload the project folder as a zip (or set `REPO_URL` below), optionally add a Colab secret named
`FOOTBALL_DATA_API_TOKEN` (free token: <https://www.football-data.org/client/register>), then *Runtime -> Run all*.
No GPU is needed.
""")

md("## 1. Install dependencies and locate the project")
code('''
import os, sys, glob, pathlib, subprocess

IN_COLAB = "google.colab" in sys.modules
REPO_URL = ""            # optional: "https://github.com/<you>/ucl-champion-predictor.git" to clone instead of uploading a zip

def find_root():
    here = pathlib.Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "src" / "pipeline.py").exists():
            return p
    return None

ROOT = find_root()
if ROOT is None and IN_COLAB:
    if REPO_URL:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "/content/ucl-champion-predictor"], check=True)
    else:
        from google.colab import files
        import zipfile
        print("Upload the zipped project folder (ucl-champion-predictor.zip):")
        for name in files.upload():
            zipfile.ZipFile(name).extractall("/content")
    hits = glob.glob("/content/**/src/pipeline.py", recursive=True)
    ROOT = pathlib.Path(hits[0]).parents[1] if hits else None
if ROOT is None:
    raise RuntimeError("Could not find the project (src/pipeline.py). Open this notebook from inside the project folder, "
                       "or upload the project zip in Colab.")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

if IN_COLAB:   # Colab already ships pandas / numpy / scikit-learn / matplotlib; this adds what is missing (e.g. catboost)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
print("project root:", ROOT, "| Colab:", IN_COLAB, "| python", sys.version.split()[0])
''')

md("""
## 2. Configure API credentials securely

The football-data.org token is read from a **Colab secret**, an **environment variable**, or a git-ignored `.env` file -
never from the notebook source, and it is never printed. Without a token the project still runs on the public-domain
openfootball dataset (historical results only); only the *live* 2026/27 data needs the API.
""")
code('''
from src.data_loader import TOKEN_ENV_VAR, get_api_token

ASK_IF_MISSING = False   # set True in Colab to be prompted for the token with a hidden input box

try:                                                   # 1) Colab secret (key icon in the left sidebar)
    from google.colab import userdata
    secret = userdata.get(TOKEN_ENV_VAR)
    if secret:
        os.environ[TOKEN_ENV_VAR] = secret
except Exception:
    pass

if not get_api_token() and ASK_IF_MISSING:             # 2) hidden prompt
    import getpass
    typed = getpass.getpass("football-data.org API token (input hidden, Enter to skip): ").strip()
    if typed:
        os.environ[TOKEN_ENV_VAR] = typed

TOKEN_OK = bool(get_api_token())                       # 3) environment variable / .env were checked by get_api_token()
print("football-data.org token configured:", TOKEN_OK)
if not TOKEN_OK:
    print("-> no token: historical data comes from the openfootball dataset and the live 2026/27 season cannot be loaded.")
''')

md("""
### Run configuration
Everything that changes a result lives in this one cell. `mode="auto"` tries the live 2026/27 forecast and otherwise falls
back to a labelled historical replay.
""")
code('''
import warnings
import numpy as np
import pandas as pd
from IPython.display import display, Image

from src import pipeline as pl
from src.data_loader import as_of, describe_football_data_payload, clean_matches
from src.features import build_features, build_frozen_features
from src.simulator import (MODE_LIVE, MODE_REPLAY, MODE_SCENARIO, SimConfig, build_setup_from_matches,
                           build_scenario_setup, simulate_tournament)
from src.utils import season_label

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 160, "display.max_columns", 30)

cfg = pl.RunConfig(
    mode="auto",                 # "auto" | "live" | "replay" | "scenario"
    target_season=2026,          # 2026 == 2026/27
    replay_season=2025,          # used for the historical replay (2024 or 2025: the 36-team format)
    replay_cutoff="2025-09-01",  # the replay may only use information from before this date
    n_sims=10_000, seed=42,
    include_catboost=True,       # optional gradient-boosting model; skipped automatically if not installed
    run_sensitivity=True,        # re-simulate under every candidate model to show model-choice uncertainty
    refresh=False,               # True = ignore cached raw snapshots and call the data sources again
)
cfg.paths.ensure()
print(cfg.mode, "| simulations:", cfg.n_sims, "| seed:", cfg.seed)
''')

md("""
## 3. Download and validate the data

Sources are tried in priority order **per season**: football-data.org (REST API, needs a token) and, for every season it cannot
supply, openfootball (CC0 text files). Raw responses are snapshotted under `data/raw/` with the retrieval time, so re-running
does not hit the network again. Duplicates, impossible rows and missing scores are removed *and reported*; unavailable seasons
are listed with the reason - nothing is substituted silently.
""")
code('''
full, provenance = pl.load_data(cfg)

print("seasons loaded per source:", pd.Series(provenance["season_source"]).value_counts().to_dict())
print("seasons NOT loaded:", provenance["seasons_not_loaded"] or "none")
for key, reason in provenance["unavailable"].items():
    print(f"  unavailable  {key}: {reason[:140]}")
print("\\nvalidation report:", {k: v for k, v in provenance["validation"].items() if k != "warnings"})
for w in provenance["validation"]["warnings"]:
    print("  warning:", w)
print("retrieved at:", provenance["retrieved_at"])
print(f"\\n{len(full):,} matches, {full['season'].nunique()} seasons "
      f"({season_label(int(full['season'].min()))} - {season_label(int(full['season'].max()))}), "
      f"{int(full['is_completed'].sum()):,} completed, {int((~full['is_completed']).sum())} not yet played")
''')
code('''
# Verify the real response schema of football-data.org instead of assuming it (only possible when that source was used).
import json
snap_dir = cfg.paths.data_raw / "football_data"
snaps = sorted(snap_dir.glob("CL_*.json")) if snap_dir.exists() else []
if snaps:
    probe = describe_football_data_payload(json.loads(snaps[-1].read_text(encoding="utf-8")))
    print("latest football-data.org snapshot:", snaps[-1].name)
    for k in ("n_matches", "stages", "statuses", "durations", "score_nodes", "unmapped_stages", "missing_required"):
        print(f"  {k}: {probe[k]}")
else:
    print("football-data.org was not used in this run (no token or no cached snapshot), so its response schema could not be\\n"
          "verified here. The parser accepts both 'home/away' and 'homeTeam/awayTeam' score keys, derives the 90-minute score\\n"
          "from regularTime (or fullTime - extraTime - penalties) and reports unmapped stage names; with a token this cell\\n"
          "prints what the API actually returns.")
''')

md("## 4. Explore the dataset")
code('''
by_season = full.groupby("season").agg(matches=("match_id", "size"), completed=("is_completed", "sum"),
                                       teams=("home_team_id", "nunique"))
done = full[full["is_completed"]]
by_season["goals_per_match"] = (done["home_goals_90"] + done["away_goals_90"]).groupby(done["season"]).mean()
by_season["home_win_%"] = 100 * (done["home_goals_90"] > done["away_goals_90"]).groupby(done["season"]).mean()
by_season["draw_%"] = 100 * (done["home_goals_90"] == done["away_goals_90"]).groupby(done["season"]).mean()
by_season.index = [season_label(int(s)) for s in by_season.index]
display(by_season.round(2))
print("matches by stage:", full["stage"].value_counts().to_dict())
print("distinct clubs:", pd.concat([full["home_team_id"], full["away_team_id"]]).nunique())
''')
code('''
missing = full.isna().sum()
print("missing values (columns with any):"); display(missing[missing > 0].to_frame("n_missing"))
print("duplicate match_id:", int(full["match_id"].duplicated().sum()),
      "| completed rows without a score:", int((full["is_completed"] & full["home_goals_90"].isna()).sum()),
      "| unplayed fixtures:", int((~full["is_completed"]).sum()))
''')
code('''
from src.plots import plot_outcome_shares, plot_goal_histogram
import matplotlib.pyplot as plt
fig1 = plot_outcome_shares(full, cfg.paths.outputs / "eda_outcome_shares.png"); plt.show()
fig2 = plot_goal_histogram(full, cfg.paths.outputs / "eda_goals.png"); plt.show()
''')

md("""
## 5. Build leakage-safe features

Decide the forecast mode and the **prediction cutoff** first, then restrict the table to what was known at that moment
(`as_of`): later results are hidden, and in a replay the participants of knockout ties not yet determined are hidden too.

Features are computed in one chronological pass: a match's features come from team state frozen **before** the match
(Elo, rolling goals scored/conceded over the previous 5 and 10 matches, recent points, opponent strength, rest days, venue),
and the state is updated **after** the features are recorded.
""")
code('''
mode, cutoff, season, mode_notes = pl.resolve_mode(cfg, full)
label, banner = pl.describe_mode(mode, season, cutoff)
for n in mode_notes:
    print("NOTE:", n)
print("=" * len(banner)); print(banner); print("=" * len(banner))

visible = as_of(full, cutoff, mask_undetermined_knockout=(mode != MODE_LIVE))
features, builder = build_features(visible)       # ordinary one-step-ahead features
frozen = build_frozen_features(visible)           # same matches, features frozen at each season's start (simulator horizon)
print(f"\\ncutoff {cutoff}: {int(visible['is_completed'].sum()):,} completed matches visible; "
      f"{len(features):,} feature rows x {features.shape[1]} columns")
display(features[features["is_completed"]].tail(5)[["utc_date", "home_team_id", "away_team_id", "result", "elo_home", "elo_away",
                                                    "elo_diff", "gf5_home", "ga5_home", "pts5_home", "rest_days_home"]])
''')
code('''
# Leakage test, executed on the real data: rewrite EVERY result in the second half of history; the first half's features
# must not change by a single bit (features only ever depend on earlier matches).
tampered = visible.copy()
cut_row = int(len(tampered) * 0.5)
order = tampered.sort_values(["utc_date", "match_id"]).index
late = order[cut_row:]
tampered.loc[late, ["home_goals_90", "away_goals_90", "home_goals_aet", "away_goals_aet"]] = 9
f_tampered, _ = build_features(tampered)
first_half = features.sort_values(["utc_date", "match_id"]).iloc[: int(len(features) * 0.40)]
same = f_tampered.set_index("match_id").loc[first_half["match_id"]]
from src.features import FEATURE_COLUMNS
pd.testing.assert_frame_equal(first_half.set_index("match_id")[FEATURE_COLUMNS], same[FEATURE_COLUMNS])
print("OK - features of earlier matches are unchanged when all later results are rewritten: no look-ahead.")
''')

md("""
## 6. Train baseline and ML models

Five learners are compared against a **no-skill climatology baseline** (the training-period H/D/A frequencies):

| model | what it is |
|---|---|
| `elo` | Elo win expectancy plus a one-parameter draw model (**baseline A**) |
| `logistic_elo`, `logistic` | multinomial logistic regression on Elo only / Elo + form (**B**) |
| `poisson_elo`, `poisson` | two Poisson regressions for home/away goals -> scoreline -> H/D/A (**D**) |
| `catboost` | gradient-boosted trees on all features (**C**, optional) |

Imputation and scaling are fitted inside each model on the **training rows only**. No hyper-parameter search is run: with a
couple of thousand matches, tuning would mostly fit noise.
""")
code('''
ev = pl.run_evaluation(features, frozen, cfg, cutoff)      # fits every model on every split / fold (~1 min with CatBoost)
split = ev["split"]
print("Chronological split (older -> newer, no overlap):")
display(split.describe())
print("Models:", list(ev["factories"]))
''')

md("""
## 7. Chronological validation

Two horizons are scored, because they answer different questions:

* **`one_step_ahead`** - features computed just before each match: how good is the model at forecasting the *next* match?
* **`frozen_state`** - features frozen at the *start* of the season, applied to every match of it. This is exactly how the
  tournament simulator predicts (ratings are fixed at the cutoff and used for matches weeks or months later), so the model
  for the simulation is **selected on this horizon**, using only rolling-origin folds that end *before* the test seasons.

Metrics: multiclass log loss (primary), multiclass Brier score, accuracy, and class-wise expected calibration error.
""")
code('''
fixed = ev["fixed"].metrics
tbl = (fixed[fixed["split"].isin(["validation", "test"])]
       .pivot_table(index="model", columns=["horizon", "split"], values="log_loss"))
print("Log loss on the fixed validation / test seasons (lower is better):"); display(tbl.round(4))
print("Test-season sizes:", {s: int(len(p)) for s, p in (("validation", split.val), ("test", split.test))})
''')
code('''
from src.evaluation import summarise_rolling, HORIZON_FROZEN, HORIZON_ONE_STEP
last_val = ev["last_val_season"]
sel = ev["selection"]
pooled = {h: summarise_rolling(ev["rolling"][h].folds, last_val) for h in (HORIZON_ONE_STEP, HORIZON_FROZEN)}
print(f"Rolling-origin (expanding window), folds for seasons <= {season_label(last_val)} "
      f"({int(pooled[HORIZON_FROZEN]['n_test_matches'].iloc[0])} out-of-sample matches per model):")
display(pooled[HORIZON_FROZEN][["model", "log_loss", "brier", "accuracy", "ece"]].round(4).rename(columns={"model": "model (frozen_state)"}))
display(pooled[HORIZON_ONE_STEP][["model", "log_loss", "brier", "accuracy", "ece"]].round(4).rename(columns={"model": "model (one_step_ahead)"}))
from src.plots import plot_model_comparison
plot_model_comparison(pooled[HORIZON_FROZEN], sel.chosen, "Out-of-sample log loss at the simulator horizon"); plt.show()
''')
code('''
print("MODEL SELECTION  (rule:", sel.rule + ")"); display(sel.table.round(4))
for n in sel.notes:
    print(" -", n)
print("\\nCaveats:")
for c in ev["caveats"]:
    print(" -", c)
''')
code('''
from src.evaluation import calibration_table, expected_calibration_error
from src.plots import plot_calibration
y_pool, p_pool = ev["rolling"][HORIZON_FROZEN].pooled_predictions(sel.chosen)
cal = calibration_table(y_pool, p_pool, n_bins=6)
print(f"Calibration of '{sel.chosen}' on {len(y_pool)} pooled out-of-sample predictions (equal-count bins); "
      f"class-wise ECE = {expected_calibration_error(y_pool, p_pool, 6):.3f}")
display(cal.round(3).pivot_table(index="bin", columns="class", values=["mean_predicted", "observed_frequency"]).round(3))
cal_png = plot_calibration(cal, cfg.paths.outputs / "calibration.png", title=f"Calibration of the '{sel.chosen}' model",
                           subtitle=f"Simulator horizon, out-of-sample, {len(y_pool)} matches")
display(Image(filename=str(cal_png)))
print("Reading it: points near the diagonal mean the stated probability matches how often the outcome happened. "
      "Bins hold ~200 matches, so deviations of a few points are within noise; no claim of exact calibration is made.")
''')

md("""
## 8. Select and fit the model using only information available by the cutoff

The selected model is re-fitted on every completed match before the cutoff - nothing later. The team state (Elo ratings,
form) used by the simulator is the snapshot after the last of those matches.
""")
code('''
chosen = sel.chosen
model = pl.fit_final_model(features, ev["factories"], chosen)
train_rows = features[features["is_completed"]]
assert train_rows["utc_date"].max() < cutoff, "training data must end before the cutoff"
print(f"selected model: {chosen!r}")
print(f"fitted on {len(train_rows):,} matches from {train_rows['utc_date'].min():%Y-%m-%d} to {train_rows['utc_date'].max():%Y-%m-%d} "
      f"(cutoff {cutoff:%Y-%m-%d %H:%M})")
ratings = pd.Series(builder.elo.ratings).sort_values(ascending=False)
names = pl.build_team_directory(visible).set_index("team_id")["team_name"]
display(pd.DataFrame({"club": ratings.head(10).index.map(names), "elo": ratings.head(10).values.round(0).astype(int)}))
''')

md("""
## 9. Simulate the remaining tournament

For each of the simulated tournaments: every **completed** league-phase result is kept as it is, only unplayed matches are
simulated; standings follow the official tie-breakers; positions 1-8 go to the round of 16, 9-24 to the play-offs, 25-36 are
out; the knockout draw is sampled from the outcomes consistent with the pairings already known; two-legged ties are
aggregated (no away-goals rule) with extra time and penalties; the final is a single neutral match.
""")
code('''
if mode == MODE_SCENARIO:
    setup = build_scenario_setup(cfg.scenario_pots, model, builder, season=season, seed=cfg.seed, label=label)
else:
    setup = build_setup_from_matches(visible, season, model, builder, mode=mode, label=label, cutoff=cutoff)

print(f"{setup.label}\\n  league matches already played: {int(setup.league_played.sum())} / 144 | "
      f"known knockout legs: {setup.known_ko.n_known_legs}")
sim = simulate_tournament(setup, SimConfig(n_sims=cfg.n_sims, seed=cfg.seed))
print(f"{sim.n_sims:,} simulated tournaments | seed {sim.seed} | knockout draw outcomes consistent with the data: {sim.n_draw_outcomes}")
for team, note in sim.notes["team_notes"].items():
    print(f"  note for {team}: {note}")
''')
code('''
sens = pl.model_sensitivity(setup, builder, features, ev["factories"], [m for m in ev["factories"] if m != "climatology"],
                            chosen, cfg, sim) if cfg.run_sensitivity else None
replay_check = pl.replay_check(full, season, sim) if mode == MODE_REPLAY else None
''')

md("## 10. Championship probabilities")
code('''
print("=" * len(banner)); print(banner); print("=" * len(banner))
champ = sim.champion_table()
shown = champ.head(20).assign(
    probability=lambda d: (100 * d["champion_probability"]).map("{:.1f}%".format),
    interval_95=lambda d: [f"{100*a:.1f}% - {100*b:.1f}%" for a, b in zip(d["ci95_low"], d["ci95_high"])])
display(shown[["rank", "team_name", "simulated_championships", "probability", "interval_95"]].set_index("rank"))
print(f"sum of probabilities over all {len(champ)} clubs = {champ['champion_probability'].sum():.6f}")
print("\\nHow to read the interval:", pl.CI_INTERPRETATION)
''')
code('''
plots_path = pl.plot_champion_probabilities(champ, cfg.paths.outputs / "champion_probabilities.png",
                                            title="Champions League winner probabilities", banner=banner, n_sims=sim.n_sims)
display(Image(filename=str(plots_path)))
''')
code('''
q = sim.qualification_table().head(16)
cols = ["team_name", "p_top8_direct_r16", "p_playoff", "p_eliminated", "p_reach_quarter_final", "p_reach_final", "p_champion"]
display(q[cols].assign(**{c: (100 * q[c]).round(1) for c in cols[1:]}).rename(columns=lambda c: c.replace("p_", "% ")))
if sens is not None:
    print("Sensitivity to the model choice - championship probability under every candidate model (top 10 by the selected model):")
    display((sens.head(10).set_index("team_name").drop(columns=["team_id", "selected_model"]) * 100).round(1))
    print("A wide spread across models for a club means its probability depends on modelling choices that out-of-sample "
          "validation cannot separate; Monte Carlo intervals above do not capture this.")
if replay_check:
    print("\\nPost-hoc replay check (an anecdote, not validation):", replay_check)
''')

md("## 11. Save predictions and model artifacts")
code('''
result = pl.PipelineResult(cfg, visible, provenance, features, ev, chosen, setup, sim, mode, label, banner, cutoff, mode_notes,
                           replay_check, sens, model, builder)
written = pl.save_outputs(result, started_at=None)
for name, path in written.items():
    print(f"{name:34s} {path.relative_to(ROOT)}   ({path.stat().st_size / 1024:,.0f} KB)")
''')
code('''
import json
meta = json.loads((cfg.paths.outputs / "run_metadata.json").read_text(encoding="utf-8"))
print("forecast mode :", meta["forecast"]["mode"], "| live forecast:", meta["forecast"]["is_live_forecast"])
print("selected model:", meta["evaluation"]["selected_model"])
print("data          :", meta["data"]["n_completed"], "completed matches,", meta["data"]["n_teams"], "clubs")
print("assumptions   :")
for a in meta["forecast"]["assumptions"]:
    print("  -", a)
''')

md("""
## Limitations you should keep in mind

* **Sample size.** About 2,000 matches in 15 seasons; log-loss differences between the candidate models are within one or two
  standard errors, so the models cannot be reliably ranked - which is why the simplest adequate model is chosen.
* **Only Champions League matches are observed.** No domestic-league form, squad values, injuries, xG or coefficients, so
  clubs with little Champions League history are rated at the default and flagged in the output.
* **Ratings are frozen at the cutoff** during the simulation and scores are independent Poisson draws; real tournaments
  have momentum, injuries and tactics that this ignores.
* **Intervals are Monte Carlo noise only.** They shrink with more simulations; the uncertainty about the model itself is
  larger and is shown separately by the sensitivity table.
* **Tie-break fallbacks.** Disciplinary record and club coefficient are unavailable, so remaining ties are broken at random.
* A single replayed season (e.g. "PSG won and the model gave them X%") proves nothing about calibration.
""")


def build(execute: bool) -> Path:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}, "colab": {"provenance": []}}
    for kind, src in CELLS:
        nb["cells"].append(nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src))
    if execute:
        from nbclient import NotebookClient

        NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    return OUT


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="run the notebook and store its outputs")
    path = build(ap.parse_args().execute)
    print("wrote", path)
