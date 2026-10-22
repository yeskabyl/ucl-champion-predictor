# UEFA Champions League 2026/27 - champion probability model

A reproducible ML + Monte Carlo project that estimates **each club's probability of winning the UEFA Champions League**.
It collects real match results, builds leakage-safe features, compares models with **chronological** validation, and simulates
the 36-team league phase and the knockout bracket thousands of times.

> **The output is a ranked probability table, not a claim to know the winner.** Every output file and chart is stamped with the
> *forecast mode* that produced it (`LIVE_FORECAST`, `HISTORICAL_REPLAY` or `SCENARIO`), and a replay is never labelled as a
> forecast of the current season.

## Status - what has and has not been verified

| | |
|---|---|
| **Verified** | 228 automated tests pass (offline; Python 3.12 + pandas 3.1, **Python 3.11 with the minimum versions in `requirements.txt`**, and a Colab-like stack: pandas 2.2 / numpy 2.0 / scikit-learn 1.6). The full pipeline and the notebook were executed end-to-end on **real data** (15 seasons, 1,997 matches). The simulator's standings and bracket were checked against the real 2024/25 and 2025/26 seasons (see [Verification](#verification)). |
| **Not verified** | **The live football-data.org API.** No API token was available while building this, so the client is tested against scripted fake responses only (success, 429/5xx/timeouts/403/404/bad JSON, caching) and its field handling follows the provider's documentation, not a real response. The notebook has a cell that prints the real response schema as soon as you run it with a token. The 2026/27 season itself therefore has **not** been forecast: as of the last run the official participants/fixtures were not in any loaded data, so the notebook fell back to a labelled historical replay. |
| **Not run** | Google Colab itself (no access from here). Compatibility was checked with the Colab-like environment above. |

## Quick start

### Google Colab (no GPU needed)
1. Zip the project folder and upload `ucl-champion-predictor.zip` when the first notebook cell asks (or set `REPO_URL` to clone from GitHub).
2. *(Optional, for live 2026/27 data)* get a free token at <https://www.football-data.org/client/register> and add it as a Colab **secret** named `FOOTBALL_DATA_API_TOKEN`.
3. Open `notebooks/UCL_Champion_Predictor.ipynb` -> *Runtime -> Run all* (about 1-2 minutes with cached data).

### Local
```bash
python -m venv ../ucl-venv          # keep the venv outside OneDrive/Dropbox folders: it holds thousands of small files
../ucl-venv/Scripts/activate        # Windows (use `source ../ucl-venv/bin/activate` on macOS/Linux)
pip install -r requirements.txt
pytest -q                            # 228 tests, ~1.5 minutes, no network needed
python -m src.pipeline               # auto mode; add --mode replay --replay-cutoff 2025-09-01 --n-sims 10000 ...
python scripts/build_notebook.py --execute   # regenerate + run the notebook (needs nbclient and ipykernel)
```
Without a token the pipeline uses the public-domain openfootball data and produces a historical replay.
With `FOOTBALL_DATA_API_TOKEN` set (environment variable or a git-ignored `.env`, see `.env.example`) it requests every season
from the API first and only falls back to openfootball for seasons the API cannot supply.

## What you get

`python -m src.pipeline` (or the notebook) writes:

| file | content |
|---|---|
| `data/processed/matches.csv` | normalised match table exactly as visible at the cutoff (source + retrieval time on every row) |
| `outputs/champion_probabilities.csv` | team, simulated championships, probability, Monte Carlo interval, forecast mode |
| `outputs/qualification_probabilities.csv` | P(top 8) / P(play-off) / P(eliminated) / P(reach R16, QF, SF, final) / P(champion) |
| `outputs/final_standings_probabilities.csv` | P(finishing in each league-phase position 1..36), expected position |
| `outputs/champion_probabilities_by_model.csv` | the same probabilities under every candidate model (model-choice sensitivity) |
| `outputs/model_metrics.csv` | log loss, Brier, accuracy, calibration error for every model, split, fold and horizon, with sizes and dates |
| `outputs/calibration_table.csv`, `calibration.png` | reliability diagnostics of the selected model |
| `outputs/champion_probabilities.png` | chart with 95% Monte Carlo whiskers and the mode banner |
| `outputs/elo_ratings_at_cutoff.csv`, `model.joblib` | the ratings and fitted model used for the simulation |
| `outputs/run_metadata.json` | mode, cutoff, data provenance, exact split dates and sizes, selection table, assumptions, package versions, file hashes |

The probabilities always sum to 1 (checked at save time). **How to read the interval:** `ci95_low/high` is a Wilson 95% interval
for the share of simulated tournaments won. It measures **only simulation noise** (it shrinks as `n_sims` grows). It does *not*
include uncertainty about the model, the ratings, or the draw, so the real uncertainty is larger.

### Example output (historical replay, **not** a 2026/27 forecast)
Replay of 2025/26 as of 2025-09-01 - the model sees only matches before that date; 10,000 simulations, seed 42, selected model `elo`:

| rank | club | titles / 10,000 | probability | 95% MC interval |
|---:|---|---:|---:|---|
| 1 | Real Madrid | 1422 | 14.2% | 13.5-14.9% |
| 2 | Bayern München | 1330 | 13.3% | 12.6-14.0% |
| 3 | Paris Saint-Germain | 1266 | 12.7% | 12.0-13.3% |
| 4 | Liverpool | 1193 | 11.9% | 11.3-12.6% |
| 5 | Manchester City | 961 | 9.6% | 9.0-10.2% |

(After the fact PSG won; the model had given them 12.7%, rank 3. One season says nothing about calibration.)
The sensitivity table shows how much the model choice moves such numbers - e.g. PSG ranges from 11.8% to 25.3% across candidate
models - which is exactly why the interval above should not be read as the full uncertainty.

## Project layout

```
src/data_loader.py   football-data.org client + openfootball parser, snapshots, validation, as_of() cutoff
src/features.py      leakage-safe features (one code path for history, fixtures and simulator pairings)
src/elo.py           Elo rating system
src/models.py        climatology baseline, Elo, logistic, Poisson goals, CatBoost; probability hygiene
src/evaluation.py    metrics, chronological split, rolling-origin evaluation, calibration, model selection
src/simulator.py     standings, bracket/draw space, two-legged ties, Monte Carlo engine, setup builders
src/pipeline.py      orchestration + CLI + output writing        src/plots.py   charts       src/utils.py   helpers
notebooks/UCL_Champion_Predictor.ipynb   the 11-step notebook (generated by scripts/build_notebook.py)
tests/               228 tests; tests/fixtures holds 15 real, unmodified CC0 season files so tests run offline
```

## How it works

### 1. Data
* **football-data.org v4** (`/competitions/CL/matches?season=YYYY`, header `X-Auth-Token`; `season=2026` is 2026/27). The client uses
  timeouts, exponential back-off for network errors and 5xx, honours the server's "Wait N seconds" on HTTP 429, keeps a
  local 10-requests-per-minute limiter (the free-tier limit as I understand it - check your plan), and raises typed errors for
  401/403/404/bad JSON. Raw responses are snapshotted in `data/raw/` with URL, retrieval time and SHA-256.
* **Score semantics.** football-data's `fullTime` already **includes** extra-time and shoot-out goals, so the 90-minute result used
  as the label comes from `regularTime` or `fullTime - extraTime - penalties`; if neither can be derived the match is *not* treated
  as completed (never guessed). Live/in-play scores are never carried into the table.
* **Fallback source:** openfootball (`github.com/openfootball/champions-league`, CC0) provides 2011/12-2025/26. Its notation
  `3-2 a.e.t. (3-0, 1-0)` is parsed as *after extra time (90 minutes, half-time)*; all 15 files parse with zero unparsed lines.
  **It has no 2026/27 file**, so the live season needs the API. Both sources feed one normalised schema; a source is chosen per
  season, never mixed row by row.
* **Identity.** Club names differ between sources and seasons ("Inter" / "FC Internazionale Milano"); a normaliser plus a reviewed
  alias table maps 136 raw (name, country) spellings onto 108 stable team ids (`find_possible_duplicate_teams` flags suspects).
* **Cleaning.** Duplicate ids, duplicated fixtures, a team playing itself, negative/missing scores are removed and **reported**;
  unavailable seasons are listed with the reason. No synthetic data exists outside unit tests.

### 2. Features (all known before the match)
Elo ratings (K=20, home advantage 65, goal-difference multiplier, 10% shrink between seasons), goals scored/conceded over the previous
5 and 10 matches, points over the previous 5, strength of recent opponents, days since the previous Champions League match,
venue/neutral and knockout flags, plus home-minus-away differences. Features come from team state frozen **before** the match and
the state is updated **after**. Rolling windows exclude the current match by construction. Caveat: only Champions League matches
are observed, so "form" and "rest" ignore domestic games; xG and domestic-league strength are deliberately not used because no
reliable source is wired in.

### 3. Models (all share one interface: `predict_proba` -> H/D/A, rows sum to 1, never exactly 0)
`elo` (Elo expectancy + a one-parameter draw model) - baseline A; `logistic` / `logistic_elo` (multinomial logistic regression on Elo+form / Elo only) - B;
`catboost` (optional) - C; `poisson` / `poisson_elo` (two Poisson goal regressions -> scoreline matrix -> H/D/A) - D;
`climatology` (training-period class frequencies) as the no-skill yardstick. Imputation/scaling are fitted on training rows only
(tested). No hyper-parameter tuning: with ~2,000 matches it would mostly fit noise.
Outcome-only models get goal rates for the simulator by solving for Poisson rates that reproduce the model's P(H)-P(A) at the
training-average total goals - an explicit assumption.

### 4. Honest evaluation
* Split by **season order**, never shuffled. For the replay at 2025-09-01: train 2011/12-2020/21 (1,244 matches), validation 2021/22-2022/23 (250), test 2023/24-2024/25 (314). The exact dates and sizes are in `model_metrics.csv` and `run_metadata.json`.
* Rolling-origin (expanding window) folds: train on all earlier seasons, test on the next one.
* Metrics: multiclass log loss (primary), multiclass Brier, accuracy, class-wise expected calibration error, always with the sample size.
* **Two horizons.** `one_step_ahead` = features computed just before each match. `frozen_state` = features frozen at the *start* of the season,
  which is how the simulator predicts (ratings are fixed at the cutoff and reused for months). Fast-moving form features go stale at
  that horizon, so the simulation model is **selected on the frozen-state folds that end before the test seasons**, using a
  one-standard-error rule (simplest model within one SE of the best). The test seasons are never used to choose.

Results for the replay (log loss, lower is better; no-skill climatology = 1.020 on the test seasons):

| model | one-step, test (n=314) | frozen-state, test (n=314) | frozen-state, selection folds (n=869) |
|---|---:|---:|---:|
| **elo** (selected) | 0.938 | 0.957 | 0.961 |
| poisson | 0.932 | 0.966 | 0.963 |
| catboost | 0.945 | 0.952 | 0.979 |
| logistic | 0.974 | 0.970 | 0.965 |
| climatology | 1.020 | 1.020 | 1.061 |

**What this does and does not show.** Every model beats the no-skill baseline out of sample (by roughly 0.04-0.09 log loss, depending on model and period). Differences
*between* models are 0.00-0.03 and, with a few hundred matches (SE roughly 0.03), **are not statistically distinguishable** - the
model ranking even changes between validation and test. So the project does not claim a best model; it picks the simplest adequate
one and reports the spread (the sensitivity table). Calibration of the selected model on 1,183 pooled out-of-sample predictions
has class-wise ECE 0.038 with bins of ~200 matches; points scatter around the diagonal (`calibration.png`). That is *consistent with*
reasonable calibration but too noisy to certify it, and no claim of calibration is made.

### 5. The simulator (rules and sources)
| rule | implementation |
|---|---|
| League phase | 36 clubs, 8 matches each (4 home/4 away), 144 matches; fixtures validated (4H/4A, no repeats) |
| Ranking | Article 18 order: points, goal difference, goals scored, away goals, wins, away wins, opponents' points, opponents' goal difference, opponents' goals; then club coefficient if supplied, else random (disciplinary points and coefficients are not available) |
| Zones | 1-8 round of 16; 9-24 knockout play-offs; 25-36 eliminated |
| Play-offs | 9/10 v 23/24, 11/12 v 21/22, 13/14 v 19/20, 15/16 v 17/18 (draw inside each block); two legs, seeded club hosts the return leg |
| Round of 16 | seeds 1/2 face the 15/16-v-17/18 winners, 3/4 the 13/14-v-19/20, 5/6 the 11/12-v-21/22, 7/8 the 9/10-v-23/24; seeds of a pair sit in opposite halves |
| QF / SF / final | block A v D and B v C inside a half; semi-final pairs the two quarter-final winners of a half; final = one neutral match. Seeds 1-4 host QF return legs, seeds 1-2 host SF return legs (Art. 19.04) |
| Level ties | aggregate over two legs (no away-goals rule), then extra time (1/3 of the 90-minute rates, at the second leg's ground), then a 50/50 shoot-out |

Official sources: UEFA Regulations 2026/27 Articles 18 and 19. **The Annex B bracket diagrams could not be read**, so the quarter-/semi-final
tree was derived from Article 19's block structure and **checked against the real brackets** (below). The 2026/27 draw itself is
not assumed anywhere: when official fixtures are absent the code raises `OfficialDataUnavailable`, and scenario mode needs
user-supplied participants and labels its schedule as generated.

What is simulated: unplayed league matches (completed results are kept, never re-drawn), standings, the knockout draw (sampled uniformly from
the 4,096 outcomes consistent with whatever pairings are already known - so a simulation started after the play-off draw respects it), every
tie, extra time and penalties. Reproducible with `seed`; `n_sims` defaults to 10,000.

**Modes.** `LIVE_FORECAST`: the target season has all 144 official fixtures in the data. `HISTORICAL_REPLAY`: a past 36-team season is replayed as of a cutoff; results
after it are hidden *and so are the participants of knockout ties not yet determined*. `SCENARIO`: pass four pots of nine clubs
(`--scenario-json pots.json`); the league schedule is then a random pot-respecting assumption, flagged in the output.

## Verification
* **Real brackets.** For 2024/25 and 2025/26 my standings code reproduces the real play-off field, and exactly **2 of the 4,096** draw outcomes (the draw and its mirror
  image) reproduce all 23 real knockout pairings - so the bracket structure is pinned down by data, not assumed. Re-simulating the real seasons from several cutoffs (after the league phase, the play-offs, the round of 16, the quarter-finals; 2025/26 at four
  cutoffs, 2024/25 at one) shows that for every tie already known exactly one club advances in every simulation, i.e. the simulator's wiring reproduces the real bracket.
* **Independent reference.** `tests/test_simulator_reference.py` re-implements the whole tournament one simulation at a time from the rules text and requires the
  vectorised simulator to agree on title, final, semi-final, quarter-final, round-of-16, top-8 and elimination probabilities (5-sigma).
* **Mutation checks.** Five deliberate bracket bugs (wrong quarter-final pairing, semi-final crossing halves, wrong play-off seeding, ...) were each caught by the suite.
* **Standings.** Each Article 18 criterion is isolated in a hand-built case (both ways round), and the vectorised table is compared with an independent pure-Python ranking on random leagues full of ties.
* **Leakage.** Tests rewrite all future results and assert earlier features are bit-identical; the notebook repeats this on the real data. A *spy model* proves rolling-origin folds never train on the season they test.

## Honest limitations
* **No live API run** (see Status). Field names for football-data.org are taken from its documentation; run the notebook once with your token and read the schema printout.
* **Sample size.** ~2,000 matches; models cannot be reliably ranked; calibration bins are small.
* **Only Champions League data.** Clubs with no earlier CL match are rated at the default and flagged; no domestic form, squad value, injuries or xG.
* **Frozen state, Poisson scores.** No in-tournament updating of ratings, no momentum; independent Poisson goals understate draw correlation.
* **Model-choice uncertainty is large** for clubs whose probability depends on form features (see the sensitivity table); Monte Carlo intervals do not capture it.
* **Tie-break fallback.** Disciplinary record and club coefficients are unavailable, so remaining ties are random (rare).
* **Hosting order.** Return-leg hosting follows the 2026/27 rule, which holds for every tie of 2025/26 but not for the 2024/25 quarter/semi-finals; it only affects second-leg/extra-time home advantage.
* openfootball times are kick-off times in CET/CEST converted to UTC; times only matter for ordering and rest days.

## Design decisions worth explaining in an interview
1. **A replay leak I found and fixed.** Hiding scores after the cutoff was not enough: future knockout fixtures still carried team names, which would have told the simulator the play-off pairings in September. `as_of(..., mask_undetermined_knockout=True)` now models what a provider would have shown.
2. **Selecting for the way a model is used.** The first selection picked a model with form features that gave PSG 25%. A control run showed the form features drove it, and the real problem: they were validated one match ahead but used frozen for a season. Evaluating at the simulator's horizon changed the choice (to the plain Elo model) and is now part of the pipeline.
3. **One-standard-error rule** because the differences between models are inside the noise; the report says so instead of crowning a winner.
4. **Conditioning on the known draw** by enumerating all 4,096 draw outcomes and filtering - exact, tiny, and it makes the simulator valid from any cutoff.
5. **One feature code path** (`FeatureBuilder`) for training rows, upcoming fixtures and the simulator's hypothetical pairings, so there is no train/serve skew.
6. **Bugs the tests caught:** snapshot metadata recorded the folder name instead of the provider; a Wilson bound of `1e-19` instead of 0; corrupt snapshots crashed instead of refetching; the 2024/25 return-leg hosting rule differing from 2025/26.

## Updating data
Re-run the notebook/pipeline: completed seasons are served from `data/raw/` snapshots; the current season is re-fetched when its snapshot is older than 6 hours
(`refresh=True` forces everything). New results change the cutoff, the Elo state, the set of known league results and, once the play-off draw is in the data, the conditioned bracket.

## Data licences and attributions
openfootball/champions-league: CC0 1.0. football-data.org: subject to its terms for your plan. UEFA regulations: UEFA documents site (read for Articles 18-19). Club and competition names belong to their owners.
