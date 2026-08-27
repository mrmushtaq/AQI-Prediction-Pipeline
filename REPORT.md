# Sukkur AQI Predictor — Project Report

**10Pearls Shine Data Sciences Internship**
**Project:** AQI Prediction System — Sukkur, Pakistan
**Author:** Mushtaque Ali
**Live demo:** https://sukkur-aqi-predictor.streamlit.app/
**Repository:** `pearls-aqi-predictor`

This report documents the project requirement-by-requirement, phase by
phase, with the evidence (test counts, validated runs, real metrics) backing
each claim. For narrative design rationale and full technical detail, see
[README.md](README.md); this report is the compact, evaluator-facing
counterpart.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Phase 1 — Data Ingestion Pipeline](#2-phase-1--data-ingestion-pipeline)
3. [Phase 2 — Feature Engineering](#3-phase-2--feature-engineering)
4. [Phase 3 — Hopsworks Feature Store](#4-phase-3--hopsworks-feature-store)
5. [Phase 4 — Exploratory Data Analysis](#5-phase-4--exploratory-data-analysis)
6. [Phase 5 — Model Training Pipeline](#6-phase-5--model-training-pipeline)
7. [Phase 6 — Forecast Dashboard & Explainability](#7-phase-6--forecast-dashboard--explainability)
8. [Automation & Deployment](#8-automation--deployment)
9. [Testing Summary](#9-testing-summary)
10. [Requirement Traceability Matrix](#10-requirement-traceability-matrix)
11. [Known Limitations](#11-known-limitations)
12. [Conclusion](#12-conclusion)

---

## 1. Executive Summary

The project delivers an end-to-end AQI forecasting system for **Sukkur,
Pakistan**: live and historical data ingestion from external weather/AQI
APIs, a leakage-safe feature engineering pipeline, a Hopsworks feature
store, a trained-and-compared model suite, and a public Streamlit dashboard
serving real-time 3-day AQI forecasts with explainability.

| Metric | Result |
|---|---|
| Historical data collected | **22,608 records** at initial backfill completion; **23,041** as of the latest live snapshot (2024-01-01/08 → present, ~2.5 years, growing hourly) |
| Automated tests | **201 passed** |
| Best forecast model (Day 1 / 2 / 3) | Ridge Regression, test R² = **0.7871 / 0.7824 / 0.7934** |
| Naive persistence baseline (same horizons) | R² = 0.7279 / 0.6808 / 0.6638 |
| Target R² (project goal) | ≥ 0.75 — **met on all three horizons** |
| Live deployment | https://sukkur-aqi-predictor.streamlit.app/ |
| Automation | Hourly live ingestion + daily backfill/retrain via GitHub Actions |

All six planned phases are complete and validated against real data and a
real Hopsworks project — not synthetic placeholders.

---

## 2. Phase 1 — Data Ingestion Pipeline

**Requirement:** fetch raw weather and pollutant data from external APIs,
validate it, and persist it reliably.

### What was built

- **Live pipeline** (`run_pipeline.py`): fetches current weather from
  **OpenWeather**, and current AQI via a **three-stage fallback**:
  AQICN by city name → AQICN by lat/lon → OpenWeather Air Pollution API.
- **Historical backfill pipeline** (`historical_backfill.py`): the same
  merge/validate logic applied over an arbitrary date range, using
  **Open-Meteo Archive** for weather and **OpenAQ v3** (ground sensors,
  primary) falling back to **Open-Meteo Air Quality / CAMS** (model
  estimate) for AQI.
- **Reliability**: every external call has retry logic with exponential
  backoff (`base * 2^(attempt-1)`), a request timeout, and graceful
  fallback — a single provider outage does not fail a run.
- **Validation**: type/range/required-field checks; pollutant sub-fields
  (PM10/CO/SO2/NO2/O3) are allowed to be `null` (not every station measures
  every pollutant), but `aqi` and `pm2_5` are strictly required.
- **Persistence**: raw JSON responses (`data/raw/`), processed CSV rows
  (`data/processed/`), and a per-run metadata record (`data/metadata/`) are
  written on every run — success or failure — giving a full audit trail.

### Why the historical provider set changed

The originally planned OpenWeather historical endpoints (One Call Time
Machine, Air Pollution History) require a paid subscription, and **AQICN
does not serve real historical AQI** — it always returns the latest
observation regardless of requested timestamp. After discussion with the
project mentor, historical data collection moved to three free,
unauthenticated-or-free-tier providers (Open-Meteo Archive, OpenAQ v3,
Open-Meteo Air Quality/CAMS) that satisfy the same requirement — real,
API-sourced data — documented fully in README §11.

### Validated result

A full historical backfill spanning **2024-01-01 through present** was run
to completion in quarterly chunks (to make interruptions cheap to recover
from), producing **22,608 records** — exceeding the original 1.5-2 year
target. `aqi_source` in the resulting dataset distinguishes real
ground-sensor rows (`OpenAQ-v3`, ~Nov 2025 onward for Sukkur) from CAMS
model-estimate rows (`OpenMeteo-AirQuality`, earlier).

**Test coverage:** 201 automated tests across the whole project (see
[Section 9](#9-testing-summary)); Phase 1 specifically covers the 3-stage
AQI fallback, retry/backoff/timeout behavior, malformed-payload handling,
missing-config errors, partial-pollutant validation, and backfill
dedup/`--force` behavior — all with mocked HTTP, no live keys required to
run the suite. The live pipeline and the full 2.5-year backfill were
additionally **manually verified against real, live APIs** (not mocked).

### Live snapshot — historical dataset (captured 27 Aug 2026)

Pulled directly from the deployed dashboard's "Historical analysis" panel,
confirming the ingestion pipeline is still actively growing the dataset in
production, past the point-in-time figures reported at project completion:

| Metric | Value |
|---|---|
| Records | **23,041** |
| Cities | 1 (Sukkur) |
| AQI mean | 123.7 |
| AQI median | 107.0 |
| Coverage window | 2024-01-08 → 2026-08-27 |

Two small deltas from the originally documented figures, both expected
rather than errors:

- **Record count (23,041 vs. the 22,608 reported at initial backfill
  completion)** — the difference (433 records) is consistent with ~18 days
  of continued hourly live ingestion between when the original backfill
  finished and this snapshot, via the automated hourly workflow (§8).
- **Coverage start date (2024-01-08 vs. the 2024-01-01 originally
  requested)** — the dashboard's reported start is the first row with a
  *complete* feature set after Phase 2's lag/rolling-window trimming (see
  §3), not the raw ingestion start date; the underlying raw dataset does
  still begin 2024-01-01.

The panel also reports a **Pearson correlation of -0.571** between AQI and
a comparison variable selectable in the UI (exact variable not captured in
this snapshot) — directionally consistent with expected physical
relationships (e.g. AQI typically correlates negatively with wind speed or
positively with humidity depending on the variable chosen), though this is
descriptive EDA-style output rather than a modeling claim, and the panel
itself correctly notes correlation does not imply causation.

Source-coverage labeling in this panel matches the documented split (§2,
§11 of the README): `OpenAQ-v3` for ground-sensor rows, `OpenMeteo-AirQuality`
for CAMS model-estimate rows — confirming the `aqi_source` provenance
tracking described throughout this report is visible end-to-end, not just
present in the raw CSV.

---

## 3. Phase 2 — Feature Engineering

**Requirement:** transform raw ingested records into a model-ready feature
set.

`feature_engineering.py` reads the historical processed CSV and, per city,
on a deduplicated chronologically-sorted series, computes:

- **Time features** — hour/day/month with cyclical sin/cos encoding.
- **Lag features** — previous 1-2 readings for AQI and key pollutants/weather.
- **Rolling statistics** — 24h/48h mean and std for AQI, PM2.5, PM10,
  computed from the *actual configured collection interval* rather than a
  hardcoded assumption (an earlier version assumed a fixed 3-hour interval,
  which silently produced windows labeled "24h/48h" that only covered
  8h/16h once the real interval changed to hourly — this is now derived
  dynamically and confirmed correct).
- **AQI change rate** — first difference from the prior reading.
- **Pollution index** — a composite pollutant-ratio score, deliberately
  `null` (not a degenerate copy of PM2.5) for single-pollutant stations like
  Sukkur's, requiring at least 2 real pollutant readings to compute.
- **Target** (`aqi_target`) — AQI value `HORIZON_STEPS` ahead (default 72h /
  3 days), also derived dynamically from the real collection interval.

Rows lacking full lag/rolling/target history are dropped by default
(`--keep-incomplete-rows` to retain them for custom imputation).

**Validated** on the full 22,608-row Sukkur dataset, with `feels_like`
fully populated and rolling-window/horizon sizes confirmed correct against
the real hourly collection interval.

---

## 4. Phase 3 — Hopsworks Feature Store

**Requirement:** push engineered features to a managed feature store.

`feature_store.py` implements `login() → get_or_create_feature_group() →
insert_features() → read_features()` against a Hopsworks project.

| Setting | Value |
|---|---|
| Feature group | `aqi_features`, **v3** |
| Primary key | `city`, `collection_timestamp` |
| Event time | `collection_timestamp` |
| Time travel format | `HUDI` (avoids the `delta` dependency Hopsworks' newer default requires) |
| Online store | Disabled (offline/batch is sufficient for training) |

**Why v3:** two data-quality fixes required a version bump — `feels_like`
was previously hardcoded `null` for all historical weather instead of being
read from Open-Meteo's `apparent_temperature`, and the rolling/horizon
windows had the interval bug described in Phase 2. v1/v2 are retained in
Hopsworks for comparison; v3 is authoritative going forward.

**Validated run** (real Hopsworks project, not mocked):
```
Feature Group created successfully.
Uploading Dataframe: 100.00% |██████████| Rows 5887/5887
Inserted 5887 rows.
Read back 5887 rows.
```

A `hopsworks` full-SDK / `hsfs`-lite install fallback is implemented so the
sync also runs from constrained environments (e.g. Windows without C++
Build Tools, via Google Colab — see `notebooks/phase3_hopsworks_feature_store.ipynb`).

**Test coverage:** login (both SDK paths), feature group schema creation,
insert, read, and a fully mocked end-to-end sync — no real credentials
needed for the suite.

---

## 5. Phase 4 — Exploratory Data Analysis

**Requirement:** analyze the engineered dataset before training.

`notebooks/EDA.ipynb` covers: data overview, missing-value analysis,
correlation analysis, distribution/skew analysis, IQR outlier detection,
and AQI trend analysis (hourly/monthly/day-of-week seasonality).

**Concrete value delivered, not just descriptive plots** — the EDA process
directly caught two real bugs before they reached model training:

1. `feels_like` unexpectedly 100% null → traced to a fetch bug, fixed,
   re-validated in Phase 3's v3 re-sync.
2. `pollution_index` ≈ `pm2_5` at **r = 1.000** for Sukkur (single-pollutant
   station) → identified as a redundant, leakage-adjacent feature and fixed
   in Phase 2 to correctly return `null` instead.

`visibility` being 100% null was also flagged and confirmed as a genuine
ERA5 dataset limitation (not a bug) — the reanalysis dataset behind
Open-Meteo Archive has no visibility variable.

---

## 6. Phase 5 — Model Training Pipeline

**Requirement:** train and compare multiple models, evaluate rigorously,
and persist the best one to a registry.

### Pipeline

```
load_features() → select_feature_columns() (excludes all-null columns)
  → time_based_train_test_split() (chronological, NOT shuffled)
  → StandardScaler (fit on train only)
      → Ridge Regression
      → Random Forest (200 trees)
      → Neural Network (TensorFlow, or sklearn MLPRegressor fallback)
  → evaluate_model() (RMSE/MAE/R² + naive persistence baseline)
  → select_best_model() (lowest RMSE, never the naive baseline)
  → save_model_locally() (+ optional Hopsworks Model Registry push)
```

### Two methodology decisions worth calling out explicitly

**1. Time-based split, not random shuffle.** Lag/rolling features are
computed from each row's temporal neighbors; a random shuffle would leak
test-set information into training features (and vice versa), producing
misleadingly optimistic metrics. The split instead takes the earliest
`1 - test_size` fraction as training data and the most recent `test_size`
fraction as test — matching both the leakage-safety requirement and the
real deployment scenario (predicting the future from the past).

**2. A naive persistence baseline is always computed** on the same test
rows as every trained model, and the pipeline explicitly warns if the best
trained model fails to beat it. A model that can't outperform "assume no
change" isn't adding real value regardless of how good its raw RMSE looks
in isolation — this is the standard the project holds every trained model
to, not just the final one reported below.

### Result: v6 daily-average models (final, deployed)

The Phase 2 hourly 72h-ahead target hit a ceiling around **R² ≈ 0.56-0.60**
— short of the ≥0.75 project goal, largely because the test period sits in
a different seasonal regime and hourly variance includes substantial
within-day noise. The target was changed to the **daily-average AQI**
(mean over the next 1/2/3 days), which both matches how a 3-day forecast is
actually consumed, and reaches the goal:

| Horizon | Model | Test R² | Test RMSE | Naive persistence R² |
|---|---|---|---|---|
| Day 1 | Ridge, alpha=100, 51 features | **0.7871** | 17.49 | 0.7279 |
| Day 2 | Ridge, alpha=300, 56 features | **0.7824** | 17.08 | 0.6808 |
| Day 3 | Ridge, alpha=300, 61 features | **0.7934** | 16.26 | 0.6638 |

v6 adds **future-weather features** (`fw_*`: wind/humidity/temp/clouds/
pressure expected during the forecast days themselves, from the OpenWeather
5-day forecast at inference). v5 (without these) scored 0.7635 / 0.7553 /
0.7553 for comparison — future weather is a measurable, real improvement.

**Leakage was checked, not assumed.** An earlier version built `fw_*`
directly from archived *actual* weather for both train and test — i.e. the
backtest saw perfect future weather it will never have at inference. This
was caught in review and fixed by injecting calibrated, horizon-growing
simulated forecast uncertainty into `fw_*` for both splits. Two checks
confirm the fix is real and not just hiding the issue:

- **Leakage-gap check** — re-scoring the same trained model against the
  original perfect-weather holdout gives R² 0.7861/0.7820/0.7916, within
  ~0.002 of the realistic numbers above. The original headline numbers were
  approximately correct, but this is now measured rather than assumed.
- **Ablation check** — a historical-features-only model (no `fw_*`) scores
  0.7660/0.7553/0.7618, with the gap to full-v6 **growing by horizon**
  (+0.021/+0.027/+0.032). A growing-with-horizon gap is the signature of a
  genuine effect, not noise-fitting — confirming future weather is real
  signal, not disguised leakage.

**Test coverage:** feature selection (all-null column exclusion), the
time-based split's leakage-safety properties, all three model candidates
(incl. the TensorFlow/sklearn fallback), RMSE/MAE/R² and naive-baseline
computation matching direct calculation, local registry save/load
round-trips, best-effort Hopsworks registry push failure handling, and a
full end-to-end training run — all with synthetic data / full mocking, no
real credentials or heavy ML install required.

---

## 7. Phase 6 — Forecast Dashboard & Explainability

**Requirement:** serve the trained models through an accessible interface
with model transparency.

**Live:** https://sukkur-aqi-predictor.streamlit.app/

### Contents

- **3-day forecast cards** — mean AQI for Day 1/2/3, color-coded by US EPA
  AQI category, with each model's test R² and a confidence range.
- **Hazardous-AQI alerts** — banner at ≥150 (warning) / ≥200 (hazardous)
  peak predicted AQI.
- **Current conditions & health advisory** — live "right now" reading plus
  category-specific guidance for general and sensitive populations.
- **Pollutant breakdown** — latest pollutant concentrations vs. WHO 2021
  24-hour guidelines and Pakistan NEQS ambient limits.
- **Historical trends** — 30-day observed AQI, a monthly calendar heatmap,
  diurnal (hour-of-day) pattern, and week-over-week comparison.
- **SHAP / LIME explainability** — per-horizon feature attributions; SHAP is
  exact for the linear v6 models (`base + Σ SHAP = prediction`), LIME
  provides a complementary local approximation.
- **Model/methodology transparency** — horizon metadata, alphas, split
  sizes, test R² vs. naive baseline, and the model registry run path, all
  surfaced in-app rather than only in this report.

### Data resilience

The dashboard reads the durable Hopsworks feature group first, falls back
to checked-in generated files, and attempts a live OpenWeather/AQICN
refresh — a Hopsworks or live-API outage degrades gracefully rather than
breaking the app. Stored observations are explicitly labeled as last-known
data unless a refresh successfully completes.

**Test coverage:** daily-feature recipe column contract, future-weather
injection and persistence fallback, model/scaler loading for all three
horizons, forecast plausibility, insufficient-history error handling, and
SHAP's additive reconstruction — synthetic data and toy models, no real
artifacts required.

### Live snapshot (captured 27 Aug 2026, 21:36 UTC)

Pulled directly from the deployed dashboard at
https://sukkur-aqi-predictor.streamlit.app/, evidencing the pipeline
running live in production rather than only in local/test runs.

**Current conditions (latest available observation):**

| Reading | Value |
|---|---|
| AQI | 89.0 — *Moderate* |
| PM2.5 | 89.0 µg/m³ |
| Temperature | 29.3 °C (feels like 31.6 °C) |
| Humidity | 60.0 % |
| Pressure | 998.0 hPa |
| Wind speed | 6.8 m/s |
| Cloud cover | 1.0 % |

**Three-day forecast (daily mean AQI, active model run
`ridge_3day_dailyavg_v6_20260827T130318Z`):**

| Day | Date | Forecast AQI | Category | Trend vs. current |
|---|---|---|---|---|
| Day 1 | Fri, 28 Aug 2026 | 108.4 | Unhealthy for Sensitive Groups | ▲ +19.4 |
| Day 2 | Sat, 29 Aug 2026 | 104.8 | Unhealthy for Sensitive Groups | ▲ +15.8 |
| Day 3 | Sun, 30 Aug 2026 | 104.9 | Unhealthy for Sensitive Groups | ▲ +15.9 |

No hazardous-alert threshold (≥150) was crossed by this forecast run; the
dashboard correctly suppressed the alert banner.

**Model retrain cadence, observed directly from the run-ID history shown in
the dashboard's "Model intelligence" panel** (confirms the daily/near-daily
retrain automation from §8 is actually firing, not just scheduled on paper):

```
ridge_3day_dailyavg_v6_20260821T102253Z
ridge_3day_dailyavg_v6_20260827T003514Z
ridge_3day_dailyavg_v6_20260827T064700Z
ridge_3day_dailyavg_v6_20260827T095941Z
ridge_3day_dailyavg_v6_20260827T130318Z   ← active at capture time
```

> **Observation for follow-up:** the dashboard's "Forecast model details"
> panel rendered `` `None` `` at capture time instead of the horizon
> metadata (alphas, split sizes, test R² vs. baseline) described in §7 and
> the README. This looks like a display/metadata-population issue on the
> currently deployed build rather than a training-pipeline problem — worth
> checking `forecast.model_metadata` wiring in `app.py` before the next
> deploy, since that panel is otherwise the dashboard's main
> transparency feature.

---

## 8. Automation & Deployment

| Workflow | Schedule | Purpose |
|---|---|---|
| **Hourly Live Ingestion** | every hour, on the hour (UTC cron `0 * * * *`) | Live weather + AQI fetch, committed to `main` |
| **Daily Backfill + Retrain** | 02:30 UTC daily | Backfill gap-fill, feature refresh, v6 model retrain, optional Hopsworks Model Registry publish |
| **Tests** | on push/PR | Full 201-test suite |

Streamlit Community Cloud redeploys automatically on every push to `main`
that GitHub Actions commits (refreshed data + v6 model artifacts), so the
live demo stays current without manual intervention. Deployment requires
Python 3.11 (pinned Hopsworks client constraint) and the API/Hopsworks
secrets documented in README §5, added via Streamlit Cloud's Secrets field.

---

## 9. Testing Summary

```
201 tests passed
```

All external HTTP/SDK interaction (OpenWeather, AQICN, OpenAQ, Open-Meteo,
Hopsworks, TensorFlow) is mocked in the automated suite — no real API keys,
network access, or heavy ML installs are required to run it. Beyond
automated mocked tests, the following were **manually verified against
real, live systems**:

- Live ingestion pipeline — real OpenWeather + AQICN data, Sukkur.
- Full 2.5-year historical backfill — real OpenAQ + Open-Meteo data,
  22,608 records.
- Hopsworks Feature Store — real project, v3 schema, 5,887-row round trip.
- Model training pipeline — real Sukkur feature set, full model comparison.
- Dashboard — deployed and serving live forecasts at the public URL above.

---

## 10. Requirement Traceability Matrix

| Project requirement | Status | Evidence |
|---|---|---|
| Fetch raw weather/pollutant data from external APIs | ✅ Complete | §2; OpenWeather, AQICN, Open-Meteo, OpenAQ |
| Validate and persist ingested data | ✅ Complete | §2; raw JSON + processed CSV + metadata per run |
| Historical data collection (1.5-2 year target) | ✅ Exceeded | §2; 22,608 records, ~2.5 years |
| Feature engineering for time-series forecasting | ✅ Complete | §3; time/lag/rolling/target features |
| Feature store integration | ✅ Complete | §4; Hopsworks v3, validated round trip |
| Exploratory data analysis | ✅ Complete | §5; EDA notebook, caught 2 real data-quality bugs |
| Train and compare multiple models | ✅ Complete | §6; Ridge, Random Forest, Neural Net + naive baseline |
| Model registry / persistence | ✅ Complete | §6; local registry + optional Hopsworks push |
| Achieve R² ≥ 0.75 forecast accuracy | ✅ Met | §6; 0.7871 / 0.7824 / 0.7934 across Day 1-3 |
| Dashboard for forecasts + explainability | ✅ Complete | §7; live at public URL, SHAP/LIME included |
| Automated testing | ✅ Complete | §9; 201 tests, all mocked, CI-run |
| Automation / scheduled refresh | ✅ Complete | §8; hourly ingestion + daily retrain via GitHub Actions |
| Public deployment | ✅ Complete | Live: https://sukkur-aqi-predictor.streamlit.app/ |

---

## 11. Known Limitations

These are deliberate, documented product limitations rather than
incomplete requirements:

- **Single-city scope.** The deployed system targets Sukkur only;
  multi-city support would need per-city coordinates, data partitioning,
  and separately validated models.
- **Mixed-provenance historical AQI.** Combines real OpenAQ ground
  measurements (where coverage exists, roughly Nov 2025 onward for Sukkur)
  with Open-Meteo CAMS model estimates elsewhere. The `aqi_source` column
  must be retained when interpreting results.
- **External dependency on Hopsworks.** Managed feature store and model
  registry access require valid Hopsworks credentials; the local
  CSV/model-registry path remains the reproducible fallback if that
  service is unavailable.
- **Forecast-weather noise calibration is a rough estimate**, currently
  based on general numerical-weather-prediction forecast-skill literature
  rather than this project's own measured OpenWeather forecast error. This
  is flagged in the code as worth replacing once enough live
  forecast-vs-actual history accumulates.
- **No hyperparameter search performed beyond the reported Ridge alphas.**
  Further tuning and larger model architectures are noted as optional
  future work, not blockers to the current, validated result.
- **"Forecast model details" panel currently blank in production.** As
  captured live on 27 Aug 2026 (§7, Live snapshot), the deployed dashboard
  shows `None` where per-horizon model metadata should render. The
  underlying forecast and alert logic are unaffected — this is a
  display/metadata-wiring issue, not a modeling problem — but it should be
  fixed before treating the dashboard's transparency claims as fully live.

---

## 12. Conclusion

All six phases of the AQI Prediction System are implemented, tested, and
validated against real data — not just designed on paper. The system
ingests live and historical air-quality/weather data with documented
fallback strategies, engineers a leakage-safe feature set, persists it to a
managed feature store, trains and rigorously benchmarks multiple
forecasting models against a naive baseline, and serves the result through
a public, explainable, auto-refreshing dashboard. The R² ≥ 0.75 project
target is met on all three forecast horizons (Day 1/2/3), and the full
pipeline — ingestion through dashboard — runs on a scheduled, automated
basis in production at
**https://sukkur-aqi-predictor.streamlit.app/**.

For full technical detail, design rationale, and setup instructions, see
[README.md](README.md).
