# Pearls AQI Predictor — AQI Prediction System

Production-grade data pipeline for the **AQI Prediction System**, built for
the **10Pearls Shine Data Sciences Internship**.

Phase 1 implements the ingestion pipeline: automatically fetching weather
and air-quality data from external APIs, validating it, and persisting raw
data, processed data, and per-run metadata to disk. Phase 2 adds a
feature-engineering step on top of that data. Phase 3 pushes the engineered
features into a **Hopsworks Feature Store**. Phase 4 is exploratory data
analysis (EDA) on the engineered dataset. Phase 5 trains and compares
multiple forecasting models (Ridge Regression, Random Forest, and a neural
network) and saves the best one to a model registry. A serving/dashboard
layer is not included yet — that belongs to a later phase.

**Status:**
- **Phase 1 (ingestion)** complete and validated — 100+ automated tests, see
  [Section 8](#8-testing--validated-scenarios).
- **Phase 2 (feature engineering)** complete and validated — see
  [Section 12](#12-phase-2--feature-engineering).
- **Phase 3 (Hopsworks Feature Store)** complete and validated, now on
  **v2** — see [Section 13](#13-phase-3--hopsworks-feature-store).
- **Phase 4 (EDA)** complete — see [Section 14](#14-phase-4--exploratory-data-analysis-eda).
- **Phase 5 (model training)** complete and validated end-to-end, including
  a naive-baseline sanity check — see
  [Section 15](#15-phase-5--model-training-pipeline).
- **Phase 6 (dashboard + explainability)** complete — a Streamlit dashboard
  serving real-time next-3-day AQI forecasts from the v6 daily-average models
  (R² ≈ 0.79 / 0.78 / 0.79) with SHAP/LIME explanations and hazardous-AQI
  alerts — see [Section 16](#16-phase-6--forecast-dashboard--explainability).
- **Historical backfill** extended to a ~2.5-year range (2024-01-01 through
  present, **22,608 records**), exceeding the original 1.5-2 year target —
  see [Section 11](#11-historical-data-providers).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Folder Structure](#3-folder-structure)
4. [Installation](#4-installation)
5. [API Keys & Configuration](#5-api-keys--configuration)
6. [How to Run](#6-how-to-run)
7. [Sample Output](#7-sample-output)
8. [Testing & Validated Scenarios](#8-testing--validated-scenarios)
9. [Run Metadata](#9-run-metadata)
10. [Design Notes](#10-design-notes)
11. [Historical Data Providers](#11-historical-data-providers)
12. [Phase 2 — Feature Engineering](#12-phase-2--feature-engineering)
13. [Phase 3 — Hopsworks Feature Store](#13-phase-3--hopsworks-feature-store)
14. [Phase 4 — Exploratory Data Analysis (EDA)](#14-phase-4--exploratory-data-analysis-eda)
15. [Phase 5 — Model Training Pipeline](#15-phase-5--model-training-pipeline)
16. [Phase 6 — Forecast Dashboard & Explainability](#16-phase-6--forecast-dashboard--explainability)
17. [Out of Scope (Later Phases)](#17-out-of-scope-later-phases)

---

## 1. Project Overview

The pipeline, on every run:

1. Loads configuration from environment variables (`.env`), validating every
   required value up front.
2. Fetches current weather conditions from the **OpenWeather API**.
3. Fetches air-quality data using a **three-stage fallback strategy**:
   1. **AQICN**, queried by city name.
   2. **AQICN**, queried by latitude/longitude, if the city-name query fails
      or returns an invalid response (e.g. an unrecognized "Unknown station").
   3. **OpenWeather Air Pollution API**, used only if both AQICN attempts fail.
4. Validates the merged record (types, ranges, required fields, timestamps),
   tolerating pollutant sub-fields (PM10/CO/SO2/NO2/O3) being `null` when a
   monitoring station doesn't measure them.
5. Saves the raw API responses as JSON under `data/raw/`.
6. Appends the processed, flattened record as a row in a CSV dataset under
   `data/processed/`.
7. Writes a per-run metadata record (status, sources used, error if any)
   under `data/metadata/`.
8. Logs every stage to both the console and `logs/pipeline.log`.

A separate **historical backfill pipeline** (`historical_backfill.py`) reuses
the exact same fetch/merge/validate logic to collect data for a **past date
range** instead of "now" — see [Section 6](#6-how-to-run) and
[Section 11](#11-historical-data-providers).

Every external call implements **retry logic with exponential backoff**, a
**request timeout**, and **graceful fallback**, so a transient network blip or
a single provider outage does not fail the whole pipeline run.

---

## 2. Architecture

```
                            ┌───────────────────────┐
                            │   .env  (config)       │
                            └───────────┬───────────┘
                                        │
                                        ▼
                            ┌───────────────────────┐
                            │   load_settings()     │  configs/config.py
                            │  (validates env vars) │
                            └───────────┬───────────┘
                                        │
               ┌────────────────────────┼────────────────────────┐
               │                        │                         │
               ▼                        ▼                         ▼
   ┌───────────────────────┐   ┌───────────────────┐   ┌──────────────────────┐
   │  Live WeatherFetcher  │   │ Hist. Weather     │   │ AirQualityFetcher    │
   │  (OpenWeather)        │   │ Fetcher           │   │ Live: 3-stage        │
   │  retry + backoff +    │   │ (Open-Meteo       │   │ fallback (AQICN city │
   │  timeout              │   │  Archive)         │   │ -> geo -> OpenWeather│
   └───────────┬───────────┘   └────────┬──────────┘   │ AirPollution)        │
               │                        │               │ Historical: OpenAQ   │
               │                        │               │ v3 -> Open-Meteo Air │
               │                        │               │ Quality (fallback)   │
               │                        │               └──────────┬─────────┘
               └────────────────────────┼──────────────────────────┘
                                        ▼
                          ┌───────────────────────────┐
                          │  merge_weather_and_aqi()   │  feature_pipeline.py
                          └───────────┬───────────────┘
                                      ▼
                          ┌───────────────────────────┐
                          │      validate_record()     │  data_validator.py
                          │  (types, ranges, nulls)    │
                          └───────────┬───────────────┘
                           success    │    failure
                       ┌───────────────┴───────────────┐
                       ▼                                ▼
        ┌─────────────────────────────┐    ┌───────────────────────────┐
        │  data/raw/*.json             │    │  data/metadata/           │
        │  data/processed/*.csv        │    │  run_<ts>.json /          │
        │  data/metadata/run_<ts>.json │    │  metadata_historical_...  │
        │                              │    │  (status=FAILED + error)  │
        └─────────────────────────────┘    └───────────────────────────┘
                       │                                │
                       └───────────────┬────────────────┘
                                       ▼
                          console + logs/pipeline.log
                                       │
                                       ▼
             Phase 2: feature_engineering.py → feature_df.csv
                                       │
                                       ▼
             Phase 3: feature_store.py → Hopsworks Feature Store
                                       │
                       ┌───────────────┴───────────────┐
                       ▼                                ▼
        Phase 4: EDA.ipynb (analysis)      Phase 5: train_pipeline.py
                                            → model_registry (local + Hopsworks)
```

Live weather flows through the **OpenWeather** fetcher; historical weather
flows through the **Open-Meteo Archive** fetcher; historical AQI flows
through **OpenAQ v3** first, falling back to **Open-Meteo's Air Quality API**
when OpenAQ has no ground-sensor coverage for that time/location (see
[Section 11](#11-historical-data-providers)). All paths converge on the same
`merge_weather_and_aqi()` and `validate_record()` pipeline, so downstream
processing is identical regardless of data source.

A metadata record is written **regardless of outcome** — success or failure —
so every ingestion run leaves an auditable trail.

> **Note on phases:** Phases 2-5 (feature engineering, Hopsworks feature
> store, EDA, model training) all sit *downstream* of Phase 1's
> `data/processed/` output and are documented in their own sections below.
> Phase 6 (dashboard + explainability) is documented in
> [Section 16](#16-phase-6--forecast-dashboard--explainability).

---

## 3. Folder Structure

> **Note:** `data/raw/`, `data/processed/`, `data/metadata/`, and
> `data/model_registry/` are git-ignored by design (see `.gitignore`) —
> these directories contain generated, runtime output that can be
> reproduced by running the pipeline scripts documented in
> [Section 6](#6-how-to-run). This keeps the repository lightweight and
> avoids committing large or stale datasets/models. Folder structure is
> preserved via `.gitkeep` placeholders, and real sample output is shown in
> [Section 7](#7-sample-output) so the pipeline's behavior is visible
> without needing the data files themselves in the repo.

```
pearls-aqi-predictor/
├── configs/
│   └── config.py                  # Centralized, validated configuration loader
├── notebooks/
│   ├── EDA.ipynb                              # Phase 4: exploratory data analysis
│   └── phase3_hopsworks_feature_store.ipynb   # Colab notebook: Phase 3 walkthrough
├── data/
│   ├── raw/                       # Raw JSON API responses (one file per run, git-ignored)
│   ├── processed/                 # Processed CSV datasets (appended each run, git-ignored)
│   ├── metadata/                  # Per-run metadata JSON (status, sources, errors, git-ignored)
│   └── model_registry/            # Phase 5: saved models + scalers + metadata (git-ignored)
├── logs/
│   └── pipeline.log               # Rotating log file (console output is mirrored here)
├── src/
│   ├── feature_pipeline/
│   │   ├── fetch_weather.py       # OpenWeather (live) + Open-Meteo Archive (historical)
│   │   ├── fetch_air_quality.py   # Live: 3-stage AQICN/OpenWeather fallback.
│   │   │                          # Historical: OpenAQ v3 -> Open-Meteo Air Quality fallback
│   │   ├── feature_pipeline.py    # Shared merge_weather_and_aqi() + validate_record() logic
│   │   ├── feature_engineering.py # Phase 2: builds feature_df.csv (time/lag/rolling features, target)
│   │   ├── historical_backfill.py # Orchestrates backfill over a date range
│   │   ├── feature_store.py       # Phase 3: Hopsworks login/create/insert/read
│   │   ├── live_pipeline.py       # Orchestrates a single current-time run
│   │   └── run_pipeline.py        # CLI entry point for a single live run + metadata
│   ├── training_pipeline/
│   │   ├── data_loader.py         # Phase 5: load features from Hopsworks or a local CSV
│   │   ├── preprocessing.py       # Feature selection + time-based (leakage-safe) train/test split
│   │   ├── models.py              # Ridge / Random Forest / Neural Net (TensorFlow, sklearn fallback)
│   │   ├── evaluate.py            # RMSE/MAE/R² + naive persistence baseline
│   │   ├── model_registry.py      # Local model registry + optional Hopsworks Model Registry push
│   │   └── train_pipeline.py      # CLI orchestrator: load -> split -> train -> evaluate -> save
│   ├── dashboard/
│   │   ├── forecast.py            # Phase 6: daily features (incl. future weather) + Day1/2/3 forecast generation
│   │   ├── explainability.py      # Phase 6: SHAP + LIME per-forecast feature contributions
│   │   └── app.py                 # Phase 6: Streamlit dashboard (forecast, alerts, explanations)
│   ├── utils/
│   │   ├── storage.py             # JSON/CSV read & write helpers
│   │   └── data_validator.py      # Validation rules for merged records
│   └── common/
│       ├── logger.py              # Centralized console + file logging
│       └── exceptions.py          # APIError, ConfigurationError, ValidationError, StorageError,
│                                   # PipelineError, FeatureStoreError, TrainingError
├── tests/
│   ├── conftest.py                     # Shared fast-failing Settings fixture
│   ├── test_config.py                  # Missing/invalid env var -> ConfigurationError
│   ├── test_fetch_weather.py           # Success, retry, timeout, non-retryable, malformed payload
│   ├── test_fetch_air_quality.py       # Live 3-stage fallback + exponential backoff + historical
│   │                                   # OpenAQ v3 -> Open-Meteo Air Quality fallback
│   ├── test_data_validator.py          # Required/optional fields, ranges, timestamps
│   ├── test_storage.py                 # JSON/CSV round-trip, directory creation, errors
│   ├── test_feature_pipeline.py        # Merge + validation logic
│   ├── test_historical_backfill.py     # Dedup, incremental writes, metadata, --force
│   ├── test_live_pipeline.py           # Live pipeline end-to-end
│   ├── test_run_pipeline.py            # Metadata on success/failure, e2e run
│   ├── test_feature_store.py           # Phase 3: login/create/insert/read, hopsworks + hsfs fallback
│   ├── test_training_data_loader.py    # Phase 5: Hopsworks / local CSV loading
│   ├── test_training_preprocessing.py  # Phase 5: feature selection, time-based split, build_xy
│   ├── test_training_models.py         # Phase 5: model candidates, TensorFlow/sklearn fallback
│   ├── test_training_evaluate.py       # Phase 5: RMSE/MAE/R², naive baseline, comparison table
│   ├── test_training_model_registry.py # Phase 5: local save/load round-trip, Hopsworks best-effort
│   ├── test_train_pipeline.py          # Phase 5: full end-to-end training run
│   ├── test_dashboard_forecast.py      # Phase 6: daily features, model loading, forecast generation
│   └── test_dashboard_explainability.py # Phase 6: SHAP/LIME explanations
├── requirements.txt
├── .env.example
├── pytest.ini
└── README.md
```

---

## 4. Installation

**Prerequisites:** Python 3.11+

```bash
# 1. Clone / unzip the project, then move into it
cd pearls-aqi-predictor

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

`requirements.txt` includes `scikit-learn` and `joblib` (required for
Phase 5) and `matplotlib`/`seaborn` (required for Phase 4's EDA notebook).
`tensorflow` is listed as an optional extra for Phase 5's neural-network
candidate — if it isn't installed (or can't be, e.g. limited disk space),
the training pipeline automatically falls back to scikit-learn's
`MLPRegressor` instead; see [Section 15](#15-phase-5--model-training-pipeline).

---

## 5. API Keys & Configuration

1. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

2. Fill in your credentials in `.env`:

| Variable                  | Description                                                  |
|---------------------------|----------------------------------------------------------------|
| `AQICN_TOKEN`              | Free token from https://aqicn.org/data-platform/token/         |
| `OPENWEATHER_API_KEY`      | Free API key from https://openweathermap.org/api. **Note:** a newly created key can take 1-2 hours to activate; a `401` right after signup is expected during that window. |
| `OPENAQ_API_KEY`           | Free API key from https://explore.openaq.org/register (required for historical AQI backfill's primary OpenAQ v3 stage) |
| `DEFAULT_CITY`              | City name used for the AQICN city-name stage (e.g. `Sukkur`)     |
| `OPENWEATHER_LAT`           | Latitude, used for OpenWeather, the AQICN geo-fallback stage, and both historical AQI providers |
| `OPENWEATHER_LON`           | Longitude, used the same way as `OPENWEATHER_LAT`                |
| `LOG_LEVEL`                 | `DEBUG` / `INFO` / `WARNING` / `ERROR` (default: `INFO`)          |
| `REQUEST_TIMEOUT_SECONDS`   | Per-request timeout in seconds (default: `10`)                   |
| `MAX_RETRIES`               | Retry attempts per API call (default: `3`)                       |
| `RETRY_BACKOFF_SECONDS`     | Base backoff between retries; grows **exponentially** as `base * 2^(attempt-1)` (default base: `2`, i.e. 2s, 4s, 8s...) |
| `HOPSWORKS_API_KEY`         | Hopsworks API key (Account Settings → API Keys on https://app.hopsworks.ai). Required for Phase 3 (`feature_store.py`) and optionally Phase 5's Hopsworks Model Registry push. |
| `HOPSWORKS_PROJECT_NAME`    | Name of your Hopsworks project. Required the same places as `HOPSWORKS_API_KEY`. |

   `.env` is git-ignored — never commit real credentials. If any required
   variable is missing, empty, or an invalid lat/lon, the pipeline raises a
   `ConfigurationError` immediately, before any network call is made.

   **No key required** for the Open-Meteo Archive (historical weather) or
   Open-Meteo Air Quality (historical AQI fallback) APIs — both are free,
   unauthenticated endpoints.

---

## 6. How to Run

### Live pipeline (current conditions, one run)

```bash
python -m src.feature_pipeline.run_pipeline
```

On success you'll see a summary printed to the console, a new raw JSON file
in `data/raw/`, a new/updated row in `data/processed/aqi_dataset.csv`, a new
metadata file in `data/metadata/`, and a full trace in `logs/pipeline.log`.

**Scheduling:** to run automatically (e.g. hourly), add a cron entry:

```bash
0 * * * * cd /path/to/pearls-aqi-predictor && venv/bin/python -m src.feature_pipeline.run_pipeline
```

(On Windows, use Task Scheduler with the same command, or a GitHub Actions
scheduled workflow if the machine can't stay on continuously.)

### Historical backfill (past date range)

```bash
python -m src.feature_pipeline.historical_backfill <start_date> <end_date>

# Example:
python -m src.feature_pipeline.historical_backfill 2024-01-01 2026-07-30
```

This collects data at the pipeline's configured interval (`HISTORICAL_INTERVAL_HOURS`
in `historical_backfill.py`; currently hourly) across the given range, using
Open-Meteo Archive for weather and, for AQI, OpenAQ v3 falling back to
Open-Meteo Air Quality (see [Section 11](#11-historical-data-providers)).
Results append to `data/processed/aqi_dataset_historical.csv`.

The backfill is **idempotent and resumable**: re-running the same range
skips timestamps already saved via an **upfront** `(collection_timestamp,
city)` check performed *before* any API call is made — so re-running an
interrupted multi-day backfill re-verifies already-collected steps almost
instantly (no redundant network calls) and only fetches genuinely new
timestamps. For very long ranges (e.g. 2+ years), splitting into smaller
chunks (e.g. quarterly) and running them sequentially makes interruptions
cheap to recover from — this is exactly how the current ~2.5-year, 22,608-row
Sukkur dataset was collected:

```bash
python -m src.feature_pipeline.historical_backfill 2024-01-01 2024-03-31
python -m src.feature_pipeline.historical_backfill 2024-04-01 2024-06-30
# ...and so on
```

To force re-collection of a range that's already been backfilled (e.g. to
refresh data), add `--force`:

```bash
python -m src.feature_pipeline.historical_backfill 2025-11-21 2026-07-26 --force
```

> **Note:** `--force` appends fresh rows rather than replacing existing ones,
> since storage is append-only. Only use it if you're prepared to
> deduplicate afterward.

### Feature engineering (Phase 2)

```bash
python -m src.feature_pipeline.feature_engineering
```

See [Section 12](#12-phase-2--feature-engineering) for details.

### Hopsworks Feature Store sync (Phase 3)

```bash
python -m src.feature_pipeline.feature_store
```

See [Section 13](#13-phase-3--hopsworks-feature-store) for details, setup,
and a Windows-specific installation note.

### Exploratory data analysis (Phase 4)

Open `notebooks/EDA.ipynb` in Jupyter, VS Code, or Colab and run all cells.
See [Section 14](#14-phase-4--exploratory-data-analysis-eda).

### Model training (Phase 5)

```bash
python -m src.training_pipeline.train_pipeline --source local
# or, from the Hopsworks Feature Store directly:
python -m src.training_pipeline.train_pipeline --source hopsworks
```

See [Section 15](#15-phase-5--model-training-pipeline) for details, options,
and how results are evaluated.

### Forecast dashboard (Phase 6)

```bash
streamlit run src/dashboard/app.py
```

See [Section 16](#16-phase-6--forecast-dashboard--explainability) for details.

---

## 7. Sample Output

### Sample console/log output (real live run, Sukkur)

```
2026-07-26 21:59:12 | INFO  | __main__ | Pipeline started
2026-07-26 21:59:12 | INFO  | __main__ | Configuration loaded: city=Sukkur lat=27.7052 lon=68.8574
2026-07-26 21:59:14 | INFO  | fetch_weather | Response status ... 200
2026-07-26 21:59:14 | INFO  | __main__ | Weather data fetched successfully from OpenWeather
2026-07-26 21:59:15 | WARNING | fetch_air_quality | stage=AQICN-city event=stage_failed next_stage=AQICN-geo reason=Unknown station
2026-07-26 21:59:16 | INFO  | fetch_air_quality | stage=AQICN-geo event=response_received status_code=200
2026-07-26 21:59:16 | INFO  | __main__ | AQI data fetched successfully from AQICN-geo
2026-07-26 21:59:16 | INFO  | data_validator | Record validation passed for city=Sukkur
2026-07-26 21:59:16 | INFO  | storage | Saved JSON file: data/raw/raw_sukkur_20260726T165916Z.json
2026-07-26 21:59:16 | INFO  | storage | Saved CSV file (1 rows): data/processed/aqi_dataset.csv
2026-07-26 21:59:16 | INFO  | storage | Saved run metadata: data/metadata/run_20260726T165916Z.json
2026-07-26 21:59:16 | INFO  | __main__ | Pipeline completed successfully
```

This trace shows the fallback chain working exactly as designed in a real
run: the AQICN city-name query failed ("Unknown station" — Sukkur is not a
named AQICN station), so the pipeline automatically retried via AQICN's geo
lookup, which succeeded with real station data.

### Sample raw JSON output (`data/raw/raw_sukkur_<timestamp>.json`)

```json
{
  "weather_raw": { "...": "full OpenWeather response, standardized" },
  "aqi_raw": { "...": "full AQICN response, standardized" },
  "merged": {
    "timestamp": "2026-07-26T16:59:14.379417+00:00",
    "collection_timestamp": "2026-07-26T16:59:14.379417+00:00",
    "city": "Sukkur",
    "latitude": 27.7052,
    "longitude": 68.8574,
    "temperature": 37.33,
    "feels_like": 41.51,
    "humidity": 39,
    "pressure": 998,
    "wind_speed": 5.3,
    "wind_direction": 235,
    "clouds": 44,
    "sunrise": 1785026642,
    "sunset": 1785075472,
    "visibility": 10000,
    "weather": "scattered clouds",
    "aqi": 119,
    "pm2_5": 119,
    "pm10": null,
    "co": null,
    "so2": null,
    "no2": null,
    "o3": null,
    "weather_source": "OpenWeather",
    "aqi_source": "AQICN-geo"
  }
}
```

Note `pm10`/`co`/`so2`/`no2`/`o3` are `null` here — the nearest AQICN station
to Sukkur only has AQI + PM2.5 sensors. This is expected and handled: see
[Section 10](#10-design-notes).

### Sample processed CSV output (`data/processed/aqi_dataset.csv`)

| timestamp | city | latitude | longitude | temperature | humidity | aqi | pm2_5 | pm10 | weather_source | aqi_source |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-26T16:59:14+00:00 | Sukkur | 27.7052 | 68.8574 | 37.33 | 39 | 119 | 119 | | OpenWeather | AQICN-geo |

*(table truncated for readability — the real CSV has all 22 columns)*

---

## 8. Testing & Validated Scenarios

```bash
pytest            # run all tests
pytest -v         # verbose, one line per test
```

All external HTTP calls are mocked — no real API keys or network access are
needed to run the suite. The following scenarios are explicitly covered:

| Test Case                          | Expected Result                          | Covered by |
|-------------------------------------|-------------------------------------------|------------|
| City-name AQICN lookup succeeds      | `source == "AQICN-city"`, no fallback      | `test_stage1_aqicn_city_succeeds_without_further_stages` |
| Invalid/unrecognized city ("Unknown station") | Falls back to AQICN geo lookup | `test_stage2_aqicn_geo_used_when_city_query_fails` |
| AQICN fully down (both stages)       | Falls back to OpenWeather Air Pollution    | `test_stage3_openweather_used_when_both_aqicn_stages_fail` |
| Historical: OpenAQ v3 has no data for this time/location | Falls back to Open-Meteo Air Quality | `test_fetch_air_quality.py` |
| Both AQI sources down                | Pipeline raises `APIError`, exits gracefully | `test_raises_when_all_three_stages_fail`, `test_run_pipeline_failure_still_writes_failed_metadata` |
| Invalid/expired API key (HTTP 401)   | Non-retryable `APIError` raised immediately | `test_fetch_current_weather_raises_api_error_on_non_retryable_status` |
| Request timeout / connection drop    | Retries with exponential backoff, then `APIError` | `test_fetch_current_weather_raises_api_error_on_timeout`, `test_exponential_backoff_used_between_retries` |
| Malformed/unexpected API response    | `APIError` raised rather than silently corrupting data | `test_fetch_current_weather_raises_api_error_on_malformed_payload` |
| Missing required `.env` variable     | `ConfigurationError` raised before any network call | `test_load_settings_raises_configuration_error_when_required_var_missing` (parametrized over all required vars) |
| Partial pollutant data (station lacks some sensors) | Validation passes, nulls preserved | `test_validate_record_accepts_partial_pollutant_data` |
| Historical backfill dedup            | Re-running the same range skips already-saved timestamps, before any API call | `test_historical_backfill.py` |
| Historical backfill `--force`        | Bypasses dedup, re-fetches requested range | `test_historical_backfill.py` |
| Pipeline succeeds/fails end-to-end   | Metadata `status=SUCCESS`/`FAILED` written correctly | `test_run_pipeline_success_writes_success_metadata`, `test_run_pipeline_failure_still_writes_failed_metadata` |
| Hopsworks login (full SDK or `hsfs` fallback) | Returns a feature store handle either way | `test_login_success_returns_feature_store_handle`, `test_login_falls_back_to_hsfs_when_hopsworks_not_installed` |
| Feature group create/insert/read (v2) | Correct schema, round-trip data integrity | `test_get_or_create_feature_group_uses_expected_schema`, `test_sync_feature_store_end_to_end` |
| Training: feature selection excludes all-null columns | `pm10_*`, `pollution_index` (single-pollutant cities) excluded automatically | `test_training_preprocessing.py` |
| Training: time-based split has no leakage | Train set strictly precedes test set chronologically | `test_time_based_split_produces_expected_sizes_and_order` |
| Training: TensorFlow unavailable       | Falls back to scikit-learn's `MLPRegressor` transparently | `test_get_model_candidates_uses_sklearn_fallback_when_tensorflow_unavailable` |
| Training: naive baseline computed correctly | Matches direct RMSE/MAE/R² computation | `test_compute_naive_baseline_metrics_matches_compute_metrics` |
| Training: Hopsworks Model Registry push fails | Returns `False`, does not raise; local registry stays authoritative | `test_register_model_to_hopsworks_returns_false_on_failure` |
| Training: full pipeline end-to-end     | Comparison table includes naive baseline, best model saved locally | `test_train_pipeline.py` |

**Manually verified against live APIs** (not mocked) for **Sukkur**
(`27.7052, 68.8574`): a successful live end-to-end run producing real weather
+ AQI data (Section 7); and a full historical backfill spanning
**2024-01-01 through present (22,608 records)** combining real OpenAQ
ground-sensor measurements (~Nov 2025 onward) with Open-Meteo Air Quality
model estimates for the earlier period (see
[Section 11](#11-historical-data-providers)). **Phase 3 was manually
verified against a real Hopsworks project**, including the v2 re-sync after
the `feels_like` and rolling-window/horizon fixes — see
[Section 13](#13-phase-3--hopsworks-feature-store). **Phase 5 was manually
run end-to-end** against the real Sukkur feature set, producing a full
Ridge/Random Forest/Neural-Network comparison and a locally-saved best model
— see [Section 15](#15-phase-5--model-training-pipeline).

---

## 9. Run Metadata

Every pipeline run — whether it succeeds or fails — writes a small JSON
record to `data/metadata/run_<timestamp>.json` (live) or
`data/metadata/metadata_historical_<timestamp>.json` (backfill):

```json
{
  "run_time": "2026-07-26T16:59:16.000000+00:00",
  "city": "Sukkur",
  "weather_source": "OpenWeather",
  "aqi_source": "AQICN-geo",
  "records": 1,
  "status": "SUCCESS",
  "error": null
}
```

On failure, `status` is `"FAILED"`, `error` contains the exception message,
and `weather_source`/`aqi_source` report whatever partial progress was made
before the failure (e.g. weather succeeded but AQI fetching did not).
Metadata writes are best-effort and never mask the pipeline's real
success/failure outcome — a metadata write failure is logged as a warning,
not raised.

This gives a lightweight, greppable audit trail of every ingestion run
without needing to parse the full `logs/pipeline.log` file.

---

## 10. Design Notes

- **Custom exception hierarchy** (`APIError`, `ConfigurationError`,
  `ValidationError`, `StorageError`, `PipelineError`, `FeatureStoreError`,
  `TrainingError`) makes failure modes explicit and lets callers catch
  precisely what they need to handle.
- **Fail fast, fail loud**: validation raises rather than silently coercing or
  dropping bad data, so problems surface at ingestion time rather than deep
  inside a future model.
- **Provider-agnostic schema**: every weather/AQI source (AQICN by city or
  lat/lon, OpenWeather's Air Pollution API, OpenAQ v3, Open-Meteo Air
  Quality) is normalized into the exact same dictionary shape, so downstream
  code never needs to know which provider/stage actually responded. The
  `source` field on the returned record is the only place that distinction
  is visible.
- **Retry + exponential backoff + timeout** are applied uniformly to every
  outbound request via a shared `_request_with_retry` helper in each fetcher.
  Backoff grows as `retry_backoff_seconds * 2 ** (attempt - 1)` (e.g. with a
  2s base: 2s, 4s, 8s, ...).
- **Optional pollutant fields**: not every AQI source measures every
  pollutant — many report only AQI + PM2.5. `pm10`/`co`/`so2`/`no2`/`o3`
  are therefore allowed to be `null` (the key must still be present, so the
  dataset schema stays consistent), while `aqi` and `pm2_5` remain strictly
  required since they're universally available across every source this
  project uses (see [Section 11](#11-historical-data-providers) for why this
  is enforced even for the Open-Meteo Air Quality fallback, which *could*
  technically supply the other pollutants).
- **Metadata is diagnostic, not load-bearing**: `save_run_metadata` swallows
  its own storage errors (logs a warning) so a metadata-write hiccup can
  never mask or override the pipeline's real success/failure result.
- **Historical dedup is per `(collection_timestamp, city)`, checked before
  any API call**: re-running a backfill for the same range is always safe
  and idempotent, *and* cheap to resume after an interruption, since
  already-saved timestamps are recognized and skipped up front rather than
  being re-fetched from both APIs and discarded afterward.

## 11. Historical Data Providers

### Why These Providers?

The original implementation used **OpenWeather One Call API 3.0 Time Machine**
for historical weather and **OpenWeather Air Pollution History** for historical
AQI. However, both endpoints require a **paid subscription** — the free tier
only covers current data.

Additionally, **AQICN does not provide historical AQI**. It always returns the
latest observation regardless of the requested timestamp, which would silently
fabricate history if used for backfill.

After discussion with the project mentor, the historical providers were
replaced with **free API alternatives** that satisfy the project requirement
of collecting historical data through external APIs:

| Historical provider   | API                                          | Cost  |
|-----------------------|-----------------------------------------------|-------|
| **Weather**           | Open-Meteo Archive                             | Free  |
| **AQI (primary)**     | OpenAQ v3                                       | Free  |
| **AQI (fallback)**    | Open-Meteo Air Quality (CAMS model)             | Free  |

The project requirements state: *"Fetch raw weather and pollutant data from
external APIs like AQICN or OpenWeather."* The wording is illustrative, not
mandatory. What matters is that the pipeline collects real, API-sourced data
— which all three providers here satisfy, each in a documented way.

### Open-Meteo Archive (Weather)

- **Endpoint**: `https://archive-api.open-meteo.com/v1/archive`
- **No API key required** — unlimited free access for moderate use.
- Covers reanalysis data from the 1940s to near-real-time.
- Returns hourly temperature, feels-like (`apparent_temperature`), humidity,
  pressure, wind speed/direction, cloud cover, precipitation, visibility (not
  provided by this dataset — see below), and WMO weather codes.

Weather descriptions are mapped from WMO weather interpretation codes using
a static lookup table (see `_map_wmo_weather_code` in `fetch_weather.py`).
`visibility` is always `null` for historical records: it's a genuine
limitation of the ERA5 reanalysis dataset this endpoint is built on (unlike
some live forecast models, ERA5 has no visibility variable) — not a bug.
`feels_like` **is** available (`apparent_temperature`) and is requested and
mapped explicitly.

### AQI Provider 1 — OpenAQ v3 (ground-sensor measurements)

- **OpenAQ v2 retired** — the legacy `/v2/measurements` endpoint is gone
  (returns HTTP 410 Gone). All historical AQI queries try v3 first.
- **Authentication required** — `OPENAQ_API_KEY` must be set in `.env`.
  Get a free API key at https://explore.openaq.org/register.
- **Hierarchical API** — v3 uses a locations → sensors → measurements model:
  1. `GET /v3/locations?coordinates=lat,lon&radius=25000` finds nearby stations.
  2. Sensor objects (embedded in the location response) identify which
     pollutants each station measures.
  3. `GET /v3/sensors/{id}/measurements?datetime_from=...&datetime_to=...`
     returns raw measurement values per parameter, searched over a widened
     time window (±6h) and across up to 5 candidate sensors per pollutant to
     tolerate sparsely-reporting stations.
- Since OpenAQ returns **raw pollutant concentrations** rather than a ready-made
  AQI value, the AQI is computed from concentrations using the official
  **US EPA AQI breakpoint formula** (see below). This ensures historical AQI
  values are derived from real ground measurements rather than fabricated data.
- **Coverage limitation**: OpenAQ's ground-sensor network only covers
  wherever a physical station happens to be reporting. For **Sukkur**
  specifically, the nearest sensor was confirmed active but returned no
  measurements before **~November 21, 2025** — the station simply wasn't
  reporting data yet, diagnosed by binary-searching backward from a
  known-working date to locate the actual activation boundary.

### AQI Provider 2 — Open-Meteo Air Quality (model-based fallback)

Since OpenAQ's ground-sensor coverage for Sukkur only extends back to late
2025 — well short of the project's 1.5-2 year target — **Open-Meteo's Air
Quality API** is used as an automatic fallback whenever the OpenAQ v3 stage
raises an `APIError` (no location/sensor/measurement found for that
time/location):

- **Endpoint**: `https://air-quality-api.open-meteo.com/v1/air-quality`
- **No API key required.**
- Backed by **CAMS** (Copernicus Atmosphere Monitoring Service) global
  atmospheric-composition model output (~40 km resolution) — **this is a
  model estimate, not a ground measurement.** Verified manually for Sukkur's
  coordinates: plausible, physically-consistent seasonal values at least as
  far back as **July 2023** (e.g. higher PM2.5/AQI in winter, lower in
  summer, matching the pattern seen in the real OpenAQ-covered months).
- Open-Meteo computes the **US EPA AQI** (`us_aqi`) directly from its
  modeled concentrations, so it's used as-is here rather than recomputed.
- **Deliberately narrowed schema**: although Open-Meteo's response includes
  `pm10`, `co`, `so2`, `no2`, and `o3` alongside `pm2_5`/`us_aqi`, this
  fallback's parsing code (`_fetch_openmeteo_air_quality` in
  `fetch_air_quality.py`) **discards those five fields, setting them to
  `null`** rather than passing them through. Only `aqi` and `pm2_5` are
  populated. This is intentional: OpenAQ-covered records (the
  ground-truth era) already have `pm10`/`co`/`so2`/`no2`/`o3` as `null` for
  Sukkur (single-pollutant station — see [Section 10](#10-design-notes)).
  If the Open-Meteo fallback populated those columns only for the *older*
  era, their mere presence/absence would become a spurious proxy for
  *which calendar period a row belongs to* rather than any real physical
  signal — a model could learn "pm10 is present" as a stand-in for "this
  is old data" instead of learning genuine pollution dynamics. Keeping the
  schema uniform (only `aqi`/`pm2_5` reliable, everything else `null`)
  avoids that confound across the entire historical range.
- `source` is set to `"OpenMeteo-AirQuality"` on every record from this
  stage, so which provider actually served each row remains fully
  traceable in the processed CSV and metadata, exactly like the live
  pipeline's AQICN-city/AQICN-geo/OpenWeather distinction.

#### AQI Calculation (OpenAQ path)

The EPA breakpoint formula is:

```
AQI = ((I_high - I_low) / (C_high - C_low)) * (C - C_low) + I_low
```

Where *C* is the pollutant concentration and the breakpoint ranges
*(C_low, C_high)* and corresponding AQI ranges *(I_low, I_high)* are defined
in the EPA Technical Assistance Document. The overall AQI is the **maximum**
of all pollutant sub-indices. Unit conversions are applied automatically
when OpenAQ returns concentrations in µg/m³ rather than ppm/ppb (standard
conversion factors at 25°C, 1013 hPa). The Open-Meteo fallback path skips
this calculation entirely, using Open-Meteo's own `us_aqi` output instead.

### Extended historical range — validated

With the Open-Meteo Air Quality fallback in place, a backfill covering
**2024-01-01 through present (~2.5 years)** was run to completion, collected
in quarterly chunks to make interruptions cheap to resume from (see
[Section 6](#6-how-to-run)):

```
ALL CHUNKS COMPLETE — 22,608 total records
```

This exceeds the original 1.5-2 year project target. `aqi_source` in the
resulting CSV distinguishes real ground-sensor rows (`OpenAQ-v3`, roughly
Nov 2025 onward) from CAMS model-estimate rows (`OpenMeteo-AirQuality`,
everything earlier) — see [Section 12](#12-phase-2--feature-engineering) for
how this dataset feeds into feature engineering. Cities with denser,
longer-established OpenAQ monitoring networks (e.g. Karachi, Lahore,
Islamabad) may get real ground-sensor coverage across a much larger
fraction of a similar range; this is worth re-checking per city rather than
assumed.

### Current Architecture

```
Live Pipeline

OpenWeather (current weather)
        +
AQICN (current AQI, 3-stage fallback)
        |
        v
Feature Pipeline (merge + validate)
        |
        v
data/processed/aqi_dataset.csv

Historical Pipeline

Open-Meteo Archive (historical weather)
        +
OpenAQ v3 (historical AQI, ground sensor)
   -> falls back to ->
Open-Meteo Air Quality (historical AQI, CAMS model)
        |
        v
Feature Pipeline (merge + validate)
        |
        v
data/processed/aqi_dataset_historical.csv
```

The live and historical pipelines converge on the **same feature pipeline**
(`process_features`), so all downstream processing — validation, storage,
metadata — is identical regardless of data source. Phase 2 builds directly
on top of this output (see [Section 12](#12-phase-2--feature-engineering)).

### What stays the same

| Component                | Provider              | Status |
|--------------------------|-----------------------|--------|
| **Live weather**         | OpenWeather           | Unchanged |
| **Live AQI**             | AQICN (3-stage fallback) | Unchanged |
| **Historical weather**   | Open-Meteo Archive    | Unchanged |
| **Historical AQI**       | OpenAQ v3 → Open-Meteo Air Quality | Extended with a fallback stage |
| Feature pipeline         | —                     | Unchanged |
| Validation & storage     | —                     | Unchanged |
| Metadata & logging       | —                     | Unchanged |

### Configuration

The historical backfill requires:

| Variable         | Description                                           |
|------------------|-------------------------------------------------------|
| `OPENAQ_API_KEY` | Free API key from https://explore.openaq.org/register |

Plus the existing `OPENWEATHER_LAT` and `OPENWEATHER_LON` coordinates (already
required for the live pipeline, and used by both historical AQI providers).
If `OPENAQ_API_KEY` is missing or empty, `load_settings()` raises a
`ConfigurationError` immediately. No key is needed for the Open-Meteo
Air Quality fallback.

### Logging

Historical records are logged with one of:

```
weather_source = "Open-Meteo Archive"
aqi_source = "OpenAQ-v3"                  stage=OpenAQ-v3
aqi_source = "OpenMeteo-AirQuality"       stage=OpenMeteo-AirQuality (fallback, after OpenAQ-v3 failure)
```

Live records continue to use:

```
weather_source = "OpenWeather"
aqi_source = "AQICN-city" | "AQICN-geo" | "OpenWeather-AirPollution"
```

This distinction is visible in both the console log output and the
`weather_source` / `aqi_source` columns in the processed CSV dataset.

---

## 12. Phase 2 — Feature Engineering

`src/feature_pipeline/feature_engineering.py` builds a training-ready
dataset from the Phase 1 processed CSV. Run with:

```bash
python -m src.feature_pipeline.feature_engineering
```

This reads `data/processed/aqi_dataset_historical.csv`, and writes
`data/processed/feature_df.csv` (also git-ignored, like the rest of `data/`
— see [Section 3](#3-folder-structure)).

**Features generated (all computed per-city, on a deduplicated,
chronologically-sorted series, so nothing leaks across city boundaries or
out-of-order backfill runs):**

- **Time features** — `hour`, `day_of_week`, `day_of_month`, `month`, plus
  cyclical `sin`/`cos` encodings of each, so e.g. hour 23 and hour 0 are
  close in feature space rather than maximally far apart.
- **Lag features** — the previous 1-2 readings for AQI and key
  pollutants/weather fields (`aqi`, `pm2_5`, `pm10`, `temperature`,
  `humidity`).
- **Rolling averages** — 24h and 48h rolling mean/std for `aqi`, `pm2_5`,
  and `pm10`, computed as `24 // HISTORICAL_INTERVAL_HOURS` and
  `48 // HISTORICAL_INTERVAL_HOURS` steps respectively (imported directly
  from `historical_backfill.py`'s interval constant). This is deliberately
  **not hardcoded** to a specific step count: an earlier version assumed a
  fixed 3-hour collection interval (8/16 steps for 24h/48h), which silently
  became wrong once the pipeline's actual interval changed to hourly — the
  windows were still *labeled* 24h/48h but only covered 8h/16h of real
  time. Deriving the step counts from the actual configured interval at
  import time keeps this correct automatically if the interval ever changes
  again.
- **AQI change rate** — first difference from the previous reading, for
  the same set of columns as the lag features.
- **Pollution index** — a simple composite score (`pollution_index`)
  averaging each available pollutant's ratio to its EPA "Moderate"
  breakpoint threshold. This is distinct from the official EPA `aqi`
  column already in the dataset (which is the *max* of pollutant
  sub-indices). **Requires at least 2 pollutants with real (non-null) data**
  to be computed — for single-pollutant locations like Sukkur (PM2.5 only),
  this is `null` rather than a degenerate, perfectly-correlated copy of
  `pm2_5` alone (an earlier version didn't check for this and produced
  exactly that redundant copy; see the function docstring for the full
  rationale).
- **Target variable** — `aqi_target`: the AQI value `HORIZON_STEPS`
  intervals ahead, defaulting to **72 hours / 3 days** (also computed
  dynamically from `HISTORICAL_INTERVAL_HOURS`, for the same reason as the
  rolling windows above), configurable via `--horizon-steps`.

**Handling missing pollutants:** not every monitoring source measures every
pollutant (e.g. Sukkur's nearest station only reports AQI + PM2.5, so
`pm10`/`co`/`so2`/`no2`/`o3` are `null` in the raw data across the *entire*
historical range, by design — see [Section 11](#11-historical-data-providers)).
Rather than requiring completeness on a derived column that can *never* be
non-null for such a location (which would silently drop the entire
dataset), the script detects and excludes any derived column with no real
data anywhere in the dataset from the row-completeness check, logging a
warning naming exactly which columns were skipped and why.

Rows lacking full lag/rolling/target history (each city's first few rows
and last `horizon_steps` rows) are dropped by default, leaving a dataset
immediately usable for model training. Pass `--keep-incomplete-rows` to
keep them instead (e.g. for custom imputation).

**Validated on the real Sukkur dataset:** across the full ~2.5-year,
22,608-row historical CSV (see [Section 11](#11-historical-data-providers)),
with `feels_like` fully populated and correctly-sized 24h/48h rolling
windows and a 72h forecast horizon confirmed against the actual hourly
collection interval.

---

## 13. Phase 3 — Hopsworks Feature Store

`src/feature_pipeline/feature_store.py` connects the Phase 2 output
(`feature_df.csv`) to a Hopsworks Feature Store:

```
feature_df.csv
      │
      ▼
  login()                     -- hopsworks.login() or hsfs.connection() fallback
      │
      ▼
  get_or_create_feature_group() -- name="aqi_features", v2
      │                           primary_key=["city", "collection_timestamp"]
      │                           event_time="collection_timestamp"
      ▼
  insert_features()           -- writes to the offline (Hudi) store
      │
      ▼
  read_features()             -- confirms the round trip
```

### Feature Group Schema

| Setting | Value |
|---|---|
| Name | `aqi_features` |
| Version | `2` |
| Primary key | `city`, `collection_timestamp` |
| Event time | `collection_timestamp` |
| Time travel format | `HUDI` |
| Online store | Disabled (offline/batch only — sufficient for model training) |

`collection_timestamp` (the pipeline's normalized, idempotency-safe
timestamp — see [Section 10](#10-design-notes)) is used as both the event
time and half the primary key, since it uniquely identifies one observation
per city. The redundant `timestamp` column (always identical to
`collection_timestamp`) is dropped before insert.

**Why v2:** the feature group was bumped from v1 to v2 after two data-quality
fixes were made to the ingestion/feature-engineering code (see
[Section 11](#11-historical-data-providers) and
[Section 12](#12-phase-2--feature-engineering)): (1) `feels_like` was
previously hardcoded to `null` for every historical weather record instead
of being fetched from the Open-Meteo Archive's `apparent_temperature`
variable, and (2) the rolling-window and forecast-horizon sizes were
computed assuming a fixed 3-hour collection interval, which silently became
incorrect once the pipeline's actual interval changed to hourly (windows
were labeled 24h/48h but only covered 8h/16h of real time; the 3-day
forecast target was actually only 1 day ahead). v1 (pre-fix data) is kept in
Hopsworks for history/comparison rather than deleted; v2 is the version all
Phase 5 training should use.

> **Known caveat:** the `pollution_index` fix and the extended (~2.5-year,
> 22,608-row) historical backfill (see [Section 11](#11-historical-data-providers))
> both landed *after* the v2 sync, so the currently-synced v2 copy has
> `pollution_index` populated with its pre-fix (redundant,
> perfectly-correlated-with-`pm2_5`) values, and only covers the earlier,
> smaller date range. This doesn't block training — `select_feature_columns`
> in the training pipeline excludes a column only when it's *entirely* null
> across the loaded dataset, so `pollution_index` will still be included as
> a harmless-but-useless feature if training directly from Hopsworks v2
> today. Re-sync to v3 after regenerating `feature_df.csv` from the full
> extended dataset to get both fixes and the larger date range reflected in
> Hopsworks; training from a freshly-generated local CSV (`--source local`)
> already has the current, correct data.

### Setup

1. Create a free account/project at https://app.hopsworks.ai.
2. Generate an API key: Account Settings → API Keys.
3. Add to `.env`:

   ```
   HOPSWORKS_API_KEY=your_api_key_here
   HOPSWORKS_PROJECT_NAME=your_project_name_here
   ```

4. Install a client library. Two options:
   - `pip install hopsworks` (full SDK)
   - `pip install "hsfs[python]"` (lighter — Feature Store only)

   Both also require `pip install confluent-kafka` (used by `insert()` to
   stream writes to the feature group).

   **Windows note:** both `hopsworks` and `hsfs` pull in a `twofish` C
   extension (used only for on-prem Hive/Kafka storage connectors this
   project doesn't need) that requires Microsoft C++ Build Tools to
   compile, and there is no prebuilt wheel for it on PyPI. If installing
   those Build Tools isn't practical (e.g. limited disk space), run the
   sync from a Linux environment instead — **Google Colab works well**,
   since `twofish` compiles there without issue. See
   `notebooks/phase3_hopsworks_feature_store.ipynb` for a ready-to-run
   notebook version of the steps below. `login()` in `feature_store.py`
   transparently tries the full `hopsworks` package first and falls back
   to the lighter `hsfs` client if only that's installed, so either
   install option works with the same code.

### How to Run

```bash
python -m src.feature_pipeline.feature_store
```

Or run the equivalent steps interactively in
`notebooks/phase3_hopsworks_feature_store.ipynb` (recommended on Windows
without Build Tools — see the setup note above).

### Validated Run

The v2 feature group was populated from the post-fix `feature_df.csv`
(`feels_like` fully populated, corrected rolling windows and forecast
horizon) and successfully round-tripped against a real Hopsworks project:

```
Logged in to project, explore it here https://eu-west.cloud.hopsworks.ai:443/p/<project_id>
Feature Group created successfully.
Uploading Dataframe: 100.00% |██████████| Rows 5887/5887
Inserted 5887 rows.
Read back 5887 rows.
```

**Note on materialization timing:** after `insert()`, Hopsworks runs an
asynchronous job (`aqi_features_2_offline_fg_materialization`) that writes
the Hudi table. Calling `read()` immediately afterward can transiently fail
with `FeatureStoreException: No hudi properties found for featuregroup: ...
This usually means that no data has been written yet` if that job hasn't
finished yet — this is expected, not a bug. Either check the job's status
in the Hopsworks UI's Jobs tab before reading, or retry `read()` a few times
with a short delay (the notebook includes a simple retry loop for this).

**Note on time travel format:** the feature group is created with
`time_travel_format="HUDI"` explicitly. Hopsworks' newer default,
`"DELTA"`, requires the `delta` library, which isn't installed by default
in a fresh environment (e.g. Colab) — using `HUDI` avoids that extra
dependency.

### Testing

```bash
pytest tests/test_feature_store.py -v
```

All Hopsworks/`hsfs` interaction is mocked — no real account or API key
required to run the suite. Covers: timestamp preparation/validation, login
(both the `hopsworks` and `hsfs` fallback paths), feature group creation,
insert, read, and a fully-mocked end-to-end sync.

---

## 14. Phase 4 — Exploratory Data Analysis (EDA)

`notebooks/EDA.ipynb` explores the Phase 2 engineered feature dataset
(`data/processed/feature_df.csv`) before model training. Open it in
Jupyter, VS Code, or Colab and run all cells; it auto-detects
`data/processed/feature_df.csv` relative to either the project root or
`notebooks/`.

**Sections:**

1. **Data overview** — shape, dtypes, city/date-range/source summary.
2. **Missing values** — per-column missing count/percentage, a bar chart,
   and a visual missing-pattern heatmap. Used during development to catch
   two real issues before they reached model training: `feels_like` being
   unexpectedly 100% null (a genuine bug, since fixed — see
   [Section 11](#11-historical-data-providers)), and `visibility` being
   100% null (a genuine ERA5 dataset limitation, not a bug — also documented
   there).
3. **Correlation analysis** — a full correlation heatmap, the top features
   ranked by correlation with `aqi_target`, and a flagged-pairs table for
   highly-correlated (`|r| >= 0.9`) feature pairs. This is what surfaced the
   `pollution_index` ≈ `pm2_5` (r = 1.000) redundancy for single-pollutant
   cities, leading to the Phase 2 fix described in
   [Section 12](#12-phase-2--feature-engineering).
4. **Histograms/distributions** — for AQI, PM2.5, temperature, humidity,
   pressure, wind speed, pollution index, and the target, plus a skewness
   table.
5. **Outlier detection** — IQR-based outlier counts/percentages and boxplots
   per numeric column (descriptive only — nothing is dropped by the
   notebook itself).
6. **AQI trends** — a time-series plot (raw + 24h rolling mean) and AQI
   distribution broken out by hour-of-day, month, and day-of-week, to
   visually confirm the seasonality the engineered cyclical features are
   meant to capture. Also used to diagnose whether the collection interval
   was consistent across the dataset (gaps in the timestamp sequence would
   show up as unusually large `aqi_change_rate`/lag values — this is how
   the interval-mismatch issue behind the Section 12 rolling-window fix was
   first suspected, before being confirmed against the actual pipeline
   config).
7. **Summary of findings** — a checklist template to fill in per run.

---

## 15. Phase 5 — Model Training Pipeline

`src/training_pipeline/` trains and compares forecasting models on the
Phase 2/3 feature set and saves the best one to a model registry.

```
Hopsworks Feature Store (or feature_df.csv)
      │
      ▼
  load_features()                    -- data_loader.py
      │
      ▼
  select_feature_columns()           -- preprocessing.py (excludes all-null
      │                                  columns: pm10_*, pollution_index, etc.)
      ▼
  time_based_train_test_split()      -- chronological split, NOT shuffled
      │                                  (see rationale below)
      ▼
  StandardScaler (fit on train only) ─┬─► Ridge Regression
                                       ├─► Random Forest
                                       └─► Neural Network (TensorFlow or
                                            scikit-learn MLPRegressor fallback)
      │
      ▼
  evaluate_model() -- RMSE / MAE / R² per candidate, PLUS a naive
      │                "persistence" baseline (predict no change)
      ▼
  select_best_model() -- lowest RMSE among trained candidates
      │                  (never the naive baseline itself)
      ▼
  save_model_locally()   -- data/model_registry/<model>_<timestamp>/
      │                      (model + scaler + metadata.json)
      ▼
  register_model_to_hopsworks()  -- optional, best-effort (--register-hopsworks)
```

### Why a time-based split, not a random shuffle

This is time-series data with lag and rolling features computed from each
row's temporal neighbors. A random shuffle-based split would let test-set
information leak into training-set rolling/lag features (and vice versa),
producing misleadingly optimistic validation metrics. `preprocessing.py`
instead sorts by `collection_timestamp` and takes the earliest
`1 - test_size` fraction as training data and the most recent `test_size`
fraction as the test set — both because it avoids that leakage, and because
it matches the real deployment scenario (predicting the future from the
past).

### Why a naive baseline is always computed

`evaluate.py`'s `compute_naive_baseline_metrics()` scores the trivial
"persistence" forecast — assume AQI doesn't change over the horizon, i.e.
predict `aqi_target ≈ current aqi` — on the exact same test rows every
trained model is judged against. This is included as its own row in the
comparison table, and `train_pipeline.py` logs an explicit
`PRODUCTION WARNING` if the best trained model does **not** beat it: a model
that can't outperform "assume no change" isn't adding real predictive
value and should not be treated as production-ready, however reasonable
its raw RMSE/MAE might look in isolation. This matters in particular for
this project because early experiments used a much smaller historical
range (see [Section 11](#11-historical-data-providers)) — a chronological
holdout test set can land in a different seasonal regime than most of the
training data, which is exactly the kind of failure mode a fixed reference
baseline is meant to catch. Re-training against the now-extended
2024-2026 dataset should be re-checked against this same baseline.

### Model candidates

| Model | Implementation |
|---|---|
| Ridge Regression | `sklearn.linear_model.Ridge` |
| Random Forest | `sklearn.ensemble.RandomForestRegressor` (200 trees) |
| Neural Network | TensorFlow/Keras (`Dense(64)→Dense(32)→Dense(1)`, early stopping) if installed, else `sklearn.neural_network.MLPRegressor` automatically |

The TensorFlow/scikit-learn fallback mirrors the same
optional-heavy-dependency pattern already used for Hopsworks in
`feature_store.py` (full `hopsworks` package vs. lighter `hsfs`): the
pipeline stays fully runnable, testable, and CI-friendly even where
installing TensorFlow isn't practical (e.g. limited disk space/bandwidth),
without silently downgrading model quality when it *is* available.

### How to Run

```bash
# From the Hopsworks Feature Store (default):
python -m src.training_pipeline.train_pipeline --source hopsworks

# From a local feature_df.csv (faster iteration, no Hopsworks credentials needed):
python -m src.training_pipeline.train_pipeline --source local

# Options:
python -m src.training_pipeline.train_pipeline \
    --source local \
    --test-size 0.2 \
    --random-state 42 \
    --register-hopsworks   # also push the best model to Hopsworks Model Registry (best-effort)
```

### Model Registry

Every run saves the best model, its fitted `StandardScaler`, and a
`metadata.json` (model name/type, feature column order, metrics, timestamp)
to a timestamped directory under `data/model_registry/` — this local copy
is always written and is always authoritative. With `--register-hopsworks`,
the pipeline *additionally* attempts to push the same model to the
Hopsworks Model Registry; this requires the full `hopsworks` package (Model
Registry access is part of `hsml`, not bundled in the lighter `hsfs`-only
install — see [Section 13](#13-phase-3--hopsworks-feature-store)'s Windows
note) and valid credentials. This push is deliberately **best-effort**: any
failure (missing package, bad credentials, network issue) is logged as a
warning and returns `False` rather than raising, since the local registry
copy is never dependent on it.

### Validated Run

Run end-to-end against the real Sukkur feature set (`--source local`);
example output shape (exact metrics vary by dataset/date range used —
see the **naive baseline** discussion above for how to interpret them, and
re-run against the newly-extended 22,608-row historical dataset once
Phase 2/3 have been re-run over it):

```
Model comparison (sorted by RMSE, best first):
                        model      rmse       mae        r2
                Random Forest     XX.XX     XX.XX     X.XXX
             Ridge Regression     XX.XX     XX.XX     X.XXX
Neural Network (MLP fallback)     XX.XX     XX.XX     X.XXX
Naive Baseline (persistence)      XX.XX     XX.XX     X.XXX

Best trained model: Random Forest
Saved to: data/model_registry/random_forest_<timestamp>/
[OK] Best model beats the naive persistence baseline.
```

If the printed result instead shows `[WARNING] Best model does NOT beat the
naive persistence baseline`, treat that run's model as **not
production-ready** — see the `PRODUCTION WARNING` log message for likely
causes (most commonly: the training/test split spans different seasonal
regimes because the available historical range was still limited at
training time). The now-extended ~2.5-year historical dataset (see
[Section 11](#11-historical-data-providers)) directly addresses this —
re-running Phase 2 → Phase 3 → Phase 5 over it is the recommended next step.

### Testing

```bash
pytest tests/test_training_*.py tests/test_train_pipeline.py -v
```

Everything is tested with synthetic data and full mocking of Hopsworks/TensorFlow
where relevant — no real account, API key, or heavy ML dependency install
required to run the suite. Covers: local/Hopsworks feature loading, feature
selection (all-null column exclusion), the time-based split's leakage-safety
properties, all three model candidates (including the TensorFlow/scikit-learn
fallback), RMSE/MAE/R² and naive-baseline computation, local model
registry save/load round-trips, best-effort Hopsworks registry push
failure handling, and a fully end-to-end training run.

---

## 16. Phase 6 — Forecast Dashboard & Explainability

`src/dashboard/` serves real-time next-3-day AQI forecasts and explains them.
It loads the **v6 daily-average models** (`ridge_3day_dailyavg_v6_*` in
`data/model_registry/`) — three Ridge regressors trained on daily features
(see below) that predict the **mean AQI over the next 1, 2, and 3 days**.
v6 extends the earlier v5 recipe with **future-weather features**: the wind,
humidity, temperature, cloud cover and pressure expected *during the days being
forecast*, taken from the OpenWeather 5-day forecast at inference and from the
archived series at training time. Models are trained by
`python -m src.training_pipeline.train_v6`.

### Why daily-average targets (v5/v6) instead of the hourly 72h target?

The Phase 2 target (`aqi_target`) predicts a single hourly AQI value 72 hours
ahead. The project's real goal — *"predict the AQI in the next 3 days"* — is
a **daily** forecast. Experimenting against the hourly target hit a hard
ceiling around R² ≈ 0.56-0.60 (the test period is a distinct seasonal/regime
shift, and much of the hourly variance is within-day noise), well short of
the R² ≥ 0.75 goal. Switching the target to the *mean AQI over the next 1/2/3
days* both matches how a 3-day AQI forecast is actually consumed and
evaluated, and reaches the target. Adding future-weather features (v6) lifts
R² further:

| Horizon | Model (Ridge, scaled) | Test R² (realistic forecast) | Test RMSE | Naive persistence R² |
|---|---|---|---|---|
| Day 1 (mean AQI, next 1 day) | alpha=100, 51 features | **0.7871** | 17.49 | 0.7279 |
| Day 2 (mean AQI, next 2 days) | alpha=300, 56 features | **0.7824** | 17.08 | 0.6808 |
| Day 3 (mean AQI, next 3 days) | alpha=300, 61 features | **0.7934** | 16.26 | 0.6638 |

For comparison, v5 (no future-weather features) scored 0.7635 / 0.7553 /
0.7553 on the same holdout.

**On leakage — verified, not assumed.** An earlier version of `train_v6.py`
built `fw_*` (future-weather) features directly from archived *actual*
weather for the target days, for both train and test — meaning the backtest
measured the model against perfect future weather it will never have at
inference. This was caught during review and fixed: `train_v6.py` now
injects calibrated, horizon-growing simulated forecast uncertainty into
`fw_*` for both train and test (see `inject_forecast_uncertainty()`), so the
holdout reflects realistic forecast imperfection rather than ground truth.

Two checks confirm the fix didn't just hide the problem:
- **Leakage-gap check**: re-evaluating the same trained model against the
  original (perfect-weather) holdout gives R² 0.7861 / 0.7820 / 0.7916 —
  within ~0.002 of the realistic numbers above. The original headline
  numbers turn out to have been approximately correct despite the flawed
  evaluation design, but this is now *measured*, not assumed.
- **Ablation check** (`train_v6_ablation.py`): a historical-features-only
  model (no `fw_*` at all) scores 0.7660 / 0.7553 / 0.7618 — meaningfully
  below the full v6 model, with the gap **growing by horizon** (+0.021 /
  +0.027 / +0.032 for Day 1/2/3). That growing-with-horizon pattern is the
  signature of a genuine effect, not noise-fitting, and confirms
  future-weather features are real, useful signal rather than leakage
  disguised as improvement.

The noise-injection calibration (`FORECAST_NOISE_STD` in `train_v6.py`) is
currently a rough estimate from general NWP forecast-skill literature, not
measured against this project's actual OpenWeather error — worth replacing
with empirically logged forecast-vs-actual error once `fetch_future_weather()`
has accumulated enough live history.

### Daily feature recipe

`src/dashboard/forecast.py::build_daily_features()` reproduces the training
recipe exactly:

1. Resample the hourly `feature_df.csv` to **daily** means of AQI, PM2.5,
   temperature, wind, humidity, pressure, clouds (plus daily max/min/std for
   AQI and wind/temp).
2. Derive 46 historical model features: cyclical day-of-year / day-of-week
   encodings, AQI lags (1-14 days), pollutant/weather lags (1-3 days), rolling
   means (3/7/14/30 days), rolling stds (7/14), anomalies and trend deltas,
   daily range, and humidity/temperature interaction terms.
3. Add v6 **future-weather features** (`fw_*`): wind/humidity/temperature/
   clouds/pressure for the next 1-3 forecast days. Each horizon uses exactly
   the days it predicts (Day 1 → `fw_*1`; Day 2 → `fw_*1..2`; Day 3 →
   `fw_*1..3`).

At inference, `fetch_future_weather()` builds these from the archived
historical dataset when the target days are already observed, and from the
OpenWeather 5-day forecast otherwise, so the forecast uses real expected
weather instead of assuming persistence.

### How to Run

```bash
# Prerequisites: v6 models trained (python -m src.training_pipeline.train_v6)
#               and feature_df.csv present (python -m src.feature_pipeline.feature_engineering)

streamlit run src/dashboard/app.py
```

The dashboard fetches live future weather on startup; on a transient API
failure it degrades gracefully to archived/persistence weather and the
forecast still renders.

### Dashboard contents

- **3-day forecast cards** — mean AQI for Day 1/2/3, color-coded by US EPA
  AQI category, with each model's test R².
- **Hazardous-AQI alerts** — a banner when the peak predicted AQI crosses the
  alert thresholds (≥150 warning, ≥200 hazardous), per the EPA category
  thresholds documented in `app.py`.
- **Historical context chart** — last 30 days of observed daily AQI plus the
  3-day forecast.
- **SHAP / LIME explanations** — for each horizon, the top contributing
  features that drove the prediction up or down (`src/dashboard/explainability.py`).
  SHAP is exact for the linear models (base + Σ SHAP = prediction); LIME
  provides a complementary model-agnostic approximation.
- **Model details** — horizon metadata, alphas, split sizes, test R² vs
  naive baseline, and the model registry run path.

### Testing

```bash
pytest tests/test_dashboard_forecast.py tests/test_dashboard_explainability.py -v
```

Tests use synthetic hourly data and toy v5/v6-style Ridge models written to a
temp registry — no real data or trained artifacts required. They cover: the
daily-feature recipe's column contract, future-weather feature injection and
persistence fallback, model/scaler loading for all three horizons, forecast
plausibility, insufficient-history error handling, and SHAP's additive
reconstruction (base + Σ SHAP = prediction).

---

## 17. Out of Scope (Later Phases)

The following are intentionally **not** implemented yet, and belong to
later phases of the project:

- Hyperparameter tuning / more extensive model experimentation
- CI/CD automation (GitHub Actions) — the feature pipeline running hourly
  and a training pipeline running daily
- Deploying the dashboard to a managed serverless host (e.g. Streamlit
  Community Cloud)
- Multi-city support (the pipeline is currently configured and validated
  for a single city, Sukkur, at a time)