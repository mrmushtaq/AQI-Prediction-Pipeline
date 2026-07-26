# Pearls AQI Predictor — Phase 1: Data Ingestion Pipeline

Production-grade data collection pipeline for the **AQI Prediction System**,
built for the **10Pearls Shine Data Sciences Internship**.

This phase implements **only** the ingestion pipeline: automatically fetching
weather and air-quality data from external APIs, validating it, and persisting
raw data, processed data, and per-run metadata to disk. No machine learning,
feature engineering, or serving layer is included in this phase — those
belong to later phases of the project.

**Status: Phase 1 complete and validated.** 100 automated tests pass, covering
every fetcher stage, every failure mode, configuration validation, and
end-to-end orchestration (see [Section 8](#8-testing--validated-scenarios)).

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
12. [Out of Scope for Phase 1](#12-out-of-scope-for-phase-1)

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
               │                        │               │ Historical: OpenAQ v3│
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
```

Live weather flows through the **OpenWeather** fetcher; historical weather
flows through the **Open-Meteo Archive** fetcher; historical AQI flows through
the **OpenAQ v3** API (replacing the retired OpenAQ v2). All three converge on
the same `merge_weather_and_aqi()` and `validate_record()` pipeline, so
downstream processing is identical regardless of data source.

A metadata record is written **regardless of outcome** — success or failure —
so every ingestion run leaves an auditable trail.

> **Note on future phases:** later phases of this project (feature
> engineering, model training, Hopsworks feature store, serving layer) will
> sit *downstream* of this pipeline's `data/processed/` output. They are not
> part of Phase 1 and are not implemented yet — see
> [Section 12](#12-out-of-scope-for-phase-1).

---

## 3. Folder Structure

```
pearls-aqi-predictor/
├── configs/
│   └── config.py                  # Centralized, validated configuration loader
├── data/
│   ├── raw/                       # Raw JSON API responses (one file per run)
│   ├── processed/                 # Processed CSV dataset (appended each run)
│   └── metadata/                  # Per-run metadata JSON (status, sources, errors)
├── logs/
│   └── pipeline.log               # Rotating log file (console output is mirrored here)
├── src/
│   ├── feature_pipeline/
│   │   ├── fetch_weather.py       # OpenWeather (live) + Open-Meteo Archive (historical)
│   │   ├── fetch_air_quality.py   # Live: 3-stage AQICN/OpenWeather fallback. Historical: OpenAQ v3
│   │   ├── feature_pipeline.py    # Shared merge_weather_and_aqi() + validate_record() logic
│   │   ├── historical_backfill.py # Orchestrates backfill over a date range
│   │   ├── live_pipeline.py       # Orchestrates a single current-time run
│   │   └── run_pipeline.py        # CLI entry point for a single live run + metadata
│   ├── utils/
│   │   ├── storage.py             # JSON/CSV read & write helpers
│   │   └── data_validator.py      # Validation rules for merged records
│   └── common/
│       ├── logger.py              # Centralized console + file logging
│       └── exceptions.py          # APIError, ConfigurationError, ValidationError, StorageError
├── tests/
│   ├── conftest.py                 # Shared fast-failing Settings fixture
│   ├── test_config.py              # Missing/invalid env var -> ConfigurationError
│   ├── test_fetch_weather.py       # Success, retry, timeout, non-retryable, malformed payload
│   ├── test_fetch_air_quality.py   # All 3 fallback stages + exponential backoff + OpenAQ v3
│   ├── test_data_validator.py      # Required/optional fields, ranges, timestamps
│   ├── test_storage.py             # JSON/CSV round-trip, directory creation, errors
│   ├── test_feature_pipeline.py    # Merge + validation logic
│   ├── test_historical_backfill.py # Dedup, incremental writes, metadata, --force
│   ├── test_live_pipeline.py       # Live pipeline end-to-end
│   └── test_run_pipeline.py        # Metadata on success/failure, e2e run
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
| `OPENAQ_API_KEY`           | Free API key from https://explore.openaq.org/register (required for historical AQI backfill) |
| `DEFAULT_CITY`              | City name used for the AQICN city-name stage (e.g. `Sukkur`)     |
| `OPENWEATHER_LAT`           | Latitude, used for OpenWeather and the AQICN geo-fallback stage  |
| `OPENWEATHER_LON`           | Longitude, used for OpenWeather and the AQICN geo-fallback stage |
| `LOG_LEVEL`                 | `DEBUG` / `INFO` / `WARNING` / `ERROR` (default: `INFO`)          |
| `REQUEST_TIMEOUT_SECONDS`   | Per-request timeout in seconds (default: `10`)                   |
| `MAX_RETRIES`               | Retry attempts per API call (default: `3`)                       |
| `RETRY_BACKOFF_SECONDS`     | Base backoff between retries; grows **exponentially** as `base * 2^(attempt-1)` (default base: `2`, i.e. 2s, 4s, 8s...) |

   `.env` is git-ignored — never commit real credentials. If any required
   variable is missing, empty, or an invalid lat/lon, the pipeline raises a
   `ConfigurationError` immediately, before any network call is made.

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

(On Windows, use Task Scheduler with the same command.)

### Historical backfill (past date range)

```bash
python -m src.feature_pipeline.historical_backfill <start_date> <end_date>

# Example:
python -m src.feature_pipeline.historical_backfill 2025-11-21 2026-07-26
```

This collects data at a fixed 3-hour interval across the given range, using
Open-Meteo Archive (weather) and OpenAQ v3 (AQI), and appends to
`data/processed/aqi_dataset_historical.csv`. The backfill is **idempotent**
— re-running the same range skips timestamps already saved, so an
interrupted run can simply be re-run safely.

To force re-collection of a range that's already been backfilled (e.g. to
refresh data), add `--force`:

```bash
python -m src.feature_pipeline.historical_backfill 2025-11-21 2026-07-26 --force
```

> **Note:** `--force` appends fresh rows rather than replacing existing ones,
> since storage is append-only. Only use it if you're prepared to
> deduplicate afterward.

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
pytest            # run all 100 tests
pytest -v         # verbose, one line per test
```

All external HTTP calls are mocked — no real API keys or network access are
needed to run the suite. The following scenarios are explicitly covered:

| Test Case                          | Expected Result                          | Covered by |
|-------------------------------------|-------------------------------------------|------------|
| City-name AQICN lookup succeeds      | `source == "AQICN-city"`, no fallback      | `test_stage1_aqicn_city_succeeds_without_further_stages` |
| Invalid/unrecognized city ("Unknown station") | Falls back to AQICN geo lookup | `test_stage2_aqicn_geo_used_when_city_query_fails` |
| AQICN fully down (both stages)       | Falls back to OpenWeather Air Pollution    | `test_stage3_openweather_used_when_both_aqicn_stages_fail` |
| OpenWeather Air Pollution still works when AQICN is down | Pipeline still returns valid data | same test — asserts `source == "OpenWeather-AirPollution"` |
| Both AQI sources down                | Pipeline raises `APIError`, exits gracefully | `test_raises_when_all_three_stages_fail`, `test_run_pipeline_failure_still_writes_failed_metadata` |
| Invalid/expired API key (HTTP 401)   | Non-retryable `APIError` raised immediately | `test_fetch_current_weather_raises_api_error_on_non_retryable_status` |
| Request timeout / connection drop    | Retries with exponential backoff, then `APIError` | `test_fetch_current_weather_raises_api_error_on_timeout`, `test_exponential_backoff_used_between_retries` |
| Malformed/unexpected API response    | `APIError` raised rather than silently corrupting data | `test_fetch_current_weather_raises_api_error_on_malformed_payload` |
| Missing required `.env` variable     | `ConfigurationError` raised before any network call | `test_load_settings_raises_configuration_error_when_required_var_missing` (parametrized over all required vars) |
| Empty/whitespace `.env` variable     | `ConfigurationError`                       | `test_load_settings_raises_configuration_error_when_required_var_empty` |
| Non-numeric or out-of-range lat/lon  | `ConfigurationError`                       | `test_load_settings_raises_configuration_error_on_*` |
| Partial pollutant data (station lacks some sensors) | Validation passes, nulls preserved | `test_validate_record_accepts_partial_pollutant_data` |
| Historical backfill dedup            | Re-running the same range skips already-saved timestamps | `test_historical_backfill.py` |
| Historical backfill `--force`        | Bypasses dedup, re-fetches requested range | `test_historical_backfill.py` |
| Pipeline succeeds end-to-end         | Metadata `status=SUCCESS`, CSV row written | `test_run_pipeline_success_writes_success_metadata` |
| Pipeline fails at any stage          | Metadata `status=FAILED` with error + partial context, exception still propagates | `test_run_pipeline_failure_still_writes_failed_metadata`, `test_run_pipeline_validation_failure_writes_failed_metadata` |

**Manually verified against live APIs** (not mocked) for **Sukkur**
(`27.7052, 68.8574`): a successful live end-to-end run producing real weather
+ AQI data (Section 7), and a real historical backfill of **1,963 records**
covering the range the AQI sensor was actually active for (see
[Section 11](#11-historical-data-providers) below). The AQICN city→geo
fallback was also observed triggering correctly on Sukkur's unrecognized
city string in both the live run and earlier debugging sessions.

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
  `ValidationError`, `StorageError`) makes failure modes explicit and lets
  callers catch precisely what they need to handle.
- **Fail fast, fail loud**: validation raises rather than silently coercing or
  dropping bad data, so problems surface at ingestion time rather than deep
  inside a future model.
- **Provider-agnostic schema**: AQICN (by city, and by lat/lon) and
  OpenWeather's Air Pollution API are all normalized into the exact same
  dictionary shape, so `run_pipeline.py` and everything downstream of it
  never needs to know which provider/stage actually responded. The `source`
  field on the returned record (`AQICN-city`, `AQICN-geo`, or
  `OpenWeather-AirPollution`) is the only place that distinction is visible.
- **Retry + exponential backoff + timeout** are applied uniformly to every
  outbound request via a shared `_request_with_retry` helper in each fetcher.
  Backoff grows as `retry_backoff_seconds * 2 ** (attempt - 1)` (e.g. with a
  2s base: 2s, 4s, 8s, ...).
- **Optional pollutant fields**: not every AQICN monitoring station measures
  every pollutant — many report only AQI + PM2.5. `pm10`/`co`/`so2`/`no2`/`o3`
  are therefore allowed to be `null` (the key must still be present, so the
  dataset schema stays consistent), while `aqi` and `pm2_5` remain strictly
  required since they're universally available.
- **Metadata is diagnostic, not load-bearing**: `save_run_metadata` swallows
  its own storage errors (logs a warning) so a metadata-write hiccup can
  never mask or override the pipeline's real success/failure result.
- **Historical dedup is per `(collection_timestamp, city)`**: re-running a
  backfill for the same range is always safe and idempotent, since already-
  saved timestamps are skipped rather than duplicated.

## 11. Historical Data Providers

### Why These Providers?

The original implementation used **OpenWeather One Call API 3.0 Time Machine**
for historical weather and **OpenWeather Air Pollution History** for historical
AQI. However, both endpoints require a **paid subscription** — the free tier
only covers current data.

Additionally, **AQICN does not provide historical AQI**. It always returns the
latest observation regardless of the requested timestamp, which would silently
fabricate history if used for backfill.

After discussion with the project mentor, the historical providers were replaced
with **free API alternatives** that satisfy the project requirement of collecting
historical data through external APIs:

| Historical provider   | API                                  | Cost  |
|-----------------------|--------------------------------------|-------|
| **Weather**           | Open-Meteo Archive                   | Free  |
| **AQI**               | OpenAQ v3                            | Free  |

The project requirements state: *"Fetch raw weather and pollutant data from
external APIs like AQICN or OpenWeather."* The wording is illustrative, not
mandatory. What matters is that the pipeline collects real, measured data
through APIs — which both Open-Meteo and OpenAQ provide.

### Open-Meteo Archive (Weather)

- **Endpoint**: `https://archive-api.open-meteo.com/v1/archive`
- **No API key required** — unlimited free access for moderate use.
- Covers reanalysis data from the 1940s to near-real-time.
- Returns hourly temperature, humidity, pressure, wind speed/direction, cloud
  cover, precipitation, visibility, and WMO weather codes.

Weather descriptions are mapped from WMO weather interpretation codes using
a static lookup table (see `_map_wmo_weather_code` in `fetch_weather.py`).

### OpenAQ v3 (AQI)

- **OpenAQ v2 retired** — the legacy `/v2/measurements` endpoint is gone
  (returns HTTP 410 Gone). All historical AQI queries now use OpenAQ v3.
- **Authentication required** — `OPENAQ_API_KEY` must be set in `.env`.
  Get a free API key at https://explore.openaq.org/register.
- **Hierarchical API** — v3 uses a locations → sensors → measurements model:
  1. `GET /v3/locations?coordinates=lat,lon&radius=25000` finds nearby stations.
  2. Sensor objects (embedded in the location response) identify which
     pollutants each station measures.
  3. `GET /v3/sensors/{id}/measurements?datetime_from=...&datetime_to=...`
     returns raw measurement values per parameter, searched over a widened
     time window (±6h) and across up to 5 candidate sensors per pollutant to
     tolerate sparsely-reporting stations (see below).
- Since OpenAQ returns **raw pollutant concentrations** rather than a ready-made
  AQI value, the AQI is computed from concentrations using the official
  **US EPA AQI breakpoint formula**. This ensures historical AQI values are
  derived from real measurements rather than fabricated data.

#### AQI Calculation

The EPA breakpoint formula is:

```
AQI = ((I_high - I_low) / (C_high - C_low)) * (C - C_low) + I_low
```

Where *C* is the pollutant concentration and the breakpoint ranges
*(C_low, C_high)* and corresponding AQI ranges *(I_low, I_high)* are defined
in the EPA Technical Assistance Document. The overall AQI is the **maximum**
of all pollutant sub-indices.

Unit conversions are applied automatically when OpenAQ returns concentrations
in µg/m³ rather than ppm/ppb (standard conversion factors at 25°C, 1013 hPa).

### Note on per-location historical coverage

OpenAQ's ground-sensor network coverage varies significantly by city, and a
location existing in OpenAQ's registry does **not** guarantee it has
historical data going back as far as needed.

For **Sukkur** specifically, the nearest OpenAQ sensors were confirmed
reachable (HTTP 200, valid sensor IDs) but returned no measurements for dates
before **~November 21, 2025** — meaning the sensor(s) simply weren't
reporting data yet, not that the pipeline was misconfigured. This was
diagnosed by:

1. Confirming the fetch logic itself worked correctly (verified against a
   recent date, which succeeded).
2. Binary-searching backwards from a known-working date to a known-failing
   date to locate the actual activation boundary, rather than assuming a
   fixed historical window would always be available.

As a result, Sukkur's historical dataset covers **2025-11-21 through
present** (**1,963 records** at a 3-hour interval) rather than the
originally targeted 1.5–2 years. Cities with denser, longer-established
monitoring networks (e.g. Karachi, Lahore, Islamabad) may have longer usable
historical windows — this is worth re-checking per city rather than assumed.

### Current Architecture (Phase 1)

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
OpenAQ v3 (historical AQI)
        |
        v
Feature Pipeline (merge + validate)
        |
        v
data/processed/aqi_dataset_historical.csv
```

The live and historical pipelines converge on the **same feature pipeline**
(`process_features`), so all downstream processing — validation, storage,
metadata — is identical regardless of data source. Later project phases
will read from these processed CSVs to build a feature store (Hopsworks),
train models, and serve predictions — none of that is implemented yet (see
[Section 12](#12-out-of-scope-for-phase-1)).

### What stays the same

| Component                | Provider              | Status |
|--------------------------|-----------------------|--------|
| **Live weather**         | OpenWeather           | Unchanged |
| **Live AQI**             | AQICN (3-stage fallback) | Unchanged |
| **Historical weather**   | Open-Meteo Archive    | Unchanged |
| **Historical AQI**       | OpenAQ v3             | Migrated from v2 → v3 |
| Feature pipeline         | —                     | Unchanged |
| Validation & storage     | —                     | Unchanged |
| Metadata & logging       | —                     | Unchanged |

### Configuration

The historical backfill requires:

| Variable         | Description                                           |
|------------------|-------------------------------------------------------|
| `OPENAQ_API_KEY` | Free API key from https://explore.openaq.org/register |

Plus the existing `OPENWEATHER_LAT` and `OPENWEATHER_LON` coordinates (already
required for the live pipeline). If `OPENAQ_API_KEY` is missing or empty,
`load_settings()` raises a `ConfigurationError` immediately.

### Logging

Historical records are logged with:

```
weather_source = "Open-Meteo Archive"
aqi_source = "OpenAQ-v3"
stage=OpenAQ-v3
```

Live records continue to use:

```
weather_source = "OpenWeather"
aqi_source = "AQICN-city" | "AQICN-geo" | "OpenWeather-AirPollution"
stage=AQICN-city | AQICN-geo | OpenWeather-AirPollution
```

This distinction is visible in both the console log output and the
`weather_source` / `aqi_source` columns in the processed CSV dataset.

---

## 12. Out of Scope for Phase 1

The following are intentionally **not** implemented yet, and belong to later
phases of the project:

- Feature engineering (time features, cyclic features, lag/rolling features,
  weather-change deltas, composite pollution indices, target variable)
- Machine learning model training
- Feature store (Hopsworks) / model registry / SHAP explainability
- Flask / Streamlit serving layer
- CI/CD automation (GitHub Actions)