"""Daily-average AQI forecasting models (v5/v6).

Reproduces the feature-engineering recipe and model loading used by
`notebooks/train_final_3day_models_v5.ipynb` (and its v6 successor) so the
dashboard can generate next-1/2/3-day (daily-average) AQI forecasts from the
hourly engineered feature dataframe without re-running the notebook.

The models are Ridge regressors trained on daily features derived by
resampling the hourly `feature_df.csv` to daily resolution, predicting the
mean AQI over the next 1, 2, and 3 days respectively. v6 adds *future
weather* features (wind/humidity/temperature/clouds/pressure for the days
being forecast), which come from the OpenWeather 5-day forecast at inference
time and from the archived series during training. Each horizon has its own
fitted `StandardScaler` and model, saved as `{label}_model.joblib` and
`{label}_scaler.joblib`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from configs.config import PROJECT_ROOT
from src.common.exceptions import TrainingError
from src.common.logger import get_logger

logger = get_logger(__name__)

DEFAULT_REGISTRY_DIR = PROJECT_ROOT / "data" / "model_registry"
V5_RUN_PREFIX = "ridge_3day_dailyavg_v5"
V6_RUN_PREFIX = "ridge_3day_dailyavg_v6"

HORIZONS = [("day1_avg", 1), ("day2_avg", 2), ("day3_avg", 3)]

# v6-only future-weather features: wind/humidity/temp/clouds/pressure for the
# next 1..3 forecast days. These are the weather expected during the target
# period (OpenWeather forecast at inference, archived values at training).
_FUTURE_WEATHER_VARS = ["wind", "humid", "temp", "clouds", "pressure"]

# Non-model features computed during daily aggregation that are used to derive
# model features but are not themselves part of the training feature set.
_DERIVED_DAILY_COLUMNS = [
    "collection_timestamp",
    "aqi_max",
    "aqi_min",
    "aqi_std",
    "wind_max",
    "temp_max",
    "day",
]


@dataclass
class Forecast:
    """A single next-3-day forecast with the context used to produce it."""

    run_dir: str
    forecast_date: str
    day1_avg: float
    day2_avg: float
    day3_avg: float
    feature_cols: list[str] = field(default_factory=list)
    model_metadata: dict[str, Any] = field(default_factory=dict)


def load_daily_avg_models(
    run_dir: Path | str | None = None,
    registry_dir: Path | str = DEFAULT_REGISTRY_DIR,
) -> tuple[dict[str, tuple[Any, Any]], dict[str, dict]]:
    """Load the daily-average Ridge models and their scalers.

    Args:
        run_dir: Explicit registry run directory, or `None` to use the most
            recent `ridge_3day_dailyavg_v*_*` run found in `registry_dir`.
        registry_dir: Root directory of the local model registry.

    Returns:
        Tuple of `(models, metadata)` where `models[label] = (model, scaler)`
        and `metadata[label]` holds the per-horizon metrics/feature list.

    Raises:
        TrainingError: If no run can be resolved or any artifact is missing.
    """
    run_dir = Path(run_dir) if run_dir else _find_latest_run(registry_dir)

    if run_dir is None or not run_dir.is_dir():
        raise TrainingError(
            "No daily-average model run found in the local registry. Train one first "
            "via the v5/v6 training notebook."
        )

    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise TrainingError(f"Missing metadata.json in model run '{run_dir}'.")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    models: dict[str, tuple[Any, Any]] = {}
    for label, _n_days in HORIZONS:
        model_path = run_dir / f"{label}_model.joblib"
        scaler_path = run_dir / f"{label}_scaler.joblib"
        if not model_path.exists() or not scaler_path.exists():
            raise TrainingError(f"Missing model/scaler artifacts for '{label}' in '{run_dir}'.")
        models[label] = (joblib.load(model_path), joblib.load(scaler_path))

    logger.info("Loaded daily-average models (%d horizons) from %s", len(models), run_dir)
    return models, metadata


def _find_latest_run(registry_dir: Path | str) -> Path | None:
    """Locate the most recent `ridge_3day_dailyavg_v*_*` run directory.

    Run directories carry a lexicographically-sortable UTC timestamp suffix
    (`<slug>_<YYYYMMDDTHHMMSSZ>`), so the latest is the max name.

    Args:
        registry_dir: Root directory of the local model registry.

    Returns:
        Path to the latest v5/v6 run, or `None` if none exists.
    """
    registry_dir = Path(registry_dir)
    if not registry_dir.is_dir():
        return None
    candidates = sorted(
        p for p in registry_dir.iterdir()
        if p.name.startswith(V5_RUN_PREFIX) or p.name.startswith(V6_RUN_PREFIX)
    )
    return candidates[-1] if candidates else None


def build_daily_features(
    df: pd.DataFrame,
    future_weather: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Resample the hourly engineered features to daily and derive model features.

    Mirrors the daily aggregation + feature derivation cells of
    `notebooks/train_final_3day_models_v5.ipynb` (and the v6 future-weather
    extension) exactly so predictions match the trained models.

    Args:
        df: The hourly engineered feature dataframe (feature_df.csv) with at
            least `collection_timestamp`, `aqi`, `pm2_5`, `temperature`,
            `wind_speed`, `humidity`, `pressure`, `clouds`.
        future_weather: Optional daily weather for the forecast days, as a
            DataFrame indexed by date with columns `wind`, `humid`, `temp`,
            `clouds`, `pressure`. When provided, the `fw_*` (future weather)
            features for the most recent row are filled from it; otherwise the
            future-weather columns are derived from the historical series (the
            training-time behaviour) and left NaN for the tail rows.

    Returns:
        Tuple of `(daily, feature_cols)` where `daily` is a daily dataframe
        containing all model features plus the derived helper columns, and
        `feature_cols` is the ordered list of the model features.

    Raises:
        TrainingError: If required source columns are missing.
    """
    required = [
        "collection_timestamp",
        "aqi",
        "pm2_5",
        "temperature",
        "wind_speed",
        "humidity",
        "pressure",
        "clouds",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise TrainingError(
            f"Input data is missing required columns: {missing}. "
            "Re-run feature engineering to regenerate feature_df.csv."
        )

    df = df.sort_values("collection_timestamp").copy()
    df["collection_timestamp"] = pd.to_datetime(df["collection_timestamp"])

    daily = (
        df.set_index("collection_timestamp")
        .resample("D")
        .agg(
            aqi=("aqi", "mean"),
            pm25=("pm2_5", "mean"),
            temp=("temperature", "mean"),
            wind=("wind_speed", "mean"),
            humid=("humidity", "mean"),
            pressure=("pressure", "mean"),
            clouds=("clouds", "mean"),
            aqi_max=("aqi", "max"),
            aqi_min=("aqi", "min"),
            aqi_std=("aqi", "std"),
            wind_max=("wind_speed", "max"),
            temp_max=("temperature", "max"),
        )
        .reset_index()
    )
    # Live ingestion can resume after several days without observations.
    # Empty calendar buckets would poison lag features and hide the live row.
    daily = daily.dropna(subset=["aqi"]).reset_index(drop=True)

    daily["day"] = daily["collection_timestamp"].dt.dayofyear
    daily["doy_sin"] = np.sin(2 * np.pi * daily["day"] / 365.25)
    daily["doy_cos"] = np.cos(2 * np.pi * daily["day"] / 365.25)
    daily["dow"] = daily["collection_timestamp"].dt.dayofweek
    daily["dow_sin"] = np.sin(2 * np.pi * daily["dow"] / 7)
    daily["dow_cos"] = np.cos(2 * np.pi * daily["dow"] / 7)

    for k in [1, 2, 3, 4, 5, 7, 10, 14]:
        daily[f"aqi_l{k}"] = daily["aqi"].shift(k)
    for k in [1, 2, 3]:
        daily[f"pm25_l{k}"] = daily["pm25"].shift(k)
        daily[f"temp_l{k}"] = daily["temp"].shift(k)
        daily[f"wind_l{k}"] = daily["wind"].shift(k)
        daily[f"humid_l{k}"] = daily["humid"].shift(k)

    for w in [3, 7, 14, 30]:
        daily[f"rm{w}"] = daily["aqi"].shift(1).rolling(w).mean()
    for w in [7, 14]:
        daily[f"std{w}"] = daily["aqi"].shift(1).rolling(w).std()

    daily["anom7"] = daily["aqi"] - daily["rm7"]
    daily["anom14"] = daily["aqi"] - daily["rm14"]
    daily["trend3"] = daily["rm7"] - daily["rm3"]
    daily["trend7"] = daily["rm7"] - daily["rm14"]
    daily["trend14"] = daily["rm14"] - daily["rm30"]
    daily["range_l1"] = daily["aqi_max"] - daily["aqi_min"]
    daily["hum_temp"] = daily["humid"] * daily["temp"] / 100
    daily["wind_hum"] = daily["wind"] * daily["humid"] / 100

    # v6: future-weather features for the days being forecast (h = 1..3).
    # Training uses the archived series (shift(-h)); at inference these are
    # filled from the supplied `future_weather` for the forecast days.
    _add_future_weather_features(daily, future_weather)

    feature_cols = [
        c for c in daily.columns if c not in _DERIVED_DAILY_COLUMNS
    ]
    return daily, feature_cols


def _add_future_weather_features(
    daily: pd.DataFrame,
    future_weather: pd.DataFrame | None,
) -> None:
    """Derive `fw_*` (future-weather) features for the next 1..3 days.

    Historical rows get their future weather from the series itself via
    `shift(-h)`. The most recent row (the forecast anchor) has no future data
    in the archive, so it falls back to persistence (its own weather) and is
    then overwritten with the supplied `future_weather` when the dates align.

    Args:
        daily: Daily dataframe (mutated in place).
        future_weather: Optional daily weather for the next 3 days, indexed by
            date, with columns `wind`, `humid`, `temp`, `clouds`, `pressure`.
    """
    for h in range(1, 4):
        for var in _FUTURE_WEATHER_VARS:
            daily[f"fw_{var}{h}"] = daily[var].shift(-h)
            # Persistence fallback so the anchor row is always scoreable even
            # when no live forecast is available / dates do not align.
            daily[f"fw_{var}{h}"] = daily[f"fw_{var}{h}"].fillna(daily[var])

    if future_weather is not None and not future_weather.empty:
        last_date = pd.Timestamp(daily["collection_timestamp"].iloc[-1]).normalize()
        fw_index_dates = {pd.Timestamp(idx).normalize(): idx for idx in future_weather.index}
        for h in range(1, 4):
            target_date = last_date + pd.Timedelta(days=h)
            if target_date not in fw_index_dates:
                continue
            row = future_weather.loc[fw_index_dates[target_date]]
            for var in _FUTURE_WEATHER_VARS:
                daily.loc[daily.index[-1], f"fw_{var}{h}"] = row[var]


def make_forecast(
    df: pd.DataFrame,
    run_dir: Path | str | None = None,
    registry_dir: Path | str = DEFAULT_REGISTRY_DIR,
    future_weather: pd.DataFrame | None = None,
) -> Forecast:
    """Generate a next-3-day daily-average AQI forecast from hourly features.

    Args:
        df: The hourly engineered feature dataframe (feature_df.csv).
        run_dir: Explicit registry run directory, or `None` for the latest.
        registry_dir: Root of the local model registry.
        future_weather: Optional daily weather for the forecast days (indexed
            by date, columns `wind`, `humid`, `temp`, `clouds`, `pressure`).
            Used to populate the v6 future-weather features for the most recent
            row; when omitted those features are taken from the archived series
            where available and NaN elsewhere.

    Returns:
        A `Forecast` with the predicted mean AQI over the next 1, 2, and 3
        days, evaluated from the most recent fully-featured daily row.

    Raises:
        TrainingError: If models cannot be loaded or no row can be scored.
    """
    models, metadata = load_daily_avg_models(run_dir=run_dir, registry_dir=registry_dir)
    daily, _feature_cols = build_daily_features(df, future_weather=future_weather)

    first_label = HORIZONS[0][0]

    # Each v6 horizon has its own feature subset.
    # Score each horizon using only the features required by that model.
    scoreable_by_horizon = {
        label: daily.dropna(subset=metadata[label]["feature_cols"])
        for label, _n_days in HORIZONS
    }
    if all(rows.empty for rows in scoreable_by_horizon.values()):
        raise TrainingError(
            "No daily row has complete feature history to score. Need ~30+ days "
            "of hourly data before the rolling features are populated."
        )

    last = max(
        (rows.iloc[-1] for rows in scoreable_by_horizon.values() if not rows.empty),
        key=lambda row: row["collection_timestamp"],
    )

    preds: dict[str, float] = {}
    for label, _n_days in HORIZONS:
        model, scaler = models[label]
        feature_cols = metadata[label]["feature_cols"]
        X = last[feature_cols].to_frame().T
        scaled = scaler.transform(X)
        preds[label] = float(model.predict(scaled)[0])

    # The anchor row is today's observed AQI; Day 1 is the following full day.
    forecast_date = (
        pd.Timestamp(last["collection_timestamp"]) + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    logger.info(
        "Forecast for %s: day1=%.1f day2=%.1f day3=%.1f",
        forecast_date,
        preds["day1_avg"],
        preds["day2_avg"],
        preds["day3_avg"],
    )

    return Forecast(
        run_dir=str(Path(run_dir) if run_dir else _find_latest_run(registry_dir)),
        forecast_date=forecast_date,
        day1_avg=preds["day1_avg"],
        day2_avg=preds["day2_avg"],
        day3_avg=preds["day3_avg"],
        feature_cols=sorted(_feature_cols),
        model_metadata=metadata,
    )


def fetch_future_weather(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build daily weather for the next 3 forecast days (archive + forecast).

    The daily-average models forecast from the most recent fully-featured
    daily row (the *anchor*, ~3 days behind the latest hourly timestamp because
    feature engineering reserves the target window). The future-weather
    features for that anchor are the weather on anchor+1..anchor+3 days:

    * If a target day is already observed in the historical dataset, its
      actual archived daily weather is used (this matches training-time
      conditions exactly).
    * Otherwise the OpenWeather 5-day forecast is averaged into a daily row.

    Args:
        df: Optional hourly engineered feature dataframe used only to determine
            the anchor date. Archived weather is read from
            `data/processed/aqi_dataset_historical.csv`, which extends further
            than feature_df.csv (whose tail is reserved for the target window).

    Returns:
        DataFrame indexed by date (UTC) with one row per forecast day and
        columns `wind`, `humid`, `temp`, `clouds`, `pressure`.

    Raises:
        ConfigurationError: If settings cannot be loaded.
        APIError: If a target day has no archived weather and no forecast.
    """
    from configs.config import PROJECT_ROOT, load_settings
    from src.common.exceptions import APIError
    from src.feature_pipeline.fetch_weather import WeatherFetcher

    hist_path = PROJECT_ROOT / "data" / "processed" / "aqi_dataset_historical.csv"

    if df is not None:
        daily_obs, _ = build_daily_features(df)
        anchor = pd.Timestamp(daily_obs["collection_timestamp"].iloc[-1]).normalize()
    else:
        anchor = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=3)

    target_dates = [anchor + pd.Timedelta(days=h) for h in range(1, 4)]

    # Archived daily weather (if the historical dataset exists).
    archived: dict[pd.Timestamp, dict[str, float]] = {}
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        hist["collection_timestamp"] = pd.to_datetime(hist["collection_timestamp"])
        for target in target_dates:
            mask = (hist["collection_timestamp"] >= target) & (
                hist["collection_timestamp"] < target + pd.Timedelta(days=1)
            )
            day = hist[mask]
            if not day.empty:
                archived[target] = {
                    "wind": day["wind_speed"].mean(),
                    "humid": day["humidity"].mean(),
                    "temp": day["temperature"].mean(),
                    "clouds": day["clouds"].mean(),
                    "pressure": day["pressure"].mean(),
                }

    # OpenWeather forecast for any target day not covered by the archive.
    forecast_daily = _fetch_openweather_daily()
    for target in target_dates:
        if target not in archived and target in forecast_daily.index:
            archived[target] = forecast_daily.loc[target].to_dict()

    missing = [t for t in target_dates if t not in archived]
    if missing:
        raise APIError(
            "No archived or forecast weather available for target day(s): "
            + ", ".join(d.date().isoformat() for d in missing)
        )

    fw = pd.DataFrame(archived).T.loc[target_dates]
    fw.index.name = "date"
    fw = fw[["wind", "humid", "temp", "clouds", "pressure"]]
    logger.info(
        "Fetched future weather for %d forecast day(s): %s .. %s",
        len(fw),
        fw.index[0].date(),
        fw.index[-1].date(),
    )
    return fw


def _fetch_openweather_daily() -> pd.DataFrame:
    """Averaged daily weather from the OpenWeather 5-day/3-hour forecast."""
    from configs.config import load_settings
    from src.feature_pipeline.fetch_weather import WeatherFetcher

    settings = load_settings()
    fetcher = WeatherFetcher(settings)
    payload = fetcher.fetch_forecast()
    entries = payload.get("forecast_entries", [])

    rows: list[dict[str, Any]] = []
    for entry in entries:
        dt = pd.to_datetime(entry.get("dt"), unit="s", utc=True)
        main = entry.get("main", {})
        wind = entry.get("wind", {})
        clouds = entry.get("clouds", {})
        rows.append(
            {
                "date": pd.Timestamp(dt).normalize(),
                "wind": wind.get("speed"),
                "humid": main.get("humidity"),
                "temp": main.get("temp"),
                "clouds": clouds.get("all"),
                "pressure": main.get("pressure"),
            }
        )
    return (
        pd.DataFrame(rows)
        .dropna()
        .groupby("date", as_index=False)
        .mean()
        .set_index("date")
    )
