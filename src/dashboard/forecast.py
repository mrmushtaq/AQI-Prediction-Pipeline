"""Daily-average AQI forecasting models (v5/v6)."""

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
HOPS_FORECAST_MODEL_PREFIX = "aqi_daily_forecast_v6"

HORIZONS = [
    ("day1_avg", 1),
    ("day2_avg", 2),
    ("day3_avg", 3),
]

_FUTURE_WEATHER_VARS = [
    "wind",
    "humid",
    "temp",
    "clouds",
    "pressure",
]

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
    """A single next-3-day forecast."""

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

    if run_dir is None:
        try:
            return _load_daily_avg_models_from_hopsworks()
        except Exception as exc:
            logger.info(
                "Hopsworks forecast model load unavailable; "
                "using local registry: %s",
                exc,
            )

<<<<<<< Updated upstream
=======
    Returns:
        Tuple of `(models, metadata)` where `models[label] = (model, scaler)`
        and `metadata[label]` holds the per-horizon metrics/feature list.

    Raises:
        TrainingError: If no run can be resolved or any artifact is missing.
    """
    if run_dir is None:
        try:
            return _load_daily_avg_models_from_hopsworks()
        except Exception as exc:
            logger.info("Hopsworks forecast model load unavailable; using local registry: %s", exc)

>>>>>>> Stashed changes
    run_dir = Path(run_dir) if run_dir else _find_latest_run(registry_dir)

    if run_dir is None or not run_dir.is_dir():
        raise TrainingError(
            "No daily-average model run found in the local registry. "
            "Train one first via the v5/v6 training notebook."
        )

    metadata_path = run_dir / "metadata.json"

    if not metadata_path.exists():
        raise TrainingError(
            f"Missing metadata.json in model run '{run_dir}'."
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    models: dict[str, tuple[Any, Any]] = {}

    for label, _n_days in HORIZONS:

        model_path = run_dir / f"{label}_model.joblib"
        scaler_path = run_dir / f"{label}_scaler.joblib"

        if not model_path.exists() or not scaler_path.exists():
            raise TrainingError(
                f"Missing model/scaler artifacts for '{label}' "
                f"in '{run_dir}'."
            )

        models[label] = (
            joblib.load(model_path),
            joblib.load(scaler_path),
        )

    logger.info(
        "Loaded daily-average models (%d horizons) from %s",
        len(models),
        run_dir,
    )

    return models, metadata


<<<<<<< Updated upstream
def _load_daily_avg_models_from_hopsworks():
=======
def _load_daily_avg_models_from_hopsworks() -> tuple[dict[str, tuple[Any, Any]], dict[str, dict]]:
    """Load the latest v6 horizon artifacts from Hopsworks Model Registry."""
    from configs.config import load_hopsworks_settings
    import hopsworks

    settings = load_hopsworks_settings()
    project = hopsworks.login(project=settings.project_name, api_key_value=settings.api_key)
    model_registry = project.get_model_registry()
    models: dict[str, tuple[Any, Any]] = {}
    metadata: dict[str, dict] = {}
    for label, _n_days in HORIZONS:
        name = f"{HOPS_FORECAST_MODEL_PREFIX}_{label}"
        candidates = model_registry.get_models(name=name)
        if not candidates:
            raise TrainingError(f"No Hopsworks model found for '{name}'.")
        model_meta = max(candidates, key=lambda item: int(item.version))
        download_dir = Path(model_meta.download())
        model_dir = download_dir if download_dir.is_dir() else download_dir.parent
        model_path = model_dir / f"{label}_model.joblib"
        scaler_path = model_dir / f"{label}_scaler.joblib"
        if not model_path.exists() or not scaler_path.exists():
            raise TrainingError(f"Hopsworks model '{name}' lacks forecast artifacts.")
        models[label] = (joblib.load(model_path), joblib.load(scaler_path))
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            raise TrainingError(f"Hopsworks model '{name}' lacks metadata.json.")
        with metadata_path.open("r", encoding="utf-8") as handle:
            registered_metadata = json.load(handle)
        metadata[label] = registered_metadata.get(label, {})
    return models, metadata


def _find_latest_run(registry_dir: Path | str) -> Path | None:
    """Locate the most recent `ridge_3day_dailyavg_v*_*` run directory.
>>>>>>> Stashed changes

    from configs.config import load_hopsworks_settings
    import hopsworks

    settings = load_hopsworks_settings()

    project = hopsworks.login(
        project=settings.project_name,
        api_key_value=settings.api_key,
    )

    model_registry = project.get_model_registry()

    models: dict[str, tuple[Any, Any]] = {}
    metadata: dict[str, dict] = {}

    for label, _n_days in HORIZONS:

        name = f"{HOPS_FORECAST_MODEL_PREFIX}_{label}"

        candidates = model_registry.get_models(name=name)

        if not candidates:
            raise TrainingError(
                f"No Hopsworks model found for '{name}'."
            )

        model_meta = max(
            candidates,
            key=lambda item: int(item.version),
        )

        download_dir = Path(model_meta.download())

        model_dir = (
            download_dir
            if download_dir.is_dir()
            else download_dir.parent
        )

        model_path = model_dir / f"{label}_model.joblib"
        scaler_path = model_dir / f"{label}_scaler.joblib"

        if not model_path.exists() or not scaler_path.exists():
            raise TrainingError(
                f"Hopsworks model '{name}' lacks forecast artifacts."
            )

        models[label] = (
            joblib.load(model_path),
            joblib.load(scaler_path),
        )

        metadata_path = model_dir / "metadata.json"

        if not metadata_path.exists():
            raise TrainingError(
                f"Hopsworks model '{name}' lacks metadata.json."
            )

        with metadata_path.open("r", encoding="utf-8") as handle:
            registered_metadata = json.load(handle)

        metadata[label] = registered_metadata.get(label, {})

    return models, metadata


def _find_latest_run(
    registry_dir: Path | str,
) -> Path | None:

    registry_dir = Path(registry_dir)

    if not registry_dir.is_dir():
        return None

    candidates = sorted(
        p
        for p in registry_dir.iterdir()
        if p.name.startswith(V5_RUN_PREFIX)
        or p.name.startswith(V6_RUN_PREFIX)
    )

    return candidates[-1] if candidates else None


def build_daily_features(
    df: pd.DataFrame,
    future_weather: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:

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

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise TrainingError(
            f"Input data is missing required columns: {missing}. "
            "Re-run feature engineering."
        )

    df = df.copy()

    # IMPORTANT:
    # Live data and historical data can contain different timestamp formats.
    df["collection_timestamp"] = pd.to_datetime(
        df["collection_timestamp"],
        format="mixed",
        utc=True,
        errors="coerce",
    )

    before = len(df)

    df = df.dropna(
        subset=["collection_timestamp"]
    )

    dropped = before - len(df)

    if dropped:
        logger.warning(
            "Dropped %d rows with invalid collection_timestamp",
            dropped,
        )

    df = df.sort_values(
        "collection_timestamp"
    )

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
<<<<<<< Updated upstream
=======
    # Live ingestion can resume after several days without observations.
    # Empty calendar buckets would poison lag features and hide the live row.
    daily = daily.reset_index(drop=True)
>>>>>>> Stashed changes

    # Remove empty calendar days.
    daily = daily.dropna(
        subset=["aqi"]
    ).reset_index(drop=True)

    # Calendar features
    daily["day"] = (
        daily["collection_timestamp"]
        .dt.dayofyear
    )

    daily["doy_sin"] = np.sin(
        2 * np.pi * daily["day"] / 365.25
    )

    daily["doy_cos"] = np.cos(
        2 * np.pi * daily["day"] / 365.25
    )

    daily["dow"] = (
        daily["collection_timestamp"]
        .dt.dayofweek
    )

    daily["dow_sin"] = np.sin(
        2 * np.pi * daily["dow"] / 7
    )

    daily["dow_cos"] = np.cos(
        2 * np.pi * daily["dow"] / 7
    )

    # AQI lags
    for k in [1, 2, 3, 4, 5, 7, 10, 14]:
        daily[f"aqi_l{k}"] = (
            daily["aqi"].shift(k)
        )

    # Weather lags
    for k in [1, 2, 3]:

        daily[f"pm25_l{k}"] = (
            daily["pm25"].shift(k)
        )

        daily[f"temp_l{k}"] = (
            daily["temp"].shift(k)
        )

        daily[f"wind_l{k}"] = (
            daily["wind"].shift(k)
        )

        daily[f"humid_l{k}"] = (
            daily["humid"].shift(k)
        )

    # Rolling AQI
    for w in [3, 7, 14, 30]:

        daily[f"rm{w}"] = (
            daily["aqi"]
            .shift(1)
            .rolling(w)
            .mean()
        )

    for w in [7, 14]:

        daily[f"std{w}"] = (
            daily["aqi"]
            .shift(1)
            .rolling(w)
            .std()
        )

    # Derived features
    daily["anom7"] = (
        daily["aqi"] - daily["rm7"]
    )

    daily["anom14"] = (
        daily["aqi"] - daily["rm14"]
    )

    daily["trend3"] = (
        daily["rm7"] - daily["rm3"]
    )

    daily["trend7"] = (
        daily["rm7"] - daily["rm14"]
    )

    daily["trend14"] = (
        daily["rm14"] - daily["rm30"]
    )

    daily["range_l1"] = (
        daily["aqi_max"] -
        daily["aqi_min"]
    )

    daily["hum_temp"] = (
        daily["humid"] *
        daily["temp"] /
        100
    )

    daily["wind_hum"] = (
        daily["wind"] *
        daily["humid"] /
        100
    )

    # Future weather
    _add_future_weather_features(
        daily,
        future_weather,
    )

    feature_cols = [
        c
        for c in daily.columns
        if c not in _DERIVED_DAILY_COLUMNS
    ]

    return daily, feature_cols


def _add_future_weather_features(
    daily: pd.DataFrame,
    future_weather: pd.DataFrame | None,
) -> None:

    for h in range(1, 4):

        for var in _FUTURE_WEATHER_VARS:

            column = f"fw_{var}{h}"

            daily[column] = (
                daily[var]
                .shift(-h)
            )

            # Persistence fallback
            daily[column] = (
                daily[column]
                .fillna(daily[var])
            )

    if (
        future_weather is None
        or future_weather.empty
    ):
        return

    future_weather = future_weather.copy()

    future_weather.index = pd.to_datetime(
        future_weather.index,
        utc=True,
        errors="coerce",
    ).normalize()

    last_date = pd.Timestamp(
        daily["collection_timestamp"].iloc[-1]
    ).normalize()

    for h in range(1, 4):

        target_date = (
            last_date +
            pd.Timedelta(days=h)
        )

        if target_date not in future_weather.index:
            continue

        row = future_weather.loc[target_date]

        for var in _FUTURE_WEATHER_VARS:

            value = row[var]

            if pd.notna(value):

                daily.loc[
                    daily.index[-1],
                    f"fw_{var}{h}",
                ] = value


def make_forecast(
    df: pd.DataFrame,
    run_dir: Path | str | None = None,
    registry_dir: Path | str = DEFAULT_REGISTRY_DIR,
    future_weather: pd.DataFrame | None = None,
) -> Forecast:

    models, metadata = load_daily_avg_models(
        run_dir=run_dir,
        registry_dir=registry_dir,
    )

    daily, _feature_cols = build_daily_features(
        df,
        future_weather=future_weather,
    )

    all_feature_cols = sorted(
        {
            c
            for m in metadata.values()
            for c in m["feature_cols"]
        }
    )

    missing_features = [
        c
        for c in all_feature_cols
        if c not in daily.columns
    ]

    if missing_features:
        raise TrainingError(
            "Required model features are missing: "
            f"{missing_features}"
        )

    scoreable = daily.dropna(
        subset=all_feature_cols
    )

    if scoreable.empty:
        raise TrainingError(
            "No daily row has complete feature history "
            "to score. Need sufficient historical data."
        )

    last = scoreable.iloc[-1]

    preds: dict[str, float] = {}

    for label, _n_days in HORIZONS:

        model, scaler = models[label]

        feature_cols = metadata[label]["feature_cols"]

        X = last[
            feature_cols
        ].to_frame().T

        scaled = scaler.transform(X)

        prediction = model.predict(
            scaled
        )[0]

        preds[label] = float(
            prediction
        )

    forecast_date = (
        pd.Timestamp(
            last["collection_timestamp"]
        )
        + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    logger.info(
        "Forecast for %s: "
        "day1=%.1f day2=%.1f day3=%.1f",
        forecast_date,
        preds["day1_avg"],
        preds["day2_avg"],
        preds["day3_avg"],
    )

    return Forecast(
        run_dir=str(
            Path(run_dir)
            if run_dir
            else _find_latest_run(
                registry_dir
            )
        ),
        forecast_date=forecast_date,
        day1_avg=preds["day1_avg"],
        day2_avg=preds["day2_avg"],
        day3_avg=preds["day3_avg"],
        feature_cols=all_feature_cols,
        model_metadata=metadata,
    )


def fetch_future_weather(
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:

    from src.common.exceptions import APIError
    from configs.config import PROJECT_ROOT
    from src.feature_pipeline.fetch_weather import WeatherFetcher

    hist_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "aqi_dataset_historical.csv"
    )

    # Determine anchor date
    if df is not None:

        daily_obs, _ = build_daily_features(df)

        anchor = pd.Timestamp(
            daily_obs[
                "collection_timestamp"
            ].iloc[-1]
        ).normalize()

    else:

<<<<<<< Updated upstream
        anchor = (
            pd.Timestamp.utcnow()
            .normalize()
            - pd.Timedelta(days=3)
=======
    target_dates = [anchor + pd.Timedelta(days=h) for h in range(1, 4)]

    # Archived daily weather (if the historical dataset exists).
    archived: dict[pd.Timestamp, dict[str, float]] = {}
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        hist["collection_timestamp"] = pd.to_datetime(
    hist["collection_timestamp"],
    format="mixed",
    utc=True,
)
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
>>>>>>> Stashed changes
        )

    target_dates = [
        anchor + pd.Timedelta(days=h)
        for h in range(1, 4)
    ]

    archived: dict[
        pd.Timestamp,
        dict[str, float],
    ] = {}

    # Read historical weather
    if hist_path.exists():

        hist = pd.read_csv(
            hist_path
        )

        hist["collection_timestamp"] = pd.to_datetime(
            hist["collection_timestamp"],
            format="mixed",
            utc=True,
            errors="coerce",
        )

        hist = hist.dropna(
            subset=["collection_timestamp"]
        )

        for target in target_dates:

            mask = (
                hist["collection_timestamp"]
                >= target
            ) & (
                hist["collection_timestamp"]
                < target + pd.Timedelta(days=1)
            )

            day = hist.loc[mask]

            if day.empty:
                continue

            archived[target] = {
                "wind": day["wind_speed"].mean(),
                "humid": day["humidity"].mean(),
                "temp": day["temperature"].mean(),
                "clouds": day["clouds"].mean(),
                "pressure": day["pressure"].mean(),
            }

    # OpenWeather forecast
    forecast_daily = _fetch_openweather_daily()

    for target in target_dates:

        if (
            target not in archived
            and target in forecast_daily.index
        ):

            archived[target] = (
                forecast_daily
                .loc[target]
                .to_dict()
            )

    missing = [
        t
        for t in target_dates
        if t not in archived
    ]

    if missing:

        raise APIError(
            "No archived or forecast weather "
            "available for target day(s): "
            + ", ".join(
                d.date().isoformat()
                for d in missing
            )
        )

    fw = pd.DataFrame(
        archived
    ).T.loc[target_dates]

    fw.index.name = "date"

    fw = fw[
        [
            "wind",
            "humid",
            "temp",
            "clouds",
            "pressure",
        ]
    ]

    logger.info(
        "Fetched future weather for %d "
        "forecast day(s): %s .. %s",
        len(fw),
        fw.index[0].date(),
        fw.index[-1].date(),
    )

    return fw


def _fetch_openweather_daily() -> pd.DataFrame:

    from configs.config import load_settings
    from src.feature_pipeline.fetch_weather import WeatherFetcher

    settings = load_settings()

    fetcher = WeatherFetcher(
        settings
    )

    payload = fetcher.fetch_forecast()

    entries = payload.get(
        "forecast_entries",
        [],
    )

    rows: list[dict[str, Any]] = []

    for entry in entries:

        dt = pd.to_datetime(
            entry.get("dt"),
            unit="s",
            utc=True,
        )

        main = entry.get(
            "main",
            {},
        )

        wind = entry.get(
            "wind",
            {},
        )

        clouds = entry.get(
            "clouds",
            {},
        )

        rows.append(
            {
                "date": pd.Timestamp(
                    dt
                ).normalize(),

                "wind": wind.get(
                    "speed"
                ),

                "humid": main.get(
                    "humidity"
                ),

                "temp": main.get(
                    "temp"
                ),

                "clouds": clouds.get(
                    "all"
                ),

                "pressure": main.get(
                    "pressure"
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "wind",
                "humid",
                "temp",
                "clouds",
                "pressure",
            ]
        ).rename_axis("date")

    return (
        pd.DataFrame(rows)
        .dropna()
        .groupby(
            "date",
            as_index=False,
        )
        .mean()
        .set_index("date")
    )