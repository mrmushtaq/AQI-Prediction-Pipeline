"""Unit tests for `src.dashboard.forecast`."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.common.exceptions import TrainingError
from src.dashboard.forecast import (
    DEFAULT_REGISTRY_DIR,
    V5_RUN_PREFIX,
    build_daily_features,
    load_daily_avg_models,
    make_forecast,
)


def _make_v5_run(registry_dir):
    """Create a realistic v5 registry run with small Ridge models + scalers."""
    run_dir = registry_dir / f"{V5_RUN_PREFIX}_20260819T000000Z"
    run_dir.mkdir(parents=True, exist_ok=True)

    n = 120
    rng = np.random.default_rng(0)
    features = {
        "aqi": rng.uniform(60, 180, n),
        "pm25": rng.uniform(40, 150, n),
        "temp": rng.uniform(10, 45, n),
        "wind": rng.uniform(0, 20, n),
        "humid": rng.uniform(20, 90, n),
        "pressure": rng.uniform(990, 1010, n),
        "clouds": rng.uniform(0, 100, n),
        "doy_sin": np.sin(rng.uniform(0, 6.28, n)),
        "doy_cos": np.cos(rng.uniform(0, 6.28, n)),
        "dow": rng.integers(0, 7, n).astype(float),
        "dow_sin": np.sin(rng.uniform(0, 6.28, n)),
        "dow_cos": np.cos(rng.uniform(0, 6.28, n)),
    }
    for k in [1, 2, 3, 4, 5, 7, 10, 14]:
        features[f"aqi_l{k}"] = rng.uniform(60, 180, n)
    for k in [1, 2, 3]:
        features[f"pm25_l{k}"] = rng.uniform(40, 150, n)
        features[f"temp_l{k}"] = rng.uniform(10, 45, n)
        features[f"wind_l{k}"] = rng.uniform(0, 20, n)
        features[f"humid_l{k}"] = rng.uniform(20, 90, n)
    for w in [3, 7, 14, 30]:
        features[f"rm{w}"] = rng.uniform(60, 180, n)
    for w in [7, 14]:
        features[f"std{w}"] = rng.uniform(1, 20, n)
    features["anom7"] = rng.uniform(-40, 40, n)
    features["anom14"] = rng.uniform(-40, 40, n)
    features["trend3"] = rng.uniform(-10, 10, n)
    features["trend7"] = rng.uniform(-10, 10, n)
    features["trend14"] = rng.uniform(-10, 10, n)
    features["range_l1"] = rng.uniform(5, 60, n)
    features["hum_temp"] = rng.uniform(5, 30, n)
    features["wind_hum"] = rng.uniform(0, 15, n)
    X = pd.DataFrame(features)
    feature_cols = list(X.columns)

    target = 0.5 * features["aqi"] + 0.3 * features["pm25"] + rng.normal(0, 5, n)

    metadata = {}
    for label, n_days in [("day1_avg", 1), ("day2_avg", 2), ("day3_avg", 3)]:
        scaler = StandardScaler().fit(X)
        model = Ridge(alpha=100).fit(scaler.transform(X), target)
        joblib.dump(model, run_dir / f"{label}_model.joblib")
        joblib.dump(scaler, run_dir / f"{label}_scaler.joblib")
        metadata[label] = {
            "n_days": n_days,
            "alpha": 100,
            "feature_cols": feature_cols,
            "n_train": n,
            "n_test": 10,
            "target": f"mean AQI over next {n_days} day(s)",
        }

    import json

    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return run_dir


def _make_hourly_df(n_days=40, hours_per_day=24):
    """Synthetic hourly engineered feature dataframe with ~40 days of history."""
    idx = pd.date_range("2026-01-01", periods=n_days * hours_per_day, freq="h")
    rng = np.random.default_rng(1)
    daily_cycle = 30 * np.sin(2 * np.pi * np.arange(len(idx)) / 24)
    aqi = 100 + daily_cycle + rng.normal(0, 8, len(idx))
    aqi = np.clip(aqi, 20, 300)
    return pd.DataFrame(
        {
            "collection_timestamp": idx,
            "aqi": aqi,
            "pm2_5": aqi * 0.85,
            "temperature": 25 + 10 * np.sin(2 * np.pi * np.arange(len(idx)) / 24),
            "wind_speed": rng.uniform(0, 15, len(idx)),
            "humidity": rng.uniform(30, 80, len(idx)),
            "pressure": rng.uniform(995, 1008, len(idx)),
            "clouds": rng.uniform(0, 100, len(idx)),
        }
    )


def test_build_daily_features_produces_expected_columns():
    df = _make_hourly_df()
    daily, feature_cols = build_daily_features(df)

    assert len(daily) == 40
    for col in [
        "aqi",
        "pm25",
        "doy_sin",
        "dow",
        "aqi_l1",
        "aqi_l14",
        "rm3",
        "rm30",
        "std7",
        "anom7",
        "trend14",
        "range_l1",
        "hum_temp",
        "wind_hum",
    ]:
        assert col in feature_cols
    assert "aqi_max" not in feature_cols  # derived helpers are excluded
    assert "collection_timestamp" not in feature_cols


def test_build_daily_features_raises_on_missing_columns():
    df = _make_hourly_df().drop(columns=["humidity"])
    with pytest.raises(TrainingError):
        build_daily_features(df)


def test_load_daily_avg_models_loads_all_horizons(tmp_path):
    run_dir = _make_v5_run(tmp_path)
    models, metadata = load_daily_avg_models(run_dir=run_dir)

    assert set(models.keys()) == {"day1_avg", "day2_avg", "day3_avg"}
    assert set(metadata.keys()) == {"day1_avg", "day2_avg", "day3_avg"}
    for label, (model, scaler) in models.items():
        assert isinstance(model, Ridge)
        assert isinstance(scaler, StandardScaler)


def test_load_daily_avg_models_raises_when_no_run(tmp_path):
    with pytest.raises(TrainingError):
        load_daily_avg_models(run_dir=None, registry_dir=tmp_path / "empty")


def test_make_forecast_returns_plausible_values(tmp_path):
    run_dir = _make_v5_run(tmp_path)
    df = _make_hourly_df()
    forecast = make_forecast(df, run_dir=run_dir)

    assert forecast.day1_avg > 0
    assert forecast.day2_avg > 0
    assert forecast.day3_avg > 0
    assert forecast.forecast_date == df["collection_timestamp"].max().floor("D").strftime("%Y-%m-%d")
    assert forecast.run_dir == str(run_dir)
    assert len(forecast.feature_cols) >= 40


def test_make_forecast_raises_with_insufficient_history(tmp_path):
    run_dir = _make_v5_run(tmp_path)
    df = _make_hourly_df(n_days=3)  # too little history for 30-day rolling features
    with pytest.raises(TrainingError):
        make_forecast(df, run_dir=run_dir)


def test_build_daily_features_adds_future_weather_columns():
    df = _make_hourly_df()
    daily, feature_cols = build_daily_features(df)

    assert "fw_wind1" in feature_cols
    assert "fw_humid3" in feature_cols
    assert "fw_pressure2" in feature_cols
    # Persistence fallback keeps the anchor row scoreable without live weather.
    for col in ["fw_wind1", "fw_humid1", "fw_temp1", "fw_clouds1", "fw_pressure1"]:
        assert daily[col].notna().all()


def test_build_daily_features_fills_anchor_from_future_weather():
    df = _make_hourly_df()
    anchor = df["collection_timestamp"].max().floor("D")
    future_weather = pd.DataFrame(
        {
            "wind": [5.0, 6.0, 7.0],
            "humid": [40.0, 45.0, 50.0],
            "temp": [30.0, 31.0, 32.0],
            "clouds": [10.0, 20.0, 30.0],
            "pressure": [1000.0, 1001.0, 1002.0],
        },
        index=[anchor + pd.Timedelta(days=h) for h in range(1, 4)],
    )
    daily, _ = build_daily_features(df, future_weather=future_weather)

    last = daily.iloc[-1]
    assert last["fw_wind1"] == pytest.approx(5.0)
    assert last["fw_wind3"] == pytest.approx(7.0)
    assert last["fw_temp2"] == pytest.approx(31.0)
    assert last["fw_pressure3"] == pytest.approx(1002.0)


def test_make_forecast_accepts_future_weather(tmp_path):
    run_dir = _make_v5_run(tmp_path)
    df = _make_hourly_df()
    anchor = df["collection_timestamp"].max().floor("D")
    future_weather = pd.DataFrame(
        {
            "wind": [5.0, 6.0, 7.0],
            "humid": [40.0, 45.0, 50.0],
            "temp": [30.0, 31.0, 32.0],
            "clouds": [10.0, 20.0, 30.0],
            "pressure": [1000.0, 1001.0, 1002.0],
        },
        index=[anchor + pd.Timedelta(days=h) for h in range(1, 4)],
    )
    forecast = make_forecast(df, run_dir=run_dir, future_weather=future_weather)

    assert forecast.day1_avg > 0
    assert forecast.day2_avg > 0
    assert forecast.day3_avg > 0