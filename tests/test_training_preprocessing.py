"""Unit tests for `src.training_pipeline.preprocessing`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.exceptions import TrainingError
from src.training_pipeline.preprocessing import (
    FEATURE_COLUMNS,
    build_xy,
    make_time_series_cv,
    select_feature_columns,
    time_based_train_test_split,
    time_based_train_val_test_split,
)


def _make_df(n: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "collection_timestamp": dates,
            "city": "Sukkur",
            "weather": "clear sky",
            "weather_source": "OpenWeather",
            "aqi_source": "AQICN-geo",
            "aqi": np.linspace(50, 150, n),
            "pm2_5": np.linspace(30, 100, n),
            "pm10": np.nan,  # entirely null -- must be excluded
            "temperature": np.linspace(10, 40, n),
            "feels_like": np.linspace(8, 38, n),
            "hour": np.arange(n) % 24,
            "day_of_week": np.arange(n) % 7,
            "month": np.full(n, 1),
            "aqi_target": np.linspace(60, 160, n),
        }
    )


def test_select_feature_columns_restricts_to_whitelist_and_drops_all_null():
    df = _make_df()
    cols = select_feature_columns(df)

    # Everything returned must come from the curated FEATURE_COLUMNS whitelist.
    assert set(cols) <= set(FEATURE_COLUMNS)

    # Constant / metadata / target / all-null columns are never selected.
    assert "pm10" not in cols
    assert "city" not in cols
    assert "weather" not in cols
    assert "weather_source" not in cols
    assert "timestamp" not in cols
    assert "collection_timestamp" not in cols
    assert "aqi_target" not in cols

    # Raw calendar integers are replaced by cyclical encodings -- excluded.
    assert "hour" not in cols
    assert "day_of_week" not in cols
    assert "month" not in cols

    # Whitelist columns present and non-null are kept (incl. feels_like).
    assert "aqi" in cols
    assert "pm2_5" in cols
    assert "temperature" in cols
    assert "feels_like" in cols


def test_select_feature_columns_raises_when_nothing_usable():
    df = pd.DataFrame({"city": ["Sukkur"], "aqi_target": [100.0]})
    with pytest.raises(TrainingError):
        select_feature_columns(df)


def test_time_based_split_produces_expected_sizes_and_order():
    df = _make_df(100)
    train_df, test_df = time_based_train_test_split(df, test_size=0.2)

    assert len(train_df) == 80
    assert len(test_df) == 20
    # Train set must be strictly earlier than test set (no shuffling / no overlap).
    assert train_df["collection_timestamp"].max() < test_df["collection_timestamp"].min()


def test_time_based_split_rejects_invalid_test_size():
    df = _make_df(50)
    with pytest.raises(TrainingError):
        time_based_train_test_split(df, test_size=0.0)
    with pytest.raises(TrainingError):
        time_based_train_test_split(df, test_size=1.0)


def test_time_based_split_rejects_missing_timestamp_column():
    df = pd.DataFrame({"aqi": range(20), "aqi_target": range(20)})
    with pytest.raises(TrainingError):
        time_based_train_test_split(df)


def test_time_based_split_rejects_too_few_rows():
    df = _make_df(5)
    with pytest.raises(TrainingError):
        time_based_train_test_split(df)


def test_time_based_train_val_test_split_produces_expected_sizes_and_order():
    df = _make_df(200)
    train_df, val_df, test_df = time_based_train_val_test_split(
        df, train_size=0.7, val_size=0.15, test_size=0.15
    )

    assert len(train_df) == 140
    assert len(val_df) == 30
    assert len(test_df) == 30
    assert train_df["collection_timestamp"].max() < val_df["collection_timestamp"].min()
    assert val_df["collection_timestamp"].max() < test_df["collection_timestamp"].min()


def test_time_based_train_val_test_split_rejects_bad_sizes():
    df = _make_df(50)
    with pytest.raises(TrainingError):
        time_based_train_val_test_split(df, train_size=0.5, val_size=0.5, test_size=0.5)
    with pytest.raises(TrainingError):
        time_based_train_val_test_split(df, train_size=0.8, val_size=0.15, test_size=0.2)


def test_make_time_series_cv_applies_gap_and_rejects_bad_args():
    cv = make_time_series_cv(n_splits=3, gap=72)
    assert cv.get_n_splits() == 3
    assert cv.gap == 72

    with pytest.raises(TrainingError):
        make_time_series_cv(n_splits=1, gap=72)
    with pytest.raises(TrainingError):
        make_time_series_cv(n_splits=3, gap=-1)


def test_build_xy_drops_incomplete_rows():
    df = _make_df(20)
    df.loc[0, "aqi"] = np.nan
    df.loc[1, "aqi_target"] = np.nan

    feature_cols = ["aqi", "pm2_5", "temperature"]
    X, y = build_xy(df, feature_cols, "aqi_target")

    assert len(X) == 18
    assert len(y) == 18
    assert list(X.columns) == feature_cols


def test_build_xy_raises_when_target_missing():
    df = _make_df(10)
    with pytest.raises(TrainingError):
        build_xy(df, ["aqi"], target_col="does_not_exist")


def test_build_xy_raises_when_no_complete_rows_remain():
    df = _make_df(5)
    df["aqi"] = np.nan
    with pytest.raises(TrainingError):
        build_xy(df, ["aqi"], "aqi_target")
