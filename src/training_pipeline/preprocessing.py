"""
Preprocessing for the model training pipeline (Phase 5).

Three responsibilities:

1. **Fixed feature whitelist** -- `FEATURE_COLUMNS` explicitly defines every
   column used for model training, so future feature additions (or metadata
   columns) never accidentally enter the model. This replaces the old
   "every numeric, non-null column" heuristic. Constant columns such as
   `city`/`latitude`/`longitude`/`weather_source` are simply not in the
   whitelist (they carry no signal for a single-city dataset), and
   `feels_like` is retained because the target models are tree-based and
   handle correlated features well.

2. **Time-based train/validation/test split** -- this is time-series data
   with lag and rolling features computed from each row's temporal
   neighbors. A random shuffle-based split (e.g. plain
   `sklearn.model_selection.train_test_split` with `shuffle=True`) would let
   test-set information leak into training-set rolling/lag features and
   vice versa, producing misleadingly optimistic validation metrics.
   Splitting chronologically (earliest rows train, latest rows validation,
   newest rows test) avoids this and matches the real deployment scenario:
   predicting the future from the past. Data is never shuffled.

3. **Time-series cross-validation** -- for hyperparameter tuning, folds are
   created chronologically with a `gap` equal to the forecast horizon (72
   samples = 72 hours) so that no training sample's target overlaps the
   next fold's feature window.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.common.exceptions import TrainingError
from src.common.logger import get_logger

logger = get_logger(__name__)

TARGET_COLUMN = "aqi_target"

# The complete, curated list of model features. Everything used for training
# must be listed here; nothing else is considered. The 51 features below come
# directly from the EDA's recommended final feature set:
#
#   Weather variables        : temperature, feels_like, humidity, pressure,
#                              wind_speed, wind_direction, clouds
#   AQI history              : pm2_5, aqi
#   Cyclical time features   : hour_sin, hour_cos, day_of_week_sin,
#                              day_of_week_cos, month_sin, month_cos
#   Short lag features       : aqi_lag_1/2, pm2_5_lag_1/2,
#                              temperature_lag_1/2, humidity_lag_1/2
#   Seasonal lag features    : aqi/pm2_5/temperature/humidity lag_24/48/72/168
#                              (same hour 1/2/3/7 days ago) -- the dominant
#                              predictors for a 24-72h AQI forecast
#   Change-rate features     : aqi_change_rate, pm2_5_change_rate,
#                              temperature_change_rate, humidity_change_rate
#   Rolling statistics       : aqi/pm2_5 rolling mean/std (24h and 48h)
#
# `aqi` is intentionally kept last so the persistence baseline (which reads
# the current-AQI column) has a stable position. Constant / metadata columns
# (city, latitude, longitude, weather_source, timestamp, collection_timestamp,
# weather, aqi_source) and the raw calendar integers (hour, day_of_week,
# month, day_of_month) are deliberately excluded -- cyclical encodings of the
# latter are used instead.
FEATURE_COLUMNS: Sequence[str] = (
    # Weather variables
    "temperature",
    "feels_like",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_direction",
    "clouds",
    # AQI history
    "pm2_5",
    # Cyclical time features
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    # Short lag features
    "aqi_lag_1",
    "aqi_lag_2",
    "pm2_5_lag_1",
    "pm2_5_lag_2",
    "temperature_lag_1",
    "temperature_lag_2",
    "humidity_lag_1",
    "humidity_lag_2",
    # Seasonal lag features (same hour 1/2/3/7 days ago)
    "aqi_lag_24",
    "aqi_lag_48",
    "aqi_lag_72",
    "aqi_lag_168",
    "pm2_5_lag_24",
    "pm2_5_lag_48",
    "pm2_5_lag_72",
    "pm2_5_lag_168",
    "temperature_lag_24",
    "temperature_lag_48",
    "temperature_lag_72",
    "temperature_lag_168",
    "humidity_lag_24",
    "humidity_lag_48",
    "humidity_lag_72",
    "humidity_lag_168",
    # Change-rate features
    "aqi_change_rate",
    "pm2_5_change_rate",
    "temperature_change_rate",
    "humidity_change_rate",
    # Rolling statistics
    "aqi_rolling_mean_24h",
    "aqi_rolling_std_24h",
    "aqi_rolling_mean_48h",
    "aqi_rolling_std_48h",
    "pm2_5_rolling_mean_24h",
    "pm2_5_rolling_std_24h",
    "pm2_5_rolling_mean_48h",
    "pm2_5_rolling_std_48h",
    # Current AQI (used by the persistence baseline)
    "aqi",
)


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the subset of the fixed `FEATURE_COLUMNS` whitelist usable for
    training.

    A whitelist column is usable if it is present in `df` and has at least
    one non-null value. Columns that are entirely null (e.g. `pm10`-derived
    features for a city whose station doesn't measure that pollutant) are
    excluded automatically, matching the behaviour of
    `feature_engineering.py`'s completeness check.

    Args:
        df: The loaded feature DataFrame (from `data_loader.load_features`).

    Returns:
        The ordered list of whitelist columns to use as model features
        (subset of `FEATURE_COLUMNS`).

    Raises:
        TrainingError: If none of the whitelist columns are usable.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        logger.warning(
            "Feature whitelist column(s) not present in the dataset (excluded): %s",
            missing,
        )

    candidate_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    feature_cols = []
    dropped_all_null = []
    for col in candidate_cols:
        if df[col].notna().any():
            feature_cols.append(col)
        else:
            dropped_all_null.append(col)

    if dropped_all_null:
        logger.warning(
            "Excluding %d all-null whitelist column(s) from model features: %s",
            len(dropped_all_null),
            dropped_all_null,
        )

    if not feature_cols:
        raise TrainingError(
            "No usable feature columns found in the dataset (checked the "
            f"FEATURE_COLUMNS whitelist of {len(FEATURE_COLUMNS)} columns)."
        )

    logger.info("Selected %d feature columns for training", len(feature_cols))
    return feature_cols


def time_based_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-series DataFrame chronologically into train/test sets.

    The DataFrame is sorted by `collection_timestamp` first, then the first
    `1 - test_size` fraction of rows becomes the training set and the
    remaining, most-recent `test_size` fraction becomes the test set. No
    shuffling is performed -- see the module docstring for why that matters
    for this dataset.

    Args:
        df: Feature DataFrame containing a `collection_timestamp` column.
        test_size: Fraction of rows (by time) to hold out for testing,
            e.g. `0.2` for the most recent 20%.

    Returns:
        `(train_df, test_df)`, both re-indexed from 0.

    Raises:
        TrainingError: If `test_size` is not in `(0, 1)`, or the DataFrame is
            missing `collection_timestamp`, or too small to split.
    """
    if not (0.0 < test_size < 1.0):
        raise TrainingError(f"test_size must be in (0, 1), got: {test_size}")

    _require_timestamp_column(df)
    _require_minimum_rows(df)

    sorted_df = df.sort_values("collection_timestamp").reset_index(drop=True)
    split_idx = int(len(sorted_df) * (1 - test_size))

    train_df = sorted_df.iloc[:split_idx].reset_index(drop=True)
    test_df = sorted_df.iloc[split_idx:].reset_index(drop=True)

    logger.info(
        "Time-based split: %d train rows (%s to %s), %d test rows (%s to %s)",
        len(train_df),
        train_df["collection_timestamp"].min(),
        train_df["collection_timestamp"].max(),
        len(test_df),
        test_df["collection_timestamp"].min(),
        test_df["collection_timestamp"].max(),
    )
    return train_df, test_df


def time_based_train_val_test_split(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-series DataFrame chronologically into train/validation/
    test sets (default 70% / 15% / 15%).

    The DataFrame is sorted by `collection_timestamp` first, then split into
    three contiguous, non-overlapping chronological segments: the earliest
    `train_size` fraction trains, the next `val_size` tunes/selects models,
    and the most-recent `test_size` fraction is the final, untouched
    hold-out. Data is never shuffled.

    Args:
        df: Feature DataFrame containing a `collection_timestamp` column.
        train_size: Fraction of rows (by time) for training.
        val_size: Fraction of rows (by time) for validation.
        test_size: Fraction of rows (by time) for final testing.

    Returns:
        `(train_df, val_df, test_df)`, each re-indexed from 0, in strictly
        increasing time order.

    Raises:
        TrainingError: If any size is not in `(0, 1)`, the sizes don't sum
            to ~1, the DataFrame is missing `collection_timestamp`, or it is
            too small to split.
    """
    sizes = {"train_size": train_size, "val_size": val_size, "test_size": test_size}
    for name, value in sizes.items():
        if not (0.0 < value < 1.0):
            raise TrainingError(f"{name} must be in (0, 1), got: {value}")

    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise TrainingError(
            "train_size + val_size + test_size must sum to 1.0, got "
            f"{train_size} + {val_size} + {test_size} = {train_size + val_size + test_size}"
        )

    _require_timestamp_column(df)
    _require_minimum_rows(df)

    sorted_df = df.sort_values("collection_timestamp").reset_index(drop=True)
    n = len(sorted_df)

    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    train_df = sorted_df.iloc[:train_end].reset_index(drop=True)
    val_df = sorted_df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = sorted_df.iloc[val_end:].reset_index(drop=True)

    logger.info(
        "Time-based split: %d train / %d val / %d test rows "
        "(train %s to %s | val %s to %s | test %s to %s)",
        len(train_df),
        len(val_df),
        len(test_df),
        train_df["collection_timestamp"].min(),
        train_df["collection_timestamp"].max(),
        val_df["collection_timestamp"].min(),
        val_df["collection_timestamp"].max(),
        test_df["collection_timestamp"].min(),
        test_df["collection_timestamp"].max(),
    )
    return train_df, val_df, test_df


def make_time_series_cv(n_splits: int = 5, gap: int = 72) -> TimeSeriesSplit:
    """Build a chronological cross-validation splitter for tuning.

    Uses :class:`sklearn.model_selection.TimeSeriesSplit` with a ``gap``
    equal to the forecast horizon (72 samples, i.e. 72 hours of hourly
    data). The gap excludes ``gap`` samples from the end of each training
    fold before its test fold, so no training sample's target falls inside
    the following fold's feature window -- preventing temporal leakage
    during hyperparameter tuning.

    Args:
        n_splits: Number of chronological folds.
        gap: Number of samples to leave between each train and test fold
            (default 72 = the 72-hour forecast horizon).

    Returns:
        A configured `TimeSeriesSplit`.

    Raises:
        TrainingError: If `n_splits` < 2 or `gap` < 0.
    """
    if n_splits < 2:
        raise TrainingError(f"n_splits must be >= 2 for time-series CV, got: {n_splits}")
    if gap < 0:
        raise TrainingError(f"gap must be >= 0 for time-series CV, got: {gap}")

    logger.info("Time-series CV: %d splits with a %d-sample gap (forecast horizon)", n_splits, gap)
    return TimeSeriesSplit(n_splits=n_splits, gap=gap)


def build_xy(
    df: pd.DataFrame, feature_cols: Sequence[str], target_col: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract the feature matrix `X` and target vector `y` from a DataFrame.

    Rows with a null value in any feature column or the target are dropped
    (defensively -- `feature_engineering.py`'s `drop_incomplete_rows` step
    should already guarantee this, but training code should not silently
    assume an upstream invariant holds).

    Args:
        df: A train, validation, or test DataFrame slice.
        feature_cols: Column names to use as features (from
            `select_feature_columns`).
        target_col: Name of the target column.

    Returns:
        `(X, y)` with any incomplete rows dropped.

    Raises:
        TrainingError: If `target_col` is missing, or no rows remain after
            dropping incomplete ones.
    """
    if target_col not in df.columns:
        raise TrainingError(f"Target column '{target_col}' not found in DataFrame.")

    subset = df[list(feature_cols) + [target_col]]
    before = len(subset)
    subset = subset.dropna()
    dropped = before - len(subset)
    if dropped:
        logger.warning("Dropped %d row(s) with missing feature/target values", dropped)

    if subset.empty:
        raise TrainingError("No complete rows remain after dropping missing feature/target values.")

    X = subset[list(feature_cols)].reset_index(drop=True)
    y = subset[target_col].reset_index(drop=True)
    return X, y


def _require_timestamp_column(df: pd.DataFrame) -> None:
    if "collection_timestamp" not in df.columns:
        raise TrainingError(
            "DataFrame is missing the 'collection_timestamp' column required for a time-based split."
        )


def _require_minimum_rows(df: pd.DataFrame, minimum: int = 10) -> None:
    if len(df) < minimum:
        raise TrainingError(
            f"Not enough rows ({len(df)}) to perform a meaningful time-based train/test split."
        )
