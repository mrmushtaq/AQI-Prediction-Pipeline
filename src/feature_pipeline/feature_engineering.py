"""
Phase 2 — Feature Engineering for the AQI predictor.

Reads the raw processed dataset produced by the Phase 1 live/historical
pipelines (see :mod:`~src.feature_pipeline.feature_pipeline`) and derives the
model inputs and target needed for training:

    * Time features -- hour, day-of-week, day-of-month, month, cyclically
      encoded with sin/cos so that, e.g., hour 23 and hour 0 are seen as
      close together rather than far apart.
    * Lag features -- the previous 1-2 readings for AQI and key pollutants,
      giving the model recent history to work from.
    * Rolling averages -- rolling mean/std over recent windows (default:
      24h and 48h, i.e. 8 and 16 steps at the pipeline's 3-hour interval)
      for AQI and key pollutants, smoothing out single-reading noise.
    * A derived "pollution index" -- a simple composite score across
      whichever pollutant concentrations are available for a row,
      normalized against reference thresholds. This is a project-specific
      feature distinct from the official EPA `aqi` column already present
      in the dataset (see :func:`_add_pollution_index` docstring for the
      exact method and its limitations).
    * AQI change rate -- first difference from the previous reading.
    * The forecasting target: AQI ``HORIZON_STEPS`` intervals ahead
      (default: 3 days, computed from the pipeline's actual
      ``HISTORICAL_INTERVAL_HOURS`` collection interval).

All derived/lag/rolling/target columns are computed **per city**, using a
sorted, deduplicated timestamp index, so that rows from different cities (or
rows appended out of chronological order across separate backfill runs)
never leak into each other's rolling calculations.

Usage::

    python -m src.feature_pipeline.feature_engineering
    python -m src.feature_pipeline.feature_engineering --input data/processed/aqi_dataset_historical.csv --output data/processed/feature_df.csv
    python -m src.feature_pipeline.feature_engineering --horizon-steps 8   # 1 day ahead instead of 3
"""

from __future__ import annotations

from src.feature_pipeline.historical_backfill import HISTORICAL_INTERVAL_HOURS
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from configs.config import load_settings
from src.common.exceptions import PipelineError
from src.common.logger import get_logger

logger = get_logger(__name__)

# The pipeline collects data at a fixed 3-hour interval (see
# HISTORICAL_INTERVAL_HOURS in historical_backfill.py). 24 steps of 3 hours
# 72 hours = 3 days, matching the project's "predict AQI 3 days ahead"
# requirement. Computed dynamically from HISTORICAL_INTERVAL_HOURS so this
# stays correct if the collection interval ever changes.
_HORIZON_HOURS = 72
DEFAULT_HORIZON_STEPS = max(1, _HORIZON_HOURS // HISTORICAL_INTERVAL_HOURS)

# Pollutant/weather columns we compute a "change rate" (first difference)
# for, in addition to AQI itself. Kept to fields that are meaningful as a
# rate of change and commonly available across sources.
CHANGE_RATE_COLUMNS = ["aqi", "pm2_5", "pm10", "temperature", "humidity"]

# How many previous readings to include as explicit lag features.
LAG_STEPS = (1, 2)

# Rolling-window sizes, in number of intervals. Computed dynamically from
# HISTORICAL_INTERVAL_HOURS (the pipeline's actual collection interval) so
# these always represent true 24h/48h windows regardless of what that
# interval is currently set to.
_ROLLING_WINDOW_HOURS = (24, 48)
ROLLING_WINDOWS_STEPS = tuple(
    max(1, hours // HISTORICAL_INTERVAL_HOURS) for hours in _ROLLING_WINDOW_HOURS
)
ROLLING_WINDOW_LABELS = dict(zip(ROLLING_WINDOWS_STEPS, [f"{h}h" for h in _ROLLING_WINDOW_HOURS]))

# Columns to compute rolling mean/std for.
ROLLING_COLUMNS = ["aqi", "pm2_5", "pm10"]

# Reference concentration thresholds used to build the composite pollution
# index (see _add_pollution_index). These correspond roughly to the EPA
# "Moderate" (AQI ~100) breakpoint upper bound for each pollutant, so a
# pollutant reading at its reference value contributes ~1.0 (on a 0-1+
# scale) to the index. This is a simple, transparent normalization choice
# -- not an official EPA methodology -- distinct from the `aqi` column
# already present in the dataset (which *is* computed via the official EPA
# breakpoint formula, see fetch_air_quality.py).
POLLUTION_INDEX_REFERENCE = {
    "pm2_5": 35.4,   # µg/m³
    "pm10": 154.0,   # µg/m³
    "co": 9.4,       # ppm
    "so2": 75.0,     # ppb
    "no2": 100.0,    # ppb
    "o3": 70.0,      # ppb
}


def _load_dataset(input_path: Path) -> pd.DataFrame:
    """Load the raw processed CSV and do basic cleanup.

    Args:
        input_path: Path to the processed historical (or live) CSV.

    Returns:
        A DataFrame parsed, deduplicated, and sorted by
        ``(city, collection_timestamp)``.

    Raises:
        PipelineError: If the file doesn't exist or is empty.
    """
    if not input_path.exists():
        raise PipelineError(f"Input dataset not found: {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise PipelineError(f"Input dataset is empty: {input_path}")

    if "collection_timestamp" not in df.columns or "city" not in df.columns:
        raise PipelineError(
            "Input dataset is missing required columns 'collection_timestamp' "
            "and/or 'city'."
        )

    df["collection_timestamp"] = pd.to_datetime(
        df["collection_timestamp"], utc=True, errors="coerce"
    )
    before = len(df)
    df = df.dropna(subset=["collection_timestamp"])
    if len(df) < before:
        logger.warning(
            "Dropped %d rows with unparseable collection_timestamp",
            before - len(df),
        )

    # Rows are appended across many separate pipeline runs (live + several
    # historical backfill invocations), so duplicates and out-of-order rows
    # are expected. Dedup on the natural key and sort chronologically
    # within each city before any rolling/lag computation.
    before = len(df)
    df = df.drop_duplicates(subset=["city", "collection_timestamp"], keep="last")
    if len(df) < before:
        logger.info(
            "Dropped %d duplicate (city, collection_timestamp) rows",
            before - len(df),
        )

    df = df.sort_values(["city", "collection_timestamp"]).reset_index(drop=True)
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add raw and cyclically-encoded time-based features.

    Adds ``hour``, ``day_of_week``, ``day_of_month``, ``month`` (raw
    integers, useful for tree-based models) plus ``*_sin`` / ``*_cos``
    pairs (useful for linear models and neural nets, so e.g. hour 23 and
    hour 0 are close in feature space instead of maximally far apart).

    Args:
        df: DataFrame with a ``collection_timestamp`` column.

    Returns:
        The same DataFrame with time-based feature columns appended.
    """
    ts = df["collection_timestamp"]

    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek  # Monday=0 .. Sunday=6
    df["day_of_month"] = ts.dt.day
    df["month"] = ts.dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def _add_change_rate_and_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-city change-rate and lag features.

    For each column in ``CHANGE_RATE_COLUMNS`` present in the dataset, adds:
        * ``<col>_change_rate`` -- first difference from the previous
          reading for that city (``NaN`` for each city's first row).
        * ``<col>_lag_<n>`` for each ``n`` in ``LAG_STEPS`` -- the value
          from ``n`` readings ago for that city.

    Grouping by ``city`` before computing ``diff()``/``shift()`` ensures a
    change-rate or lag value is never computed across a city boundary.

    Args:
        df: DataFrame sorted by ``(city, collection_timestamp)``.

    Returns:
        The same DataFrame with change-rate and lag columns appended.
    """
    grouped = df.groupby("city", sort=False)

    for col in CHANGE_RATE_COLUMNS:
        if col not in df.columns:
            logger.warning(
                "Column '%s' not found in dataset; skipping its change-rate feature",
                col,
            )
            continue
        df[f"{col}_change_rate"] = grouped[col].diff()

        for lag in LAG_STEPS:
            df[f"{col}_lag_{lag}"] = grouped[col].shift(lag)

    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-city rolling mean/std features.

    For each column in ``ROLLING_COLUMNS`` present in the dataset, adds
    ``<col>_rolling_mean_<label>`` and ``<col>_rolling_std_<label>`` for
    each window in ``ROLLING_WINDOWS_STEPS`` (e.g. ``aqi_rolling_mean_24h``).

    Computed with ``min_periods=1`` per city so a city's early rows get a
    partial-window average rather than ``NaN`` outright -- rolling std is
    still ``NaN`` for a city's very first row (undefined for a single
    point), which is expected and handled by the incomplete-row drop step.

    Grouping by ``city`` before rolling ensures a rolling value is never
    computed across a city boundary.

    Args:
        df: DataFrame sorted by ``(city, collection_timestamp)``.

    Returns:
        The same DataFrame with rolling feature columns appended.
    """
    grouped = df.groupby("city", sort=False)

    for col in ROLLING_COLUMNS:
        if col not in df.columns:
            logger.warning(
                "Column '%s' not found in dataset; skipping its rolling features",
                col,
            )
            continue

        for window in ROLLING_WINDOWS_STEPS:
            label = ROLLING_WINDOW_LABELS.get(window, f"{window}steps")
            df[f"{col}_rolling_mean_{label}"] = grouped[col].transform(
                lambda s, w=window: s.rolling(window=w, min_periods=1).mean()
            )
            df[f"{col}_rolling_std_{label}"] = grouped[col].transform(
                lambda s, w=window: s.rolling(window=w, min_periods=2).std()
            )

    return df


def _add_pollution_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add a simple composite ``pollution_index`` feature.

    This is **not** the official EPA AQI (that's already present in the
    ``aqi`` column, computed via the EPA breakpoint formula in
    ``fetch_air_quality.py``). It's a simpler, transparent composite meant
    to give models an additional signal that blends whichever pollutant
    concentrations happen to be available for a row, since not every
    monitoring station/source reports every pollutant.

    Method: for each pollutant present in ``POLLUTION_INDEX_REFERENCE`` and
    non-null for a given row, divide its concentration by its reference
    threshold (roughly the EPA "Moderate" breakpoint upper bound for that
    pollutant), then average the available ratios. A value of ~1.0 means
    pollutants are, on average, around the "Moderate" threshold; values
    below/above scale accordingly. Rows with no available pollutant data
    get ``NaN``.

    Limitations: this is a simple mean-of-ratios composite, not a
    physiologically-weighted index -- unlike the official AQI (which takes
    the *max* sub-index, since health risk is driven by the worst
    pollutant), this index can under-represent a single severe pollutant if
    other pollutants are low. It's intended as a supplementary model
    feature, not a replacement for the ``aqi`` column.

    Args:
        df: DataFrame containing zero or more of the pollutant columns in
            ``POLLUTION_INDEX_REFERENCE``.

    Returns:
        The same DataFrame with a ``pollution_index`` column appended.
    """
    # A column can exist in the dataset yet be entirely null (e.g. pm10/co/
    # so2/no2/o3 for a monitoring source that only measures AQI + PM2.5 --
    # see README Section 10). Checking "in df.columns" alone would still
    # count those as "available", making the composite silently collapse
    # into a scaled copy of whichever single pollutant actually has data
    # (perfectly correlated with it -- no new information for a model).
    # Only treat a pollutant as available if it has at least some real
    # (non-null) data somewhere in the dataset.
    available_cols = [
        c for c in POLLUTION_INDEX_REFERENCE
        if c in df.columns and df[c].notna().any()
    ]

    if len(available_cols) < 2:
        logger.warning(
            "Fewer than 2 pollutant columns with real data available "
            "(found: %s); pollution_index would just duplicate a single "
            "existing column, so it will be set to NaN and should be "
            "excluded from model training.",
            available_cols,
        )
        df["pollution_index"] = np.nan
        return df

    ratios = pd.DataFrame(
        {col: df[col] / POLLUTION_INDEX_REFERENCE[col] for col in available_cols}
    )
    df["pollution_index"] = ratios.mean(axis=1, skipna=True)
    return df


def _add_target(df: pd.DataFrame, horizon_steps: int) -> pd.DataFrame:
    """Add the forecasting target column: AQI ``horizon_steps`` ahead.

    Computed per-city via a negative ``shift()``, so the target for each
    row is the AQI value ``horizon_steps`` readings later **for that same
    city** -- never bleeding across a city boundary.

    Args:
        df: DataFrame sorted by ``(city, collection_timestamp)``.
        horizon_steps: How many rows ahead (at the pipeline's fixed
            interval) the target should look. Default corresponds to 3
            days at the pipeline's actual ``HISTORICAL_INTERVAL_HOURS``
            collection interval.

    Returns:
        The same DataFrame with an ``aqi_target`` column appended.
    """
    if "aqi" not in df.columns:
        raise PipelineError("Dataset is missing the 'aqi' column required to build the target.")

    grouped = df.groupby("city", sort=False)
    df["aqi_target"] = grouped["aqi"].shift(-horizon_steps)
    return df


def build_features(
    input_path: Path,
    horizon_steps: int = DEFAULT_HORIZON_STEPS,
    drop_incomplete_rows: bool = True,
) -> pd.DataFrame:
    """Run the full Phase 2 feature-engineering workflow.

    Workflow:
        1. Load, deduplicate, and sort the raw processed dataset.
        2. Add cyclical time-based features.
        3. Add per-city change-rate and lag features.
        4. Add per-city rolling mean/std features.
        5. Add the composite pollution index.
        6. Add the ``horizon_steps``-ahead AQI target.
        7. Optionally drop rows that don't have a full feature/target set
           (i.e. each city's first few rows, which lack lag/rolling-std
           history, and each city's last ``horizon_steps`` rows, which
           lack a future target to predict).

    Args:
        input_path: Path to the raw processed CSV
            (e.g. ``data/processed/aqi_dataset_historical.csv``).
        horizon_steps: How many intervals ahead to forecast.
        drop_incomplete_rows: If ``True`` (default), drop rows with any
            ``NaN`` in the lag/change-rate/rolling-std/target columns,
            leaving a dataset that's immediately ready for model training.
            If ``False``, incomplete rows are kept with ``NaN`` values so
            the caller can decide how to handle them (e.g. imputation).

    Returns:
        The fully feature-engineered DataFrame.

    Raises:
        PipelineError: If the input dataset is missing, empty, or malformed.
    """
    logger.info("Feature engineering: loading dataset from %s", input_path)
    df = _load_dataset(input_path)
    logger.info("Feature engineering: loaded %d rows across %d city(ies)",
                len(df), df["city"].nunique())

    df = _add_time_features(df)
    df = _add_change_rate_and_lag_features(df)
    df = _add_rolling_features(df)
    df = _add_pollution_index(df)
    df = _add_target(df, horizon_steps)

    if drop_incomplete_rows:
        feature_cols = [
            c for c in df.columns
            if c.endswith("_change_rate") or "_lag_" in c or "_rolling_std_" in c
        ]
        candidate_required_cols = feature_cols + ["aqi_target"]

        # A derived column (e.g. pm10_lag_1) can be NaN for two different
        # reasons: (a) not enough history yet for *this* row (the normal,
        # expected case at the start/end of the series -- these rows
        # should be dropped), or (b) the underlying source column (e.g.
        # pm10) has NO real data anywhere in the dataset, because the
        # station/source simply doesn't measure that pollutant (see
        # Section 10 of the README -- pm10/co/so2/no2/o3 are allowed to be
        # null). In case (b), requiring that derived column to be non-null
        # would drop every row in the dataset, which is never the intent.
        # So: only enforce completeness on derived columns that have at
        # least *some* real (non-null) values somewhere in the dataset.
        required_cols = []
        skipped_cols = []
        for col in candidate_required_cols:
            if df[col].notna().any():
                required_cols.append(col)
            else:
                skipped_cols.append(col)

        if skipped_cols:
            logger.warning(
                "Feature engineering: %d derived column(s) are entirely NaN "
                "(underlying source data not available for this location/"
                "source) and were excluded from the completeness check: %s",
                len(skipped_cols),
                ", ".join(skipped_cols),
            )

        before = len(df)
        df = df.dropna(subset=required_cols)
        logger.info(
            "Feature engineering: dropped %d rows lacking full lag/rolling/target "
            "history (kept %d rows)",
            before - len(df),
            len(df),
        )

    df = df.reset_index(drop=True)
    return df


def main() -> int:
    """CLI entry point.

    Usage::

        python -m src.feature_pipeline.feature_engineering
        python -m src.feature_pipeline.feature_engineering --input <path> --output <path>
        python -m src.feature_pipeline.feature_engineering --horizon-steps 8
        python -m src.feature_pipeline.feature_engineering --keep-incomplete-rows

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    parser = argparse.ArgumentParser(
        description="Phase 2: build training-ready features and target from "
        "the processed AQI dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to the raw processed CSV. Defaults to "
        "<processed_data_dir>/aqi_dataset_historical.csv from settings.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write the feature-engineered CSV. Defaults to "
        "<processed_data_dir>/feature_df.csv.",
    )
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=DEFAULT_HORIZON_STEPS,
        help=f"Number of intervals ahead to forecast (default: {DEFAULT_HORIZON_STEPS}, "
        "i.e. 3 days at a 3-hour interval).",
    )
    parser.add_argument(
        "--keep-incomplete-rows",
        action="store_true",
        help="Keep rows with NaN lag/rolling/change-rate/target values instead "
        "of dropping them.",
    )
    args = parser.parse_args()

    settings = load_settings()

    input_path = Path(args.input) if args.input else settings.processed_data_dir / "aqi_dataset_historical.csv"
    output_path = Path(args.output) if args.output else settings.processed_data_dir / "feature_df.csv"

    try:
        df = build_features(
            input_path=input_path,
            horizon_steps=args.horizon_steps,
            drop_incomplete_rows=not args.keep_incomplete_rows,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(
            "Feature engineering completed: %d rows, %d columns -> %s",
            len(df),
            len(df.columns),
            output_path,
        )
        print(f"Feature engineering completed: {len(df)} rows written to {output_path}")
        return 0
    except PipelineError as exc:
        logger.error("Feature engineering failed: %s", exc)
        print(f"Feature engineering failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Feature engineering failed with an unexpected error")
        print(f"Feature engineering failed with an unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())