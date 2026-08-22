"""
Hopsworks Feature Store integration (Phase 3).

Connects the Phase 2 engineered feature dataset (`data/processed/feature_df.csv`)
to a Hopsworks Feature Store:

    CSV (feature_df.csv)
        -> login()
        -> get_or_create_feature_group()
        -> insert_features()
        -> read_features()   (confirms the round trip)

This module is purely additive: it does not modify the Phase 1 ingestion
pipelines or the Phase 2 feature-engineering script. It reads their output
(`feature_df.csv`) and pushes it one layer further downstream.

Run end-to-end with:
    python -m src.feature_pipeline.feature_store
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from configs.config import PROJECT_ROOT, HopsworksSettings, load_hopsworks_settings
from src.common.exceptions import FeatureStoreError
from src.common.logger import get_logger

logger = get_logger(__name__)

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 3
FEATURE_GROUP_DESCRIPTION = (
    "Engineered AQI + weather features (Phase 2 output): time/cyclic features, "
    "lag features, rolling stats, pollution index, and the aqi_target label. "
    "v2: supersedes v1 -- fixes 'feels_like' being null for all historical "
    "records (Open-Meteo Archive wasn't requesting apparent_temperature), and "
    "fixes rolling-window/forecast-horizon sizes that were computed assuming "
    "a 3-hour collection interval when the pipeline actually runs hourly "
    "(rolling windows were silently 8h/16h instead of the labeled 24h/48h, "
    "and aqi_target was 24h ahead instead of the intended 72h/3-day horizon)."
)

# `collection_timestamp` is the pipeline's normalized, idempotency-safe
# timestamp (see the Phase 1 ingestion pipeline's timestamp-normalization
# logic) -- it, together with `city`, uniquely identifies one observation.
PRIMARY_KEY = ["city", "collection_timestamp"]
EVENT_TIME_COLUMN = "collection_timestamp"

DEFAULT_FEATURE_DF_PATH = PROJECT_ROOT / "data" / "processed" / "feature_df.csv"

# Default host for Hopsworks Serverless (https://app.hopsworks.ai) accounts,
# used only by the lightweight `hsfs`-only fallback path in `login()` below.
# Override with the HOPSWORKS_HOST env var for on-prem/self-managed clusters.
DEFAULT_HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai")


def login(settings: HopsworksSettings | None = None) -> Any:
    """Log in to Hopsworks and return the project's Feature Store handle.

    Tries the full `hopsworks` meta-package first (`hopsworks.login()`),
    which is the simplest, officially recommended path. If it isn't
    installed, falls back to the lighter `hsfs`-only client
    (`pip install hsfs[python]`) via `hsfs.connection()`.

    The fallback exists because `hopsworks`'s full install pulls in a Java
    KeyStore dependency chain (`pyjks` -> `twofish`, used only for on-prem
    Hive/Kafka storage connectors this project doesn't need) that requires a
    C compiler to build on Windows. `hsfs[python]` provides everything
    needed for plain REST-based feature group create/insert/read without
    that dependency.

    Args:
        settings: Optional pre-loaded `HopsworksSettings`. If omitted,
            credentials are loaded from the environment
            (`HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT_NAME`).

    Returns:
        A Hopsworks `FeatureStore` handle, ready for
        `get_or_create_feature_group()` / `get_feature_group()`.

    Raises:
        FeatureStoreError: If neither `hopsworks` nor `hsfs` is installed, or
            login/connection fails (invalid API key, wrong project name,
            network issue).
    """
    settings = settings or load_hopsworks_settings()

    try:
        import hopsworks
    except ImportError:
        hopsworks = None

    if hopsworks is not None:
        logger.info(
            "Logging in to Hopsworks project=%s (via 'hopsworks' package)",
            settings.project_name,
        )
        try:
            project = hopsworks.login(
                project=settings.project_name,
                api_key_value=settings.api_key,
            )
        except Exception as exc:
            raise FeatureStoreError(f"Hopsworks login failed: {exc}") from exc

        logger.info("Hopsworks login successful (project=%s)", settings.project_name)
        try:
            return project.get_feature_store()
        except Exception as exc:
            raise FeatureStoreError(f"Failed to get Hopsworks feature store handle: {exc}") from exc

    # Fallback: lightweight `hsfs`-only client. Same effective result
    # (a FeatureStore handle) via a direct connection instead of the
    # `hopsworks` meta-package's `login()` convenience wrapper.
    logger.info(
        "'hopsworks' package not installed; falling back to 'hsfs' client "
        "(host=%s, project=%s)",
        DEFAULT_HOPSWORKS_HOST,
        settings.project_name,
    )
    try:
        import hsfs
    except ImportError as exc:
        raise FeatureStoreError(
            "Neither 'hopsworks' nor 'hsfs' is installed. Run: "
            "pip install hsfs[python]  (or: pip install hopsworks)"
        ) from exc

    try:
        connection = hsfs.connection(
            host=DEFAULT_HOPSWORKS_HOST,
            project=settings.project_name,
            api_key_value=settings.api_key,
            engine="python",
        )
    except Exception as exc:
        raise FeatureStoreError(f"hsfs connection failed: {exc}") from exc

    logger.info("hsfs connection successful (project=%s)", settings.project_name)
    try:
        return connection.get_feature_store()
    except Exception as exc:
        raise FeatureStoreError(f"Failed to get feature store handle via hsfs: {exc}") from exc


def prepare_dataframe_for_feature_store(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the Phase 2 feature dataframe for Hopsworks ingestion.

    Two adjustments are made (the source CSV on disk is never modified --
    this operates on an in-memory copy):

    1. `collection_timestamp` is converted from an ISO-8601 string into a
       proper timezone-naive UTC `datetime64` column. Hopsworks requires the
       declared `event_time` column to actually be a timestamp dtype.
    2. The `timestamp` column is dropped if present: by design it is always
       identical to `collection_timestamp` (the Phase 1 pipeline normalizes
       `timestamp` to equal `collection_timestamp` for backward
       compatibility -- see the ingestion pipeline's timestamp-normalization
       logic), so keeping both would put a fully redundant column in the
       feature store.

    Args:
        df: Raw dataframe as loaded from `feature_df.csv`.

    Returns:
        A new dataframe, ready to pass to `insert_features()`.

    Raises:
        FeatureStoreError: If the expected `collection_timestamp` column is
            missing, or its values cannot be parsed as timestamps.
    """
    df = df.copy()

    if "collection_timestamp" not in df.columns:
        raise FeatureStoreError(
            "Expected 'collection_timestamp' column not found in feature "
            f"dataframe (columns present: {list(df.columns)})."
        )

    try:
        df["collection_timestamp"] = (
            pd.to_datetime(df["collection_timestamp"], utc=True).dt.tz_localize(None)
        )
    except (ValueError, TypeError) as exc:
        raise FeatureStoreError(
            f"Failed to parse 'collection_timestamp' as datetimes: {exc}"
        ) from exc

    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    return df


def get_or_create_feature_group(fs: Any, description: str = FEATURE_GROUP_DESCRIPTION) -> Any:
    """Get the existing `aqi_features` feature group, or create its metadata
    if this is the first run.

    Uses `online_enabled=False`: Phase 3's goal is an offline, training-ready
    feature store (batch reads for model training), not low-latency online
    feature serving, so no online store is provisioned.

    Args:
        fs: A Hopsworks `FeatureStore` handle, as returned by `login()`.
        description: Human-readable description shown in the Hopsworks UI.

    Returns:
        A Hopsworks `FeatureGroup` handle.

    Raises:
        FeatureStoreError: If the feature group cannot be retrieved/created.
    """
    try:
        return fs.get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            description=description,
            primary_key=PRIMARY_KEY,
            event_time=EVENT_TIME_COLUMN,
            online_enabled=False,
        )
    except Exception as exc:
        raise FeatureStoreError(
            f"Failed to get/create feature group '{FEATURE_GROUP_NAME}' "
            f"v{FEATURE_GROUP_VERSION}: {exc}"
        ) from exc


def insert_features(fg: Any, df: pd.DataFrame) -> None:
    """Insert (or upsert) a dataframe of features into a feature group.

    Args:
        fg: A Hopsworks `FeatureGroup` handle, as returned by
            `get_or_create_feature_group()`.
        df: Prepared dataframe (see `prepare_dataframe_for_feature_store()`).

    Raises:
        FeatureStoreError: If the insert fails.
    """
    try:
        fg.insert(df)
    except Exception as exc:
        raise FeatureStoreError(f"Failed to insert features into Hopsworks: {exc}") from exc
    logger.info(
        "Inserted %d rows into feature group '%s' v%s",
        len(df),
        FEATURE_GROUP_NAME,
        FEATURE_GROUP_VERSION,
    )


def read_features(fg: Any) -> pd.DataFrame:
    """Read all features back from a feature group.

    Args:
        fg: A Hopsworks `FeatureGroup` handle.

    Returns:
        A pandas DataFrame with the feature group's current contents.

    Raises:
        FeatureStoreError: If the read fails.
    """
    try:
        result_df = fg.read()
    except Exception as exc:
        raise FeatureStoreError(f"Failed to read features from Hopsworks: {exc}") from exc
    logger.info(
        "Read %d rows back from feature group '%s' v%s",
        len(result_df),
        FEATURE_GROUP_NAME,
        FEATURE_GROUP_VERSION,
    )
    return result_df


def sync_feature_store(feature_df_path: Path | str = DEFAULT_FEATURE_DF_PATH) -> pd.DataFrame:
    """End-to-end Phase 3 flow: load `feature_df.csv`, log in to Hopsworks,
    get/create the feature group, insert the data, and read it back to
    confirm the round trip.

    Args:
        feature_df_path: Path to the Phase 2 output CSV. Defaults to
            `data/processed/feature_df.csv`.

    Returns:
        The dataframe read back from Hopsworks (post-insert confirmation).

    Raises:
        FeatureStoreError: If the input file is missing, or any Hopsworks
            operation (login, create, insert, read) fails.
    """
    path = Path(feature_df_path)
    if not path.exists():
        raise FeatureStoreError(
            f"Feature dataframe not found at '{path}'. Run Phase 2 first: "
            f"python -m src.feature_pipeline.feature_engineering"
        )

    logger.info("Loading feature dataframe: %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), path)

    prepared_df = prepare_dataframe_for_feature_store(df)

    fs = login()
    fg = get_or_create_feature_group(fs)
    insert_features(fg, prepared_df)
    read_back_df = read_features(fg)

    logger.info("Phase 3 Hopsworks sync completed: %d rows round-tripped", len(read_back_df))
    return read_back_df


def main() -> int:
    """CLI entry point. Returns a process exit code (0 = success, 1 = failure)."""
    try:
        read_back_df = sync_feature_store()
        print(f"Hopsworks sync completed successfully. {len(read_back_df)} rows read back.")
        print(read_back_df.head())
        return 0
    except FeatureStoreError as exc:
        logger.error("Hopsworks sync failed: %s", exc)
        print(f"Hopsworks sync failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Hopsworks sync failed with an unexpected error")
        print(f"Hopsworks sync failed with an unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())