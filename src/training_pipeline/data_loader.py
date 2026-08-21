"""
Feature/target loader for the model training pipeline (Phase 5).

Supports two sources, selected via the ``source`` argument:

    - ``"hopsworks"`` (default, matches the project requirement "fetch
      historical features and targets from Feature Store"): reads the
      ``aqi_features`` feature group via `src.feature_pipeline.feature_store`.
    - ``"local"``: reads `data/processed/feature_df.csv` directly. Useful for
      fast local iteration/testing without needing Hopsworks credentials, and
      as a fallback if the feature store is temporarily unavailable.

Both paths return an identically-shaped DataFrame, so everything downstream
(preprocessing, training) is source-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from configs.config import PROJECT_ROOT
from src.common.exceptions import TrainingError
from src.common.logger import get_logger

logger = get_logger(__name__)

DEFAULT_FEATURE_DF_PATH = PROJECT_ROOT / "data" / "processed" / "feature_df.csv"

DataSource = Literal["local", "hopsworks"]


def load_features_from_local_csv(path: Path | str = DEFAULT_FEATURE_DF_PATH) -> pd.DataFrame:
    """Load the Phase 2 feature dataset directly from disk.

    Args:
        path: Path to `feature_df.csv`.

    Returns:
        The parsed DataFrame, with `collection_timestamp` as a proper
        datetime column.

    Raises:
        TrainingError: If the file doesn't exist or can't be parsed.
    """
    path = Path(path)
    if not path.exists():
        raise TrainingError(
            f"Feature dataframe not found at '{path}'. Run Phase 2 first: "
            f"python -m src.feature_pipeline.feature_engineering"
        )
    try:
        df = pd.read_csv(path, parse_dates=["collection_timestamp"])
    except (OSError, ValueError) as exc:
        raise TrainingError(f"Failed to load feature dataframe from '{path}': {exc}") from exc

    logger.info("Loaded %d rows, %d columns from local CSV: %s", len(df), len(df.columns), path)
    return df


def load_features_from_hopsworks() -> pd.DataFrame:
    """Load the engineered feature set from the Hopsworks Feature Store.

    Fetches the current version of the ``aqi_features`` feature group (see
    `src.feature_pipeline.feature_store` for its schema) and returns it as a
    plain pandas DataFrame, ready for the same preprocessing/training code
    used with the local-CSV path.

    Returns:
        The feature group's full contents as a DataFrame.

    Raises:
        TrainingError: If login, feature group retrieval, or the read fails.
    """
    from src.common.exceptions import FeatureStoreError
    from src.feature_pipeline.feature_store import (
        FEATURE_GROUP_NAME,
        FEATURE_GROUP_VERSION,
        login,
        read_features,
    )

    try:
        fs = login()
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        df = read_features(fg)
    except FeatureStoreError as exc:
        raise TrainingError(f"Failed to load features from Hopsworks: {exc}") from exc

    logger.info(
        "Loaded %d rows, %d columns from Hopsworks feature group '%s' v%s",
        len(df),
        len(df.columns),
        FEATURE_GROUP_NAME,
        FEATURE_GROUP_VERSION,
    )
    return df


def load_features(
    source: DataSource = "hopsworks",
    local_path: Path | str = DEFAULT_FEATURE_DF_PATH,
) -> pd.DataFrame:
    """Load the training feature set from the requested source.

    Args:
        source: `"hopsworks"` (default) or `"local"`.
        local_path: Path to `feature_df.csv`, used only when
            `source="local"`.

    Returns:
        The loaded DataFrame.

    Raises:
        TrainingError: If `source` is invalid, or the underlying load fails.
    """
    if source == "hopsworks":
        return load_features_from_hopsworks()
    if source == "local":
        return load_features_from_local_csv(local_path)
    raise TrainingError(f"Unknown feature source: {source!r} (expected 'hopsworks' or 'local')")
