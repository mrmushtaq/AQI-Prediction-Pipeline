from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from configs.config import PROJECT_ROOT

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
REGISTRY_DIR = PROJECT_ROOT / "data" / "model_registry"


def load_hopsworks_data() -> pd.DataFrame:
    """Load the durable engineered dataset from Hopsworks."""
    from src.training_pipeline.data_loader import load_features_from_hopsworks

    frame = load_features_from_hopsworks()
    if "collection_timestamp" in frame.columns:
        frame["collection_timestamp"] = pd.to_datetime(frame["collection_timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["collection_timestamp"]).sort_values("collection_timestamp").reset_index(drop=True)
    frame.attrs["source"] = "Hopsworks Feature Store"
    return frame


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "collection_timestamp" in frame.columns:
        frame["collection_timestamp"] = pd.to_datetime(frame["collection_timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["collection_timestamp"])
    return frame.sort_values("collection_timestamp").reset_index(drop=True) if "collection_timestamp" in frame.columns else frame


def load_historical_data() -> tuple[pd.DataFrame, str]:
    """Load and combine historical + live processed observations.

    Historical data provides the long-term feature history required by the
    forecasting models. Live observations are appended separately by the
    dashboard, so this function focuses on the persisted datasets.
    """
    candidates = [
        (PROCESSED_DIR / "aqi_dataset_historical.csv", "Historical processed data"),
        (PROCESSED_DIR / "feature_df.csv", "Engineered feature data"),
        (PROCESSED_DIR / "aqi_dataset.csv", "Live processed data"),
    ]

    frames = []

    for path, label in candidates:
        frame = _read_csv(path)

        if frame.empty:
            continue

        if "collection_timestamp" not in frame.columns:
            continue

        frame["collection_timestamp"] = pd.to_datetime(
            frame["collection_timestamp"],
            utc=True,
            errors="coerce",
        )

        frame = frame.dropna(subset=["collection_timestamp"]).copy()
        frame["_source_file"] = label

        frames.append(frame)

    if not frames:
        return pd.DataFrame(), "Data unavailable"

    # Combine all available persisted datasets.
    combined = pd.concat(frames, ignore_index=True, sort=False)

    # Remove exact duplicate observations.
    duplicate_keys = [
        column
        for column in ["city", "collection_timestamp"]
        if column in combined.columns
    ]

    if duplicate_keys:
        combined = combined.drop_duplicates(
            subset=duplicate_keys,
            keep="last",
        )
    else:
        combined = combined.drop_duplicates()

    # Chronological order is critical for lag/rolling features.
    combined = (
        combined
        .sort_values("collection_timestamp")
        .reset_index(drop=True)
    )

    combined.attrs["path"] = "combined processed datasets"
    combined.attrs["source"] = "Historical + engineered + live processed data"

    return combined, "Historical + live processed data"


def load_metadata() -> list[dict[str, Any]]:
    records = []
    for path in sorted(METADATA_DIR.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as handle:
                item = json.load(handle)
            records.append(item)
        except (OSError, json.JSONDecodeError):
            continue
    return records


def latest_metadata() -> dict[str, Any] | None:
    records = load_metadata()
    return records[-1] if records else None


def model_runs() -> list[dict[str, Any]]:
    runs = []
    for path in sorted(REGISTRY_DIR.iterdir()) if REGISTRY_DIR.exists() else []:
        metadata_path = path / "metadata.json"
        if not path.is_dir() or not metadata_path.exists():
            continue
        try:
            with metadata_path.open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            metadata["run_dir"] = str(path)
            metadata["run_name"] = path.name
            runs.append(metadata)
        except (OSError, json.JSONDecodeError):
            continue
    return runs


def dataset_quality(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "duplicates": 0, "missing": pd.Series(dtype=float), "invalid_aqi": 0}
    key = [column for column in ["city", "collection_timestamp"] if column in frame.columns]
    duplicate_count = int(frame.duplicated(subset=key).sum()) if key else int(frame.duplicated().sum())
    numeric_aqi = pd.to_numeric(frame["aqi"], errors="coerce") if "aqi" in frame else pd.Series(dtype=float)
    invalid_aqi = int(((numeric_aqi < 0) | (numeric_aqi > 500)).sum()) if not numeric_aqi.empty else 0
    return {"rows": len(frame), "duplicates": duplicate_count, "missing": frame.isna().mean().sort_values(ascending=False), "invalid_aqi": invalid_aqi}


def source_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if "aqi_source" not in frame.columns:
        return pd.DataFrame(columns=["source", "records", "percent"])
    counts = frame["aqi_source"].fillna("Unavailable").value_counts().rename_axis("source").reset_index(name="records")
    counts["percent"] = counts["records"] / len(frame) * 100
    return counts
