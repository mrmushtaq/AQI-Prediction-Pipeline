"""
Storage utilities for persisting raw API responses and processed datasets.

Raw payloads (exact API responses) are stored as JSON under `data/raw/`.
Processed, tabular datasets are stored as CSV under `data/processed/`.
Both directories are created automatically if they do not already exist.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from src.common.exceptions import StorageError
from src.common.logger import get_logger

logger = get_logger(__name__)


def _ensure_parent_dir(path: Path) -> None:
    """Create the parent directory of `path` if it does not already exist."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StorageError(f"Failed to create directory '{path.parent}': {exc}") from exc


def save_json(data: Mapping[str, Any] | Sequence[Any], filepath: str | Path) -> Path:
    """Save a dict/list as a JSON file, creating directories as needed.

    Args:
        data: JSON-serializable data (dict or list).
        filepath: Destination path, e.g. `data/raw/weather_20260726.json`.

    Returns:
        The resolved `Path` the file was written to.

    Raises:
        StorageError: If the directory cannot be created or the write fails.
    """
    path = Path(filepath)
    _ensure_parent_dir(path)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Saved JSON file: %s", path)
        return path
    except (OSError, TypeError) as exc:
        raise StorageError(f"Failed to save JSON file '{path}': {exc}") from exc


def load_json(filepath: str | Path) -> Any:
    """Load and parse a JSON file.

    Args:
        filepath: Path to the JSON file to load.

    Returns:
        The parsed JSON content (dict or list).

    Raises:
        StorageError: If the file does not exist or cannot be parsed.
    """
    path = Path(filepath)
    if not path.exists():
        raise StorageError(f"JSON file not found: '{path}'")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Loaded JSON file: %s", path)
        return data
    except (OSError, json.JSONDecodeError) as exc:
        raise StorageError(f"Failed to load JSON file '{path}': {exc}") from exc


def save_csv(rows: Iterable[Mapping[str, Any]], filepath: str | Path) -> Path:
    """Save an iterable of flat dict records as a CSV file.

    The column order follows the keys of the first record. All records must
    share the same set of keys; a `StorageError` is raised otherwise, since a
    silently ragged CSV would corrupt downstream feature engineering.

    Args:
        rows: Iterable of dict-like records (e.g. one row per timestamp/city).
        filepath: Destination path, e.g. `data/processed/aqi_dataset.csv`.

    Returns:
        The resolved `Path` the file was written to.

    Raises:
        StorageError: If `rows` is empty, records have inconsistent keys, or
            the write fails.
    """
    path = Path(filepath)
    rows = list(rows)

    if not rows:
        raise StorageError(f"Cannot save CSV '{path}': no rows provided.")

    fieldnames = list(rows[0].keys())
    for i, row in enumerate(rows):
        if set(row.keys()) != set(fieldnames):
            raise StorageError(
                f"Cannot save CSV '{path}': row {i} has inconsistent columns "
                f"(expected {fieldnames}, got {list(row.keys())})."
            )

    _ensure_parent_dir(path)
    try:
        file_exists = path.exists()
        with open(path, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)
        logger.info("Saved CSV file (%d rows): %s", len(rows), path)
        return path
    except OSError as exc:
        raise StorageError(f"Failed to save CSV file '{path}': {exc}") from exc


METADATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "metadata"


def save_metadata(
    metadata: Mapping[str, Any],
    pipeline_type: str,
    filepath: str | Path | None = None,
) -> Path:
    """Save pipeline run metadata as a JSON file under ``data/metadata/``.

    Each run produces a timestamped metadata file so that every pipeline
    execution is traceable.

    Example metadata structure::

        {
            "run_time": "2026-07-26T12:00:00Z",
            "pipeline_type": "live",
            "city": "Islamabad",
            "weather_source": "OpenWeather",
            "aqi_source": "AQICN-city",
            "records": 1,
            "status": "SUCCESS"
        }

    Args:
        metadata: Dict of metadata fields to persist.
        pipeline_type: Short identifier such as ``"live"`` or ``"historical"``.
        filepath: Optional explicit path. If omitted, a path is generated
            inside ``data/metadata/`` using the current timestamp and pipeline
            type.

    Returns:
        The resolved ``Path`` the file was written to.

    Raises:
        StorageError: If the directory cannot be created or the write fails.
    """
    if filepath is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"metadata_{pipeline_type}_{timestamp}.json"
        filepath = METADATA_DIR / filename
    return save_json(metadata, filepath)


def load_csv(filepath: str | Path) -> list[dict[str, str]]:
    """Load a CSV file into a list of dict records (all values as strings).

    Args:
        filepath: Path to the CSV file to load.

    Returns:
        A list of row dicts.

    Raises:
        StorageError: If the file does not exist or cannot be parsed.
    """
    path = Path(filepath)
    if not path.exists():
        raise StorageError(f"CSV file not found: '{path}'")
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [dict(row) for row in reader]
        logger.info("Loaded CSV file (%d rows): %s", len(rows), path)
        return rows
    except (OSError, csv.Error) as exc:
        raise StorageError(f"Failed to load CSV file '{path}': {exc}") from exc
