"""
Model registry for the training pipeline (Phase 5).

Two tiers, matching the pattern already established for the feature store
(`feature_store.py`'s `hopsworks`/`hsfs` fallback):

    1. **Local registry** (`data/model_registry/`) -- always used. Every
       trained model is saved here with its `feature_columns.json` and a
       metadata JSON (metrics, feature columns, split sizes, package
       versions, training date), so training never depends on Hopsworks
       being reachable or installed.
    2. **Hopsworks Model Registry** (optional, best-effort) -- if the
       `hopsworks` package is available and credentials are configured,
       the best model is *also* pushed to Hopsworks's Model Registry,
       satisfying the project requirement "Store trained models in Model
       Registry". A failure here is logged as a warning and does not fail
       the training run -- the local copy is always the source of truth.

All models saved by this pipeline are tree-based ensembles (LightGBM,
XGBoost, CatBoost, Extra Trees, Random Forest), so every model is a plain
scikit-learn-compatible estimator serialized with ``joblib`` -- there is no
TensorFlow/Keras model type in this phase. Because tree-based models are
scale-invariant, **no feature scaler is persisted**; inference simply needs
the model and the `feature_columns.json` list.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib

from configs.config import PROJECT_ROOT
from src.common.exceptions import TrainingError
from src.common.logger import get_logger

logger = get_logger(__name__)

DEFAULT_REGISTRY_DIR = PROJECT_ROOT / "data" / "model_registry"

MODEL_FILE = "model.joblib"
FEATURE_COLUMNS_FILE = "feature_columns.json"
METADATA_FILE = "metadata.json"
FORECAST_MODEL_PREFIX = "aqi_daily_forecast_v6"


def save_model_locally(
    model: Any,
    model_name: str,
    feature_columns: Sequence[str],
    metrics: Mapping[str, float],
    split_sizes: Mapping[str, int],
    package_versions: Mapping[str, str],
    registry_dir: Path | str = DEFAULT_REGISTRY_DIR,
) -> Path:
    """Save a trained model and its metadata to the local registry.

    Creates a timestamped subdirectory under `registry_dir` containing:
        - `model.joblib` -- the serialized estimator
        - `feature_columns.json` -- the ordered feature list the model expects
        - `metadata.json` -- model name, training date, feature list, split
          sizes, evaluation metrics, and package versions

    Args:
        model: The fitted tree-based model (scikit-learn-compatible).
        model_name: Human-readable model name (e.g. `"LightGBM"`).
        feature_columns: Ordered list of feature column names the model
            expects, so inference code can reconstruct `X` correctly.
        metrics: Evaluation metrics dict (`rmse`, `mae`, `r2`, `smape`).
        split_sizes: Dict with the row counts of each split, e.g.
            `{"train": ..., "val": ..., "test": ...}`.
        package_versions: Dict mapping package name -> version, for
            reproducibility.
        registry_dir: Root directory for the local model registry.

    Returns:
        Path to the created run directory.

    Raises:
        TrainingError: If saving fails.
    """
    registry_dir = Path(registry_dir)
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    run_dir = registry_dir / f"{slug}_{run_timestamp}"

    feature_list = list(feature_columns)

    try:
        run_dir.mkdir(parents=True, exist_ok=True)

        model_path = run_dir / MODEL_FILE
        joblib.dump(model, model_path)

        with open(run_dir / FEATURE_COLUMNS_FILE, "w", encoding="utf-8") as f:
            json.dump(feature_list, f, indent=2)

        training_date = datetime.now(timezone.utc).isoformat()
        metadata = {
            "model_name": model_name,
            "model_type": "sklearn",
            "model_file": MODEL_FILE,
            "feature_columns": feature_list,
            "metrics": dict(metrics),
            "split_sizes": dict(split_sizes),
            "package_versions": dict(package_versions),
            "training_date": training_date,
            "saved_at": training_date,
        }
        with open(run_dir / METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    except OSError as exc:
        raise TrainingError(f"Failed to save model to local registry at '{run_dir}': {exc}") from exc

    logger.info("Saved model '%s' to local registry: %s", model_name, run_dir)
    return run_dir


def load_model_locally(run_dir: Path | str) -> tuple[Any, dict]:
    """Load a model and its metadata from a local registry run directory.

    Args:
        run_dir: Path returned by a previous `save_model_locally` call.

    Returns:
        `(model, metadata)`, where `metadata["feature_columns"]` is the
        ordered list of features the model expects.

    Raises:
        TrainingError: If the directory or its expected files are missing.
    """
    run_dir = Path(run_dir)
    metadata_path = run_dir / METADATA_FILE
    if not metadata_path.exists():
        raise TrainingError(
            f"No {METADATA_FILE} found in '{run_dir}' -- not a valid model registry run directory."
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model_path = run_dir / metadata.get("model_file", MODEL_FILE)
    if not model_path.exists():
        raise TrainingError(f"Model file missing in '{run_dir}': expected {model_path.name}.")

    model = joblib.load(model_path)

    if "feature_columns" not in metadata:
        # Older runs may have stored the feature list only in a sidecar file.
        sidecar = run_dir / FEATURE_COLUMNS_FILE
        if sidecar.exists():
            with open(sidecar, "r", encoding="utf-8") as f:
                metadata["feature_columns"] = json.load(f)
        else:
            raise TrainingError(
                f"No feature_columns found for model in '{run_dir}' (missing "
                f"'{FEATURE_COLUMNS_FILE}' too)."
            )

    logger.info("Loaded model '%s' from local registry: %s", metadata.get("model_name"), run_dir)
    return model, metadata


def find_latest_run(registry_dir: Path | str = DEFAULT_REGISTRY_DIR) -> Path | None:
    """Resolve the most recent model run directory in the local registry.

    Run directories are named `<slug>_<YYYYMMDDTHHMMSSZ>`, so the latest is
    the one with the most recent timestamp suffix (UTC, zero-padded, hence
    lexicographically sortable). The model-name prefix is ignored, so a newer
    run of a differently-named model always wins. Returns `None` if the
    registry is empty or doesn't exist.

    Args:
        registry_dir: Root directory for the local model registry.

    Returns:
        Path to the most recent run directory, or `None`.
    """
    registry_dir = Path(registry_dir)
    if not registry_dir.is_dir():
        return None

    candidates = [
        p for p in registry_dir.iterdir()
        if p.is_dir() and (p / METADATA_FILE).exists() and (p / MODEL_FILE).exists()
    ]
    if not candidates:
        return None

    latest = max(candidates, key=_run_sort_key)
    logger.info("Resolved latest model run: %s", latest)
    return latest


_TIMESTAMP_RE = re.compile(r"\d{8}T\d{6}Z$")


def _run_sort_key(run_dir: Path) -> str:
    """Sort key for a run directory: its trailing UTC timestamp when present,
    falling back to the full directory name otherwise."""
    name = run_dir.name
    if "_" in name:
        timestamp = name.rsplit("_", 1)[-1]
        if _TIMESTAMP_RE.fullmatch(timestamp):
            return timestamp
    return name


def register_model_to_hopsworks(
    run_dir: Path | str,
    model_name: str,
    metrics: Mapping[str, float],
) -> bool:
    """Best-effort push of a locally-saved model to the Hopsworks Model Registry.

    This is intentionally best-effort: Hopsworks (or its Windows-specific
    `twofish` build dependency -- see `feature_store.py`'s setup notes) may
    not be available in every environment this pipeline runs in. A failure
    here is logged and returns `False` rather than raising, since the local
    registry copy (see `save_model_locally`) is always the authoritative
    record regardless of whether this extra sync succeeds.

    Args:
        run_dir: Local registry run directory (from `save_model_locally`)
            containing the model files to upload.
        model_name: Name to register the model under in Hopsworks.
        metrics: Evaluation metrics to attach to the registered model
            version.

    Returns:
        `True` if the push succeeded, `False` otherwise.
    """
    run_dir = Path(run_dir)
    try:
        from configs.config import load_hopsworks_settings

        settings = load_hopsworks_settings()

        import hopsworks

        project = hopsworks.login(project=settings.project_name, api_key_value=settings.api_key)
        mr = project.get_model_registry()

        model_meta = mr.python.create_model(
            name=model_name.lower().replace(" ", "_").replace("(", "").replace(")", ""),
            metrics=dict(metrics),
            description=f"AQI forecasting model: {model_name}",
        )
        model_meta.save(str(run_dir))
        logger.info("Registered model '%s' to Hopsworks Model Registry", model_name)
        return True
    except Exception as exc:  # noqa: BLE001 -- any failure here is non-fatal by design
        logger.warning(
            "Could not register model to Hopsworks Model Registry (non-fatal, "
            "local registry copy at '%s' remains authoritative): %s",
            run_dir,
            exc,
        )
        return False


def register_forecast_models_to_hopsworks(
    run_dir: Path | str,
    metadata: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Register all daily v6 forecast artifacts as one model per horizon."""
    run_dir = Path(run_dir)
    try:
        from configs.config import load_hopsworks_settings
        import hopsworks

        settings = load_hopsworks_settings()
        project = hopsworks.login(project=settings.project_name, api_key_value=settings.api_key)
        model_registry = project.get_model_registry()
        for label, item in metadata.items():
            model_meta = model_registry.python.create_model(
                name=f"{FORECAST_MODEL_PREFIX}_{label}",
                metrics=dict(item.get("test_metrics", {})),
                description=f"Sukkur AQI daily forecast model for {label}",
            )
            model_meta.save(str(run_dir))
        logger.info("Registered v6 forecast models to Hopsworks Model Registry")
        return True
    except Exception as exc:  # noqa: BLE001 -- registry sync is best-effort
        logger.warning("Could not register v6 forecast models to Hopsworks: %s", exc)
        return False
