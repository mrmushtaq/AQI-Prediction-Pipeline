"""Unit tests for `src.training_pipeline.model_registry`."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from src.common.exceptions import TrainingError
from src.training_pipeline.model_registry import (
    FEATURE_COLUMNS_FILE,
    METADATA_FILE,
    MODEL_FILE,
    find_latest_run,
    load_model_locally,
    register_model_to_hopsworks,
    save_model_locally,
)


@pytest.fixture
def fitted_sklearn_model():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    y = X[:, 0] * 2 + 1
    model = Ridge(alpha=1.0).fit(X, y)
    return model


def test_save_and_load_sklearn_model_roundtrip(tmp_path, fitted_sklearn_model):
    model = fitted_sklearn_model
    feature_columns = ["a", "b", "c"]
    metrics = {"rmse": 1.23, "mae": 0.98, "r2": 0.87, "smape": 12.5}
    split_sizes = {"train": 20, "val": 5, "test": 5}
    package_versions = {"scikit-learn": "1.9.0"}

    run_dir = save_model_locally(
        model=model,
        model_name="Ridge Regression",
        feature_columns=feature_columns,
        metrics=metrics,
        split_sizes=split_sizes,
        package_versions=package_versions,
        registry_dir=tmp_path,
    )

    assert run_dir.exists()
    assert (run_dir / MODEL_FILE).exists()
    assert (run_dir / FEATURE_COLUMNS_FILE).exists()
    assert (run_dir / METADATA_FILE).exists()
    # Tree-based pipeline persists no scaler.
    assert not (run_dir / "scaler.joblib").exists()

    loaded_model, metadata = load_model_locally(run_dir)

    assert metadata["model_name"] == "Ridge Regression"
    assert metadata["model_type"] == "sklearn"
    assert metadata["feature_columns"] == feature_columns
    assert metadata["metrics"] == metrics
    assert metadata["split_sizes"] == split_sizes
    assert metadata["package_versions"] == package_versions
    assert "training_date" in metadata

    # No scaler means inference needs only the model + feature columns.
    rng = np.random.default_rng(1)
    X_new = rng.normal(size=(5, 3))
    np.testing.assert_allclose(loaded_model.predict(X_new), model.predict(X_new))


def test_feature_columns_sidecar_matches_metadata(tmp_path, fitted_sklearn_model):
    import json

    run_dir = save_model_locally(
        model=fitted_sklearn_model,
        model_name="Ridge Regression",
        feature_columns=["a", "b", "c"],
        metrics={"rmse": 1.0, "mae": 0.8, "r2": 0.9, "smape": 5.0},
        split_sizes={"train": 20, "val": 5, "test": 5},
        package_versions={},
        registry_dir=tmp_path,
    )
    with open(run_dir / FEATURE_COLUMNS_FILE, "r", encoding="utf-8") as f:
        sidecar = json.load(f)
    assert sidecar == ["a", "b", "c"]


def test_load_model_locally_raises_when_metadata_missing(tmp_path):
    empty_dir = tmp_path / "not_a_run_dir"
    empty_dir.mkdir()
    with pytest.raises(TrainingError):
        load_model_locally(empty_dir)


def test_find_latest_run_returns_most_recent(tmp_path, fitted_sklearn_model):
    kwargs = {
        "model": fitted_sklearn_model,
        "model_name": "Ridge Regression",
        "feature_columns": ["a", "b", "c"],
        "metrics": {"rmse": 1.0, "mae": 0.8, "r2": 0.9, "smape": 5.0},
        "split_sizes": {"train": 20, "val": 5, "test": 5},
        "package_versions": {},
        "registry_dir": tmp_path,
    }
    save_model_locally(**kwargs)
    save_model_locally(**kwargs)

    latest = find_latest_run(tmp_path)
    assert latest is not None
    assert latest == max((p for p in tmp_path.iterdir() if p.is_dir()), key=lambda p: p.name)


def test_find_latest_run_prefers_newest_timestamp_over_name_prefix(tmp_path, fitted_sklearn_model):
    """The latest run is decided by its timestamp, never by its model-name
    prefix: 'extra_trees_...' created after 'random_forest_...' must win even
    though 'random_forest' sorts after 'extra_trees' alphabetically.
    """
    import joblib

    def make_run(name, timestamp):
        run_dir = tmp_path / f"{name}_{timestamp}"
        run_dir.mkdir()
        joblib.dump(fitted_sklearn_model, run_dir / MODEL_FILE)
        with open(run_dir / METADATA_FILE, "w", encoding="utf-8") as f:
            import json

            json.dump({"model_name": name, "model_file": MODEL_FILE, "feature_columns": ["a"]}, f)
        return run_dir

    older = make_run("random_forest", "20260730T114801Z")
    newer = make_run("extra_trees", "20260802T195143Z")

    latest = find_latest_run(tmp_path)
    assert latest == newer
    assert latest != older


def test_find_latest_run_returns_none_for_empty_registry(tmp_path):
    assert find_latest_run(tmp_path) is None


def test_register_model_to_hopsworks_returns_false_on_failure(tmp_path):
    """Registering to Hopsworks is best-effort: any failure (missing
    credentials, package not installed, network error) must return False
    rather than raising, since the local registry copy is authoritative.
    """
    with patch(
        "configs.config.load_hopsworks_settings",
        side_effect=Exception("HOPSWORKS_API_KEY not set"),
    ):
        result = register_model_to_hopsworks(
            run_dir=tmp_path, model_name="Ridge Regression", metrics={"rmse": 1.0}
        )
    assert result is False
