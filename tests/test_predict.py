"""Unit tests for `src.training_pipeline.predict`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.common.exceptions import TrainingError
from src.training_pipeline.model_registry import save_model_locally
from src.training_pipeline.predict import build_feature_matrix, predict


@pytest.fixture
def registry_with_model(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    y = X[:, 0] * 2 + 1
    model = Ridge(alpha=1.0).fit(X, y)
    feature_columns = ["temperature", "humidity", "aqi"]

    run_dir = save_model_locally(
        model=model,
        model_name="Ridge Regression",
        feature_columns=feature_columns,
        metrics={"rmse": 1.0, "mae": 0.8, "r2": 0.9, "smape": 5.0},
        split_sizes={"train": 20, "val": 5, "test": 5},
        package_versions={"scikit-learn": "1.9.0"},
        registry_dir=tmp_path,
    )
    return run_dir, feature_columns


def test_build_feature_matrix_selects_ordered_columns_and_drops_na():
    df = pd.DataFrame(
        {
            "temperature": [20.0, 21.0, 22.0],
            "humidity": [50.0, 60.0, 70.0],
            "aqi": [100.0, np.nan, 120.0],
            "unused_column": [1, 2, 3],
        }
    )
    X = build_feature_matrix(df, ["temperature", "humidity", "aqi"])

    assert list(X.columns) == ["temperature", "humidity", "aqi"]
    assert len(X) == 2  # the NaN row is dropped


def test_build_feature_matrix_raises_on_missing_feature_column():
    df = pd.DataFrame({"temperature": [20.0]})
    with pytest.raises(TrainingError):
        build_feature_matrix(df, ["temperature", "humidity"])


def test_predict_uses_explicit_run_dir(registry_with_model):
    run_dir, feature_columns = registry_with_model
    df = pd.DataFrame(
        {
            "temperature": [20.0, 21.0],
            "humidity": [50.0, 60.0],
            "aqi": [100.0, 110.0],
        }
    )
    predictions, used_columns, metadata = predict(df, run_dir=run_dir)

    assert predictions.shape == (2,)
    assert used_columns == feature_columns
    assert metadata["model_name"] == "Ridge Regression"


def test_predict_raises_when_no_run_found(tmp_path):
    df = pd.DataFrame({"temperature": [20.0]})
    with pytest.raises(TrainingError):
        predict(df, run_dir=tmp_path / "does_not_exist")
