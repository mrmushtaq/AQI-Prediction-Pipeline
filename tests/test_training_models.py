"""Unit tests for `src.training_pipeline.models`."""

from __future__ import annotations

import numpy as np
import pytest

from src.training_pipeline import models as models_module


def test_get_model_candidates_includes_five_tree_models_and_persistence():
    candidates = models_module.get_model_candidates(random_state=1)

    assert set(candidates) == {
        "LightGBM",
        "XGBoost",
        "CatBoost",
        "Extra Trees",
        "Random Forest",
        models_module.PERSISTENCE_MODEL_NAME,
    }


def test_get_model_candidates_raises_without_optional_deps(monkeypatch):
    monkeypatch.setattr(models_module, "HAS_LIGHTGBM", False)
    monkeypatch.setattr(models_module, "HAS_XGBOOST", False)
    monkeypatch.setattr(models_module, "HAS_CATBOOST", False)
    with pytest.raises(RuntimeError, match="lightgbm"):
        models_module.get_model_candidates()


@pytest.mark.parametrize(
    "name",
    ["LightGBM", "XGBoost", "CatBoost", "Extra Trees", "Random Forest"],
)
def test_tree_candidates_fit_predict_roundtrip(name):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = X[:, 0] * 2 + X[:, 1] - 1

    model = models_module.get_model_candidates(random_state=1)[name]
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (60,)


def test_persistence_baseline_uses_current_aqi_column():
    model = models_module.PersistenceBaselineRegressor()
    X = np.array([[1.0, 2.0, 50.0], [1.0, 2.0, 120.0], [1.0, 2.0, 80.0]])
    model.fit(X, np.array([0.0, 0.0, 0.0]))
    preds = model.predict(X)
    np.testing.assert_allclose(preds, np.array([50.0, 120.0, 80.0]))


def test_persistence_baseline_reads_named_aqi_column_from_dataframe():
    import pandas as pd

    model = models_module.PersistenceBaselineRegressor()
    X = pd.DataFrame({"temperature": [1.0, 2.0], "aqi": [50.0, 120.0]})
    preds = model.predict(X)
    np.testing.assert_allclose(preds, np.array([50.0, 120.0]))


def test_persistence_baseline_raises_when_aqi_column_missing():
    import pandas as pd

    model = models_module.PersistenceBaselineRegressor()
    with pytest.raises(ValueError):
        model.predict(pd.DataFrame({"temperature": [1.0]}))


def test_param_distributions_empty_for_persistence_and_nonempty_for_trees():
    assert models_module.get_param_distributions(models_module.PERSISTENCE_MODEL_NAME) == {}
    for name in ["LightGBM", "XGBoost", "CatBoost", "Extra Trees", "Random Forest"]:
        assert models_module.get_param_distributions(name)  # non-empty


def test_param_distributions_raises_for_unknown_model():
    with pytest.raises(KeyError):
        models_module.get_param_distributions("Not A Model")


def test_get_package_versions_includes_model_libraries():
    versions = models_module.get_package_versions()
    assert "lightgbm" in versions
    assert "xgboost" in versions
    assert "catboost" in versions
