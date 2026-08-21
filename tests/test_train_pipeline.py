"""End-to-end tests for `src.training_pipeline.train_pipeline`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.exceptions import TrainingError
from src.training_pipeline.train_pipeline import run_training_pipeline


@pytest.fixture
def synthetic_feature_csv(tmp_path):
    """A small, fast-to-train synthetic dataset with a genuine linear signal,
    matching the real feature_df.csv schema closely enough to exercise the
    full pipeline (feature selection, chronological 70/15/15 split,
    time-series-CV tuning, evaluation, and local registry save).
    """
    n = 150
    rng = np.random.default_rng(42)
    dates = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")

    aqi = 100 + np.cumsum(rng.normal(0, 3, n))
    pm2_5 = aqi * 0.8 + rng.normal(0, 2, n)
    temperature = 20 + 10 * np.sin(np.linspace(0, 10, n)) + rng.normal(0, 1, n)
    humidity = rng.uniform(20, 80, n)

    df = pd.DataFrame(
        {
            "timestamp": dates.astype(str),
            "collection_timestamp": dates.astype(str),
            "city": "Sukkur",
            "weather": "clear sky",
            "weather_source": "OpenWeather",
            "aqi_source": "AQICN-geo",
            "aqi": aqi,
            "pm2_5": pm2_5,
            "pm10": np.nan,
            "pollution_index": np.nan,
            "temperature": temperature,
            "humidity": humidity,
        }
    )
    # Target: aqi shifted 24 steps ahead (drop the tail rows lacking a target).
    df["aqi_target"] = df["aqi"].shift(-24)
    df = df.dropna(subset=["aqi_target"]).reset_index(drop=True)

    path = tmp_path / "feature_df.csv"
    df.to_csv(path, index=False)
    return path


def test_run_training_pipeline_end_to_end_local_source(synthetic_feature_csv, monkeypatch, tmp_path):
    import src.training_pipeline.model_registry as model_registry_module
    from src.training_pipeline import train_pipeline as train_pipeline_module

    def fake_save_model_locally(**kwargs):
        return model_registry_module.save_model_locally(registry_dir=tmp_path / "registry", **kwargs)

    monkeypatch.setattr(train_pipeline_module, "save_model_locally", fake_save_model_locally)

    results = run_training_pipeline(
        source="local",
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_state=1,
        n_splits=2,
        gap=1,
        n_iter=3,
        local_path=synthetic_feature_csv,
    )

    assert set(results.keys()) == {
        "comparison_table", "best_model_name", "best_model_metrics",
        "persistence_metrics", "walk_forward_metrics",
        "walk_forward_persistence_metrics", "walk_forward_n",
        "split_sizes", "registry_path", "hopsworks_registered",
    }

    # The synthetic dataset is far below the walk-forward minimum training
    # window, so the walk-forward backtest is skipped (None / 0), not absent.
    assert results["walk_forward_metrics"] is None
    assert results["walk_forward_persistence_metrics"] is None
    assert results["walk_forward_n"] == 0

    # All five tree candidates plus the persistence baseline.
    table = results["comparison_table"]
    assert len(table) == 6
    assert set(table["model"]) == {
        "LightGBM", "XGBoost", "CatBoost", "Extra Trees", "Random Forest",
        "Persistence Baseline",
    }
    # Metrics present and the table sorted ascending by RMSE (best first).
    for metric in ("rmse", "mae", "r2", "smape"):
        assert metric in table.columns
    assert table["rmse"].tolist() == sorted(table["rmse"].tolist())

    # The winner is a real trained model, never the persistence baseline.
    assert results["best_model_name"] in table["model"].tolist()
    assert results["best_model_name"] != "Persistence Baseline"
    assert set(results["best_model_metrics"].keys()) == {"rmse", "mae", "r2", "smape"}
    assert set(results["persistence_metrics"].keys()) == {"rmse", "mae", "r2", "smape"}

    assert set(results["split_sizes"].keys()) == {"train", "val", "test"}
    assert all(isinstance(v, int) and v > 0 for v in results["split_sizes"].values())
    # Chronological ordering must hold: training data is always older than test.
    assert results["split_sizes"]["train"] > results["split_sizes"]["test"]
    assert results["registry_path"].exists()
    assert (results["registry_path"] / "model.joblib").exists()
    assert (results["registry_path"] / "feature_columns.json").exists()
    assert (results["registry_path"] / "metadata.json").exists()
    assert results["hopsworks_registered"] is False  # not requested


def test_run_training_pipeline_raises_on_missing_local_file(tmp_path):
    with pytest.raises(TrainingError):
        run_training_pipeline(source="local", local_path=tmp_path / "missing.csv")
