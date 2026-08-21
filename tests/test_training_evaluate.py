"""Unit tests for `src.training_pipeline.evaluate`."""

from __future__ import annotations

import numpy as np
import pytest

from src.training_pipeline.evaluate import (
    METRIC_COLUMNS,
    _compute_smape,
    build_comparison_table,
    compute_metrics,
    evaluate_model,
    select_best_model,
)


def test_compute_metrics_perfect_predictions():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    metrics = compute_metrics(y_true, y_pred)

    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["smape"] == pytest.approx(0.0)


def test_compute_metrics_known_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 2.0, 4.0, 4.0])
    metrics = compute_metrics(y_true, y_pred)

    # errors: 1, 0, 1, 0 -> MAE = 0.5, RMSE = sqrt(0.5) approx 0.707
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.5))
    # smape = 100 * mean(2*|e|/(|y|+|yhat|)) = 100 * (2/3 + 0 + 2/7 + 0)/4
    assert metrics["smape"] == pytest.approx(100 * (2 / 3 + 2 / 7) / 4)


def test_compute_metrics_returns_all_expected_keys():
    metrics = compute_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.5, 2.5]))
    assert tuple(metrics.keys()) == METRIC_COLUMNS


def test_smape_handles_zero_denominator():
    # Both values zero -> the ratio is exact, so the error is zero.
    assert _compute_smape(np.array([0.0, 0.0]), np.array([0.0, 0.0])) == 0.0
    # One-sided zero -> 200% (the maximum possible symmetric error).
    assert _compute_smape(np.array([0.0]), np.array([5.0])) == pytest.approx(200.0)


def test_evaluate_model_calls_predict_and_computes_metrics():
    class DummyModel:
        def predict(self, X):
            return np.array([1.0, 2.0, 3.0])

    y_test = np.array([1.0, 2.0, 3.0])
    metrics = evaluate_model(DummyModel(), X_test=None, y_test=y_test)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["smape"] == pytest.approx(0.0)


def test_build_comparison_table_sorted_by_rmse_ascending():
    results = {
        "Model A": {"rmse": 10.0, "mae": 8.0, "r2": 0.5, "smape": 20.0},
        "Model B": {"rmse": 5.0, "mae": 4.0, "r2": 0.8, "smape": 10.0},
        "Model C": {"rmse": 20.0, "mae": 15.0, "r2": 0.1, "smape": 30.0},
    }
    table = build_comparison_table(results)

    assert list(table["model"]) == ["Model B", "Model A", "Model C"]
    assert table.iloc[0]["rmse"] == 5.0
    assert list(table.columns) == ["model", "rmse", "mae", "r2", "smape"]


def test_select_best_model_returns_lowest_rmse():
    results = {
        "Model A": {"rmse": 10.0, "mae": 8.0, "r2": 0.5, "smape": 20.0},
        "Model B": {"rmse": 5.0, "mae": 4.0, "r2": 0.8, "smape": 10.0},
    }
    assert select_best_model(results) == "Model B"


def test_select_best_model_raises_on_empty_results():
    with pytest.raises(ValueError):
        select_best_model({})
