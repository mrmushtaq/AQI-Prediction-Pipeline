"""Unit tests for `src.training_pipeline.horizon_sweep` helpers.

The LightGBM training orchestration (`run_horizon` / `main`) is intentionally
not unit-tested here since it runs a 40-iteration RandomizedSearchCV over five
horizons; only the pure helper functions are covered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training_pipeline import horizon_sweep


def test_metrics_perfect_predictions():
    y_true = np.array([10.0, 20.0, 30.0])
    m = horizon_sweep.metrics(y_true, y_true.copy())
    assert m["rmse"] == pytest.approx(0.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["smape"] == pytest.approx(0.0)


def test_metrics_known_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 2.0, 4.0, 4.0])
    m = horizon_sweep.metrics(y_true, y_pred)
    # errors: 1, 0, 1, 0 -> MAE = 0.5, RMSE = sqrt(0.5) approx 0.707
    assert m["mae"] == pytest.approx(0.5)
    assert m["rmse"] == pytest.approx(np.sqrt(0.5))


def test_naive_baselines_returns_all_four_baselines_with_metric_keys():
    n = 200
    aqi = pd.Series(np.arange(n, dtype=float))
    month = pd.Series([1] * 100 + [2] * 100)
    y_true = pd.Series(aqi.to_numpy() + 5.0)
    # Both months must appear in the training window so the seasonal-month
    # baseline has no NaN groups on the scored rows.
    train_mask = pd.Series(np.arange(n) < 150)

    result = horizon_sweep.naive_baselines(aqi, month, y_true, train_mask)

    assert set(result) == {"persistence", "roll_mean_24h", "roll_mean_7d", "seasonal_month"}
    for name in result:
        assert set(result[name]) == {"rmse", "mae", "r2", "smape"}


def test_naive_baselines_persistence_is_exact_for_zero_ahead():
    aqi = pd.Series(np.arange(10, dtype=float))
    month = pd.Series([1] * 10)
    train_mask = pd.Series(np.ones(10, dtype=bool))
    result = horizon_sweep.naive_baselines(aqi, month, aqi, train_mask)
    assert result["persistence"]["rmse"] == pytest.approx(0.0)
    assert result["persistence"]["r2"] == pytest.approx(1.0)


def test_volatility_drift_groups_by_month():
    df = pd.DataFrame(
        {
            "collection_timestamp": pd.date_range(
                "2026-01-01", periods=72, freq="1h", tz="UTC"
            ),
            "aqi": np.linspace(50, 150, 72),
        }
    )
    rows = horizon_sweep.volatility_drift(df)

    # 72 hourly rows starting Jan 1 fall entirely inside one calendar month.
    assert len(rows) == 1
    row = rows[0]
    assert row["month"] == "2026-01"
    assert row["n"] == 72
    assert set(row) == {"month", "n", "aqi_mean", "change_std", "roll_std_24h_mean"}
