"""Unit tests for `src.training_pipeline.walk_forward` (expanding-window
backtesting of the best model and of the persistence baseline)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.training_pipeline.walk_forward import (
    run_walk_forward,
    run_walk_forward_persistence,
)


def _make_df(n: int = 60) -> pd.DataFrame:
    """A small synthetic feature frame with an exact linear signal so the
    walk-forward predictions can be checked exactly (not just structurally)."""
    df = pd.DataFrame(
        {
            "collection_timestamp": pd.date_range(
                "2026-01-01", periods=n, freq="1h", tz="UTC"
            ),
            "x": np.arange(n, dtype=float),
        }
    )
    df["y"] = df["x"]
    return df


def _make_persistence_df(n: int = 60) -> pd.DataFrame:
    """A frame where `aqi_target` equals the current `aqi`, so the persistence
    baseline (which predicts the target using the current AQI) is exact."""
    rng = np.random.default_rng(0)
    df = _make_df(n)
    df["aqi"] = 100 + np.cumsum(rng.normal(0, 2, n))
    df["aqi_target"] = df["aqi"]
    return df


def test_run_walk_forward_skips_when_too_few_rows():
    df = _make_df(10)
    result = run_walk_forward(
        LinearRegression(), df, feature_cols=["x"], target_col="y",
        step=10, min_train=1500,
    )
    assert result == {"metrics": None, "n": 0, "n_windows": 0}


def test_run_walk_forward_returns_metrics_and_n():
    df = _make_df(60)
    result = run_walk_forward(
        LinearRegression(), df, feature_cols=["x"], target_col="y",
        step=10, min_train=20,
    )

    assert set(result.keys()) == {"metrics", "n", "n_windows"}
    assert result["n_windows"] == 4
    assert result["n"] == 40
    assert set(result["metrics"].keys()) == {"rmse", "mae", "r2", "smape"}
    # y == x exactly, so every expanding-window linear fit reproduces the
    # held-out blocks to within floating-point error.
    assert result["metrics"]["rmse"] < 1e-6
    assert result["metrics"]["r2"] > 0.999999


def test_run_walk_forward_last_block_is_shorter_than_step():
    # 25 rows: block 1 covers rows 10-19, block 2 covers rows 20-24 (5 rows).
    df = _make_df(25)
    result = run_walk_forward(
        LinearRegression(), df, feature_cols=["x"], target_col="y",
        step=10, min_train=10,
    )
    assert result["n_windows"] == 2
    assert result["n"] == 15


def test_run_walk_forward_does_not_mutate_passed_model():
    model = LinearRegression()
    model.fit(np.array([[1.0], [2.0], [3.0]]), np.array([2.0, 4.0, 6.0]))
    coef_before = model.coef_.copy()

    df = _make_df(60)
    run_walk_forward(model, df, feature_cols=["x"], target_col="y",
                     step=10, min_train=20)

    # A clone is refit per window; the passed estimator must be untouched.
    np.testing.assert_allclose(model.coef_, coef_before)


def test_run_walk_forward_persistence_exact_when_target_equals_aqi():
    df = _make_persistence_df(60)
    result = run_walk_forward_persistence(
        df, feature_cols=["x"], target_col="aqi_target", aqi_column="aqi",
        step=10, min_train=20,
    )

    assert result["n_windows"] == 4
    assert result["n"] == 40
    assert result["metrics"]["rmse"] < 1e-6
    assert result["metrics"]["r2"] > 0.999999


def test_run_walk_forward_persistence_skips_when_too_few_rows():
    df = _make_persistence_df(10)
    result = run_walk_forward_persistence(
        df, feature_cols=["x"], target_col="aqi_target", aqi_column="aqi",
        step=10, min_train=1500,
    )
    assert result == {"metrics": None, "n": 0, "n_windows": 0}
