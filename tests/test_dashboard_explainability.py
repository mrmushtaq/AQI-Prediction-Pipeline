"""Unit tests for `src.dashboard.explainability`."""

from __future__ import annotations

import numpy as np
import pytest

from src.dashboard.explainability import Explanation, explain_forecast
from tests.test_dashboard_forecast import _make_hourly_df, _make_v5_run


def test_explain_forecast_returns_one_explanation_per_horizon(tmp_path):
    run_dir = _make_v5_run(tmp_path)
    df = _make_hourly_df()
    explanations = explain_forecast(df, run_dir=run_dir, use_lime=False)

    assert len(explanations) == 3
    assert [e.label for e in explanations] == ["day1_avg", "day2_avg", "day3_avg"]
    for exp in explanations:
        assert isinstance(exp, Explanation)
        assert exp.prediction > 0
        assert exp.shap_values.shape == (len(exp.feature_cols),)
        assert exp.lime_values is None  # LIME disabled


def test_explain_forecast_shap_reconstructs_prediction(tmp_path):
    run_dir = _make_v5_run(tmp_path)
    df = _make_hourly_df()
    explanations = explain_forecast(df, run_dir=run_dir, use_lime=False)

    for exp in explanations:
        reconstructed = float(exp.base_value + exp.shap_values.sum())
        assert reconstructed == pytest.approx(exp.prediction, abs=1e-6)


def test_explain_forecast_with_lime(tmp_path):
    run_dir = _make_v5_run(tmp_path)
    df = _make_hourly_df()
    explanations = explain_forecast(df, run_dir=run_dir, use_lime=True)

    for exp in explanations:
        assert exp.lime_values is not None
        assert exp.lime_intercept is not None
        assert exp.lime_values.shape == exp.shap_values.shape
        # LIME is a local approximation; its intercept+sum should be finite.
        assert np.isfinite(exp.lime_values).all()
        assert np.isfinite(exp.lime_intercept)


def test_explain_forecast_raises_with_insufficient_history(tmp_path):
    run_dir = _make_v5_run(tmp_path)
    df = _make_hourly_df(n_days=3)
    with pytest.raises(Exception):
        explain_forecast(df, run_dir=run_dir, use_lime=False)