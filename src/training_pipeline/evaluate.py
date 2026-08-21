"""
Evaluation utilities for the model training pipeline (Phase 5).

Computes the four metrics used to compare models -- RMSE, MAE, R², and
sMAPE -- and builds a sorted comparison table across model candidates plus
the persistence baseline.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.common.logger import get_logger

logger = get_logger(__name__)

METRIC_COLUMNS = ("rmse", "mae", "r2", "smape")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, R², and sMAPE for a set of predictions.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.

    Returns:
        Dict with keys `"rmse"`, `"mae"`, `"r2"`, `"smape"`.
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    smape = _compute_smape(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2, "smape": smape}


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Predict with `model` on `X_test` and compute metrics against `y_test`.

    Args:
        model: A fitted model implementing `predict(X)`.
        X_test: Test feature matrix.
        y_test: Test target vector.

    Returns:
        Dict with keys `"rmse"`, `"mae"`, `"r2"`, `"smape"`.
    """
    y_pred = model.predict(X_test)
    return compute_metrics(y_test, y_pred)


def build_comparison_table(results: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    """Build a comparison table across model candidates, sorted by RMSE.

    Args:
        results: Mapping of model name -> metrics dict (as returned by
            `evaluate_model`).

    Returns:
        DataFrame with columns `model`, `rmse`, `mae`, `r2`, `smape`,
        sorted ascending by RMSE (best model first).
    """
    rows = [{"model": name, **metrics} for name, metrics in results.items()]
    table = pd.DataFrame(rows).sort_values("rmse", ascending=True).reset_index(drop=True)
    return table


def select_best_model(results: Mapping[str, Mapping[str, float]]) -> str:
    """Select the best-performing model name by lowest RMSE.

    Args:
        results: Mapping of model name -> metrics dict.

    Returns:
        The name of the model with the lowest RMSE.

    Raises:
        ValueError: If `results` is empty.
    """
    if not results:
        raise ValueError("Cannot select a best model from an empty results mapping.")
    return min(results, key=lambda name: results[name]["rmse"])


def _compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error, in percent.

    sMAPE = 100 * mean(2 * |y_true - y_pred| / (|y_true| + |y_pred|)).
    Observations where both values are zero are skipped to avoid dividing by
    zero; individual ratios where the denominator is zero are treated as
    zero error (both terms zero, so the forecast is exact).

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.

    Returns:
        The sMAPE percentage (0 = perfect, higher = worse).
    """
    denominator = np.abs(y_true) + np.abs(y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = 2.0 * np.abs(y_true - y_pred) / denominator
    ratio = np.where(denominator == 0, 0.0, ratio)
    return float(np.mean(ratio) * 100.0)
