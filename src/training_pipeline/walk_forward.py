"""
Expanding-window (walk-forward) evaluation for the training pipeline.

A single chronological train/validation/test split is a poor measure of
forecast skill for this dataset: the most-recent 15% of the record can land
entirely inside a single low-variance seasonal regime (e.g. the summer months
in Sukkur), where even a perfect forecast looks bad under R². Walk-forward
evaluation retrains the model on an expanding window of *all* past data and
predicts each subsequent block, mimicking real deployment (retrain on
everything known so far, then forecast). The accumulated out-of-sample
predictions therefore span every seasonal regime in the record and give an
honest, stable estimate of the model's forecasting skill.

Usage::

    metrics = run_walk_forward(
        model=best_estimator,       # fitted; cloned + refit on each window
        df=feature_df,
        feature_cols=feature_cols,
        target_col="aqi_target",
        step=336,                   # predict 2 weeks per window
        min_train=1500,             # first window needs this many rows
    )
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.common.logger import get_logger
from src.training_pipeline.evaluate import compute_metrics
from src.training_pipeline.models import PersistenceBaselineRegressor

logger = get_logger(__name__)


def run_walk_forward(
    model: Any,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "aqi_target",
    step: int = 336,
    min_train: int = 1500,
) -> dict[str, Any]:
    """Evaluate a model with expanding-window (walk-forward) backtesting.

    The DataFrame is sorted chronologically and split into contiguous blocks
    of ``step`` rows. For each block, a clone of ``model`` is fitted on every
    row before the block (expanding window) and used to predict the block.
    Predictions from all blocks are concatenated and scored at the end, so
    every prediction is strictly out-of-sample relative to its training data.

    Args:
        model: A fitted estimator to backtest (a clone is refit per window,
            so the passed instance is never mutated).
        df: Feature DataFrame containing ``feature_cols``, ``target_col``,
            and a ``collection_timestamp`` column (for stable ordering).
        feature_cols: Column names to use as features.
        target_col: Name of the target column.
        step: Number of rows per validation block.
        min_train: Minimum number of rows in the first (expanding) window.
            Rows before the first window are used for training only, not
            scored. If fewer than ``min_train + step`` rows exist, the
            evaluation degrades to a single block.

    Returns:
        Dict with ``"metrics"`` (RMSE/MAE/R²/sMAPE over all out-of-sample
        predictions), ``"n"`` (number of scored rows), and ``"n_windows"``
        (number of validation blocks scored).
    """
    df = df.sort_values("collection_timestamp").reset_index(drop=True)
    subset = df[list(feature_cols) + [target_col]].dropna().reset_index(drop=True)
    if len(subset) < min_train + 1:
        logger.warning(
            "Walk-forward: only %d complete rows (< min_train=%d + 1); "
            "skipping walk-forward evaluation",
            len(subset), min_train,
        )
        return {"metrics": None, "n": 0, "n_windows": 0}

    X = subset[list(feature_cols)]
    y = subset[target_col].to_numpy()

    preds: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    t = min_train
    n_windows = 0
    while t < len(y):
        window = min(step, len(y) - t)
        X_train, y_train = X.iloc[:t], y[:t]
        X_pred, y_pred = X.iloc[t : t + window], y[t : t + window]

        estimator = clone(model)
        estimator.fit(X_train, y_train)
        preds.append(estimator.predict(X_pred))
        ys.append(y_pred)

        t += window
        n_windows += 1

    if not preds:
        return {"metrics": None, "n": 0, "n_windows": 0}

    all_pred = np.concatenate(preds)
    all_true = np.concatenate(ys)
    metrics = compute_metrics(all_true, all_pred)
    logger.info(
        "Walk-forward (%s): RMSE=%.3f MAE=%.3f R2=%.3f sMAPE=%.3f "
        "over %d rows / %d windows",
        getattr(model, "__class__").__name__,
        metrics["rmse"], metrics["mae"], metrics["r2"], metrics["smape"],
        len(all_true), n_windows,
    )
    return {"metrics": metrics, "n": int(len(all_true)), "n_windows": n_windows}


def run_walk_forward_persistence(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "aqi_target",
    aqi_column: str = "aqi",
    step: int = 336,
    min_train: int = 1500,
) -> dict[str, Any]:
    """Walk-forward evaluation of the persistence baseline.

    Predicts the target (``horizon`` steps ahead) using the current ``aqi``
    column for every row, mirroring ``run_walk_forward`` so the numbers are
    directly comparable. The persistence model has no learned parameters, so
    no per-window refitting is needed -- the prediction for each block is
    simply the ``aqi`` value at the start of that block's feature window.

    Args:
        df: Feature DataFrame containing ``feature_cols``, ``target_col``,
            and ``aqi_column``.
        feature_cols: Column names to use as features (kept for API parity
            with `run_walk_forward`; only ``aqi_column`` is actually read).
        target_col: Name of the target column.
        aqi_column: Name of the current-AQI column used for persistence.
        step: Number of rows per validation block.
        min_train: Minimum number of rows before the first scored block.

    Returns:
        Dict with ``"metrics"``, ``"n"``, and ``"n_windows"`` (same shape as
        `run_walk_forward`).
    """
    df = df.sort_values("collection_timestamp").reset_index(drop=True)
    cols = list(dict.fromkeys(list(feature_cols) + [target_col] + [aqi_column]))
    subset = df[cols].dropna().reset_index(drop=True)
    if len(subset) < min_train + 1:
        return {"metrics": None, "n": 0, "n_windows": 0}

    y = subset[target_col].to_numpy()
    aqi = subset[aqi_column].to_numpy()

    baseline = PersistenceBaselineRegressor(aqi_column=aqi_column)
    preds: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    t = min_train
    n_windows = 0
    while t < len(y):
        window = min(step, len(y) - t)
        # The persistence baseline resolves the current-AQI column by name,
        # so it must be present in the frame passed to predict() even when it
        # is not part of ``feature_cols``.
        X_pred = subset.iloc[t : t + window][
            list(dict.fromkeys(list(feature_cols) + [aqi_column]))
        ]
        preds.append(baseline.predict(X_pred))
        ys.append(y[t : t + window])
        t += window
        n_windows += 1

    if not preds:
        return {"metrics": None, "n": 0, "n_windows": 0}

    all_pred = np.concatenate(preds)
    all_true = np.concatenate(ys)
    metrics = compute_metrics(all_true, all_pred)
    logger.info(
        "Walk-forward persistence: RMSE=%.3f MAE=%.3f R2=%.3f sMAPE=%.3f "
        "over %d rows / %d windows",
        metrics["rmse"], metrics["mae"], metrics["r2"], metrics["smape"],
        len(all_true), n_windows,
    )
    return {"metrics": metrics, "n": int(len(all_true)), "n_windows": n_windows}
