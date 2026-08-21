"""
Horizon sweep: how predictable is AQI at 6/12/24/48/72 hours ahead?

Trains a properly-tuned LightGBM (RandomizedSearchCV over a
TimeSeriesSplit with gap = horizon, exactly as in production) for each
forecast horizon on the same chronological 70/15/15 split, and compares it
with naive baselines. This answers whether 72-hour forecasting is
fundamentally harder than shorter horizons, and gives the R² ceiling that a
strong model can actually reach on the held-out test set.

Also records volatility drift over time (monthly hourly-change std and
24h-rolling-std), to quantify the regime shift seen in the diagnostics.

Results are written to data/diagnostics/horizon_sweep.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from configs.config import PROJECT_ROOT
from src.common.logger import get_logger
from src.feature_pipeline.feature_engineering import build_features
from src.training_pipeline.evaluate import compute_metrics
from src.training_pipeline.preprocessing import (
    TARGET_COLUMN,
    select_feature_columns,
    time_based_train_val_test_split,
)

logger = get_logger(__name__)

RAW_CSV = PROJECT_ROOT / "data" / "processed" / "aqi_dataset_historical.csv"
OUT_PATH = PROJECT_ROOT / "data" / "diagnostics" / "horizon_sweep.json"
HORIZONS = (6, 12, 24, 48, 72)
N_ITER = 40
N_SPLITS = 5
RANDOM_STATE = 42


def build_shared_frame() -> pd.DataFrame:
    """Build the full feature set once; targets are derived per horizon."""
    df = build_features(RAW_CSV, horizon_steps=1, drop_incomplete_rows=False)
    feature_cols = select_feature_columns(df)
    logger.info("Using %d features", len(feature_cols))
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df, feature_cols


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE/MAE/R²/sMAPE; delegates to `evaluate.compute_metrics`."""
    return compute_metrics(y_true, y_pred)


def naive_baselines(
    aqi: pd.Series, month: pd.Series, y_true: pd.Series, train_mask: pd.Series
) -> dict[str, dict[str, float]]:
    month_means = aqi[train_mask].groupby(month[train_mask]).mean()
    preds = {
        "persistence": aqi,
        "roll_mean_24h": aqi.rolling(24, min_periods=1).mean(),
        "roll_mean_7d": aqi.rolling(168, min_periods=1).mean(),
        "seasonal_month": month_means.reindex(month),
    }
    out = {}
    # All series here are positionally aligned (reset index 0..n-1), and
    # ``y_true`` is a contiguous slice of that range, so select by position:
    # label-based ``.loc`` would break for ``seasonal_month``, whose index
    # holds month values rather than row positions.
    idx = y_true.index.to_numpy()
    for name, p in preds.items():
        out[name] = metrics(y_true.to_numpy(), p.iloc[idx].to_numpy(dtype="float64"))
    return out


def run_horizon(df: pd.DataFrame, feature_cols: list[str], horizon: int) -> dict:
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

    y = df["aqi"].shift(-horizon)
    valid = y.notna()
    X = df.loc[valid, feature_cols].reset_index(drop=True)
    y = y[valid].reset_index(drop=True)

    train_df, val_df, test_df = time_based_train_val_test_split(
        df.loc[valid].reset_index(drop=True), 0.7, 0.15, 0.15
    )
    # X and y are already aligned with the valid rows of df; slice by row count.
    n = len(X)
    tr_end = int(n * 0.7)
    val_end = int(n * 0.85)
    X_tr, y_tr = X.iloc[:tr_end], y.iloc[:tr_end]
    X_va, y_va = X.iloc[tr_end:val_end], y.iloc[tr_end:val_end]
    X_te, y_te = X.iloc[val_end:], y.iloc[val_end:]

    base = LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    params = {
        "num_leaves": [16, 31, 64, 128],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "n_estimators": [200, 400, 800],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_samples": [5, 20, 50],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [0.0, 0.1, 1.0],
    }
    cv = TimeSeriesSplit(n_splits=N_SPLITS, gap=horizon)
    search = RandomizedSearchCV(
        base, params, n_iter=N_ITER, cv=cv, scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE, n_jobs=1, verbose=0,
    )
    search.fit(X_tr, y_tr.to_numpy())
    best = search.best_estimator_
    best.fit(
        pd.concat([X_tr, X_va], ignore_index=True),
        pd.concat([y_tr, y_va], ignore_index=True).to_numpy(),
    )
    lgbm_test = metrics(y_te.to_numpy(), best.predict(X_te))

    return {
        "horizon_hours": horizon,
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "n_test": int(len(X_te)),
        "target": {
            "test_mean": float(y_te.mean()),
            "test_std": float(y_te.std(ddof=0)),
        },
        "lgbm": {
            "best_cv_rmse": float(-search.best_score_),
            "best_params": search.best_params_,
            "test": lgbm_test,
        },
        "naive": naive_baselines(
            df.loc[valid, "aqi"].reset_index(drop=True),
            df.loc[valid, "month"].reset_index(drop=True),
            y_te,
            np.arange(len(X_tr)),
        ),
    }


def volatility_drift(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["change"] = df["aqi"].diff()
    df["roll_std_24h"] = df["aqi"].rolling(24, min_periods=1).std()
    rows = []
    # strftime keeps the timestamp's tz (unlike to_period) and yields a
    # sortable "YYYY-MM" label for the monthly buckets.
    for ym, g in df.groupby(df["collection_timestamp"].dt.strftime("%Y-%m")):
        rows.append(
            {
                "month": str(ym),
                "n": int(len(g)),
                "aqi_mean": float(g["aqi"].mean()),
                "change_std": float(g["change"].std(ddof=0)),
                "roll_std_24h_mean": float(g["roll_std_24h"].mean()),
            }
        )
    return rows


def main() -> int:
    df, feature_cols = build_shared_frame()
    logger.info("Shared frame: %d rows x %d features", len(df), len(feature_cols))

    results = {"features": feature_cols, "volatility_drift": volatility_drift(df), "horizons": {}}
    for h in HORIZONS:
        logger.info("=== Horizon %dh ===", h)
        results["horizons"][str(h)] = run_horizon(df, feature_cols, h)
        logger.info("horizon %dh done", h)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Horizon sweep written to %s", OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
