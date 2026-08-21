"""
Diagnosis of why the 72-hour AQI forecast achieves a low test R².

This is a pure analysis script (no model training beyond trivial naive
baselines). It checks, in order:

    1. Basic data shape / quality.
    2. Seasonality & level (monthly AQI profile) -- whether the series is
       non-stationary / driven by a strong seasonal regime.
    3. Autocorrelation (ACF) of hourly AQI -- how far into the future the
       series remains predictable by its own history.
    4. Distribution shift between the chronological train / val / test
       splits (means, stds, and two-sample KS p-values for key features).
    5. Target-variance ceiling: R² is bounded by the noise floor of the
       target; if the target's variance is small relative to irreducible
       noise, even a perfect model cannot reach high R².
    6. Naive-baseline ceilings at horizons 6/12/24/48/72h -- persistence,
       rolling-mean, and seasonal-month baselines -- to estimate the
       practical predictability ceiling at each horizon.

Everything is computed on the actual, chronologically-ordered dataset and
written to data/diagnostics/report.json (plus printed).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from configs.config import PROJECT_ROOT
from src.common.logger import get_logger

logger = get_logger(__name__)

FEATURE_DF = PROJECT_ROOT / "data" / "processed" / "feature_df.csv"
OUT_DIR = PROJECT_ROOT / "data" / "diagnostics"

HORIZONS_HOURS = (6, 12, 24, 48, 72)
SPLITS = (0.7, 0.15, 0.15)  # train / val / test


def load_data() -> pd.DataFrame:
    df = pd.read_csv(FEATURE_DF, parse_dates=["collection_timestamp"])
    df = df.sort_values("collection_timestamp").reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame):
    n = len(df)
    train_end = int(n * SPLITS[0])
    val_end = int(n * (SPLITS[0] + SPLITS[1]))
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


def aqi_acf(aqi: pd.Series, nlags: int) -> list[float]:
    """Autocorrelation of the AQI series at hourly lags 1..nlags."""
    centered = aqi.to_numpy(dtype="float64")
    centered = centered - centered.mean()
    var = np.dot(centered, centered) or 1.0
    out = []
    for lag in range(1, nlags + 1):
        ac = np.dot(centered[:-lag], centered[lag:]) / var
        out.append(float(ac))
    return out


def target_stats_at_horizon(df: pd.DataFrame, horizon: int) -> dict:
    """Target variance decomposition on the test split for a horizon."""
    y = df["aqi"].shift(-horizon)
    test_idx = int(len(df) * (SPLITS[0] + SPLITS[1]))
    y_test = y.iloc[test_idx:].dropna()
    # Naive noise ceiling: how much variance a 'seasonal-mean' predictor explains.
    train = df.iloc[:test_idx]
    y_train = y.iloc[:test_idx]
    y_test_aligned = y_test
    # Month-mean baseline fitted on train only.
    month_means = y_train.groupby(df.iloc[:test_idx]["month"]).mean()
    month_of_test = df.iloc[test_idx:]["month"].loc[y_test.index]
    baseline_pred = month_means.reindex(month_of_test).to_numpy()
    y_true = y_test_aligned.to_numpy()
    ss_res = float(np.sum((y_true - baseline_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    month_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "n": int(len(y_test)),
        "mean": float(y_true.mean()),
        "std": float(y_true.std(ddof=0)),
        "var": float(y_true.var(ddof=0)),
        "month_mean_baseline_r2": float(month_r2),
        "month_mean_baseline_rmse": float(np.sqrt(ss_res / len(y_true))),
    }


def baseline_eval(df: pd.DataFrame, horizon: int) -> dict:
    """Evaluate naive baselines on the test split at a given horizon.

    Baselines (all causal, using only information available at prediction
    time t, forecasting aqi[t+horizon]):
        - persistence: aqi[t]
        - rolling mean over the past 24h at t
        - seasonal month-mean (fit on train+val only)
        - 7-day mean at t
    """
    y = df["aqi"].shift(-horizon)
    test_idx = int(len(df) * (SPLITS[0] + SPLITS[1]))

    train_slice = df.iloc[:test_idx]
    month_means = (
        y.iloc[:test_idx].groupby(train_slice["month"]).mean()
    )

    preds = pd.DataFrame(index=y.index)
    preds["persistence"] = df["aqi"]
    preds["roll_mean_24h"] = df["aqi"].rolling(24, min_periods=1).mean()
    preds["roll_mean_7d"] = df["aqi"].rolling(168, min_periods=1).mean()
    preds["seasonal_month"] = month_means.reindex(df["month"]).to_numpy()

    y_test = y.iloc[test_idx:].dropna()
    results = {}
    for name in preds.columns:
        p = preds[name].loc[y_test.index].to_numpy(dtype="float64")
        t = y_test.to_numpy(dtype="float64")
        err = t - p
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        r2 = 1.0 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 0 else np.nan
        smape = float(
            100 * np.mean(2 * np.abs(err) / (np.abs(t) + np.abs(p)))
        )
        results[name] = {"rmse": rmse, "mae": mae, "r2": r2, "smape": smape}
    return results


def distribution_shift_report(df: pd.DataFrame) -> list[dict]:
    """Compare train vs test feature distributions (mean/std + KS p-value)."""
    train_end = int(len(df) * SPLITS[0])
    test_start = int(len(df) * (SPLITS[0] + SPLITS[1]))
    train = df.iloc[:train_end]
    test = df.iloc[test_start:]

    report = []
    for col in df.columns:
        if df[col].dtype.kind not in "fi":
            continue
        a = train[col].dropna()
        b = test[col].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        ks, pval = sps.ks_2samp(a, b)
        report.append(
            {
                "column": col,
                "train_mean": float(a.mean()),
                "test_mean": float(b.mean()),
                "train_std": float(a.std(ddof=0)),
                "test_std": float(b.std(ddof=0)),
                "ks_stat": float(ks),
                "ks_pvalue": float(pval),
            }
        )
    report.sort(key=lambda r: r["ks_stat"], reverse=True)
    return report


def main() -> int:
    df = load_data()
    logger.info("Loaded %d rows, %d cols", len(df), len(df.columns))
    report: dict = {}

    # 1. Basic shape
    report["n_rows"] = int(len(df))
    report["date_start"] = str(df["collection_timestamp"].min())
    report["date_end"] = str(df["collection_timestamp"].max())

    # 2. Seasonality
    monthly = df.groupby(df["collection_timestamp"].dt.month)["aqi"].agg(
        ["mean", "std", "count"]
    )
    report["monthly_aqi"] = {
        int(m): {"mean": float(r["mean"]), "std": float(r["std"]), "n": int(r["count"])}
        for m, r in monthly.iterrows()
    }

    # 3. ACF of AQI
    report["acf_hourly"] = {
        "lags_hours": list(range(1, 73)),
        "values": aqi_acf(df["aqi"], 72),
    }

    # 4. Distribution shift across splits
    train, val, test = chronological_split(df)
    report["split_stats"] = {
        "train": {"n": len(train), "aqi_mean": float(train["aqi"].mean()),
                  "aqi_std": float(train["aqi"].std(ddof=0)),
                  "range": [float(train["aqi"].min()), float(train["aqi"].max())]},
        "val": {"n": len(val), "aqi_mean": float(val["aqi"].mean()),
                "aqi_std": float(val["aqi"].std(ddof=0)),
                "range": [float(val["aqi"].min()), float(val["aqi"].max())]},
        "test": {"n": len(test), "aqi_mean": float(test["aqi"].mean()),
                 "aqi_std": float(test["aqi"].std(ddof=0)),
                 "range": [float(test["aqi"].min()), float(test["aqi"].max())]},
    }
    report["distribution_shift"] = distribution_shift_report(df)[:25]

    # 5 & 6. Baseline ceilings + target variance per horizon
    report["horizons"] = {}
    for h in HORIZONS_HOURS:
        report["horizons"][str(h)] = {
            "target": target_stats_at_horizon(df, h),
            "baselines": baseline_eval(df, h),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Diagnosis report written to %s", out_path)

    print(json.dumps(report, indent=2, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
