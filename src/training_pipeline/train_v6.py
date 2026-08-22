"""Train the v6 daily-average AQI models (Ridge + future-weather features).

v6 extends the v5 recipe (46 daily features) with future-weather features
(`fw_wind/humid/temp/clouds/pressure` for the next 1..3 days).

--- IMPORTANT FIX (see inject_forecast_uncertainty) -----------------------
The original version of this script built fw_* directly from the archived
*actual* weather for the target days, for both train and test. That means
the reported test R^2 measured the model against perfect future weather,
which the live system never has (forecast.py uses a real OpenWeather
forecast at inference, which is imperfect and gets worse with horizon).
That mismatch silently inflates the backtest R^2 and invalidates the
v5-vs-v6 comparison.

This version injects calibrated Gaussian noise into fw_* (growing with
horizon) for BOTH train and test, so the model is trained and evaluated on
something resembling a real forecast. It also keeps the old "perfect
future weather" evaluation around, explicitly labeled
`test_metrics_perfect_future_weather_DO_NOT_REPORT`, purely so the size of
the original leakage is visible in metadata.json rather than asserted.

The noise std values in FORECAST_NOISE_STD are rough estimates from
general NWP forecast-skill literature (temperature RMSE roughly 1-2.5C
growing with horizon; wind/humidity/cloud error growing similarly;
pressure forecasts comparatively tight) -- NOT measured against this
project's actual OpenWeather error. Once forecast.py has logged enough
real forecast-vs-actual pairs in production, replace these with the
empirically measured std per variable/horizon for an exact calibration.
----------------------------------------------------------------------------

Run from the project root:

    python -m src.training_pipeline.train_v6

Models + scalers are saved per horizon to:
    data/model_registry/ridge_3day_dailyavg_v6_<UTC-timestamp>/
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from configs.config import PROJECT_ROOT
from src.common.logger import get_logger
from src.dashboard.forecast import (
    HORIZONS,
    V6_RUN_PREFIX,
    build_daily_features,
)

import argparse

from src.training_pipeline.data_loader import DEFAULT_FEATURE_DF_PATH, load_features

logger = get_logger(__name__)

ALPHAS = {"day1_avg": 100, "day2_avg": 300, "day3_avg": 300}
TEST_FRACTION = 0.2
RANDOM_SEED = 42

# Train on the same historical endpoint the v5 models used, so the v6 holdout
# is evaluated on the identical chronological window and the comparison to v5
# (and the R2 >= 0.75 requirement) is apples-to-apples. The live dashboard
# still forecasts from whatever the latest features are.
TRAIN_END_DATE = "2026-07-27 23:00:00+00:00"

# --- Simulated forecast uncertainty -----------------------------------
# (base_std_at_day1, growth_per_extra_day) per future-weather variable.
# Applied as std = base_std + growth * (h - 1) for horizon day h.
# Units follow whatever build_daily_features uses for that column
# (e.g. wind in m/s, temp in C, humid/clouds in %, pressure in hPa).
# TODO: replace with empirically measured OpenWeather error once logged.
FORECAST_NOISE_STD = {
    "wind": (0.8, 0.4),
    "humid": (5.0, 3.0),
    "temp": (1.0, 0.5),
    "clouds": (10.0, 5.0),
    "pressure": (1.0, 0.5),
}
PERCENT_BOUNDED_VARS = {"humid", "clouds"}  # clip to [0, 100] after noise


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE and R² for a held-out evaluation."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def time_based_split(
    data: pd.DataFrame, test_fraction: float = TEST_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological 80/20 split keeping the most recent rows as the test set."""
    n_test = int(len(data) * test_fraction)
    return data.iloc[:-n_test], data.iloc[-n_test:]


def make_target(daily: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """Target = mean AQI over the next `n_days` days (the forecast horizon)."""
    dd = daily.copy()
    dd["target"] = dd["aqi"].shift(-n_days).rolling(n_days).mean()
    return dd


def inject_forecast_uncertainty(
    daily: pd.DataFrame, max_horizon: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Simulate realistic forecast error on the fw_* (future-weather) columns.

    fw_{var}{h} is built from ARCHIVED ACTUAL weather for day h -- perfect
    information the model never has in production. This adds calibrated,
    horizon-growing Gaussian noise per variable so train and test both see
    something resembling a real forecast instead of ground truth.
    """
    dd = daily.copy()
    for var, (base_std, growth) in FORECAST_NOISE_STD.items():
        for h in range(1, max_horizon + 1):
            col = f"fw_{var}{h}"
            if col not in dd.columns:
                continue
            std = base_std + growth * (h - 1)
            noise = rng.normal(loc=0.0, scale=std, size=len(dd))
            dd[col] = dd[col] + noise
            if var in PERCENT_BOUNDED_VARS:
                dd[col] = dd[col].clip(0, 100)
    return dd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", choices=["local", "hopsworks"], default="local",
        help="Where to load engineered features from (default: local CSV)",
    )
    args = parser.parse_args()

    df = load_features(source=args.source, local_path=DEFAULT_FEATURE_DF_PATH)
    df["collection_timestamp"] = pd.to_datetime(df["collection_timestamp"])
    logger.info("Loaded %d hourly rows (source=%s)", len(df), args.source)
    df = df[df["collection_timestamp"] <= TRAIN_END_DATE].reset_index(drop=True)
    logger.info(
        "Truncated to training endpoint %s (%d rows) for a v5-comparable holdout",
        TRAIN_END_DATE,
        len(df),
    )

    daily, feature_cols = build_daily_features(df)
    logger.info("Built %d daily rows with %d candidate features", len(daily), len(feature_cols))

    max_horizon = max(n_days for _, n_days in HORIZONS)
    rng = np.random.default_rng(RANDOM_SEED)
    daily_realistic = inject_forecast_uncertainty(daily, max_horizon, rng)
    logger.info(
        "Injected simulated forecast uncertainty into fw_* (max horizon=%d) so "
        "train/test reflect realistic forecast error instead of archived actuals",
        max_horizon,
    )

    out_dir = PROJECT_ROOT / "data" / "model_registry" / (
        f"{V6_RUN_PREFIX}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    os.makedirs(out_dir, exist_ok=True)

    summary: dict = {}
    for label, n_days in HORIZONS:
        # Each horizon uses the historical (non-future-weather) features plus
        # the future-weather columns for exactly the days it predicts
        # (day1 -> only fw_*1, day2 -> fw_*1..2, day3 -> fw_*1..3).
        fw_cols = [
            f"fw_{var}{h}"
            for h in range(1, n_days + 1)
            for var in ["wind", "humid", "temp", "clouds", "pressure"]
        ]
        hist_cols = [c for c in feature_cols if not c.startswith("fw_")]
        model_cols = hist_cols + fw_cols
        model_cols = [c for c in model_cols if c in daily.columns]

        # "Realistic" = noisy future weather (train AND evaluate on this).
        # "Leaky" = original perfect archived future weather, kept only to
        # measure the size of the original leak -- never used to fit anything.
        dd_real = make_target(daily_realistic, n_days)
        dd_real = dd_real.dropna(subset=model_cols + fw_cols + ["target"]).reset_index(drop=True)

        dd_leaky = make_target(daily, n_days)
        dd_leaky = dd_leaky.dropna(subset=model_cols + fw_cols + ["target"]).reset_index(drop=True)

        assert len(dd_real) == len(dd_leaky), (
            "Realistic/leaky row counts diverged -- noise injection should "
            "never change which rows survive dropna(). Investigate before trusting metrics."
        )

        tr, te = time_based_split(dd_real)
        _, te_leaky = time_based_split(dd_leaky)

        scaler = StandardScaler().fit(tr[model_cols])
        model = Ridge(alpha=ALPHAS[label]).fit(scaler.transform(tr[model_cols]), tr["target"])

        # Honest, deployable-comparable metrics: evaluated on simulated
        # realistic forecast error, same as production will actually see.
        pred = model.predict(scaler.transform(te[model_cols]))
        metrics = compute_metrics(te["target"], pred)

        # Diagnostic only -- reproduces the ORIGINAL (leaky) evaluation with
        # this same model, so the leakage gap is visible, not asserted.
        pred_leaky_eval = model.predict(scaler.transform(te_leaky[model_cols]))
        metrics_leaky_eval = compute_metrics(te_leaky["target"], pred_leaky_eval)

        naive = compute_metrics(te["target"], te["aqi"])

        joblib.dump(model, out_dir / f"{label}_model.joblib")
        joblib.dump(scaler, out_dir / f"{label}_scaler.joblib")

        summary[label] = {
            "version": "v6",
            "n_days": n_days,
            "alpha": ALPHAS[label],
            "feature_cols": model_cols,
            "n_train": len(tr),
            "n_test": len(te),
            "test_start": te["collection_timestamp"].min().strftime("%Y-%m-%d"),
            "test_end": te["collection_timestamp"].max().strftime("%Y-%m-%d"),
            "target": f"mean AQI over next {n_days} day(s)",
            "naive_baseline": naive,
            "test_metrics": metrics,  # <-- REPORT THIS ONE
            "test_metrics_perfect_future_weather_DO_NOT_REPORT": metrics_leaky_eval,
            "leakage_r2_gap": round(metrics_leaky_eval["r2"] - metrics["r2"], 4),
        }
        logger.info(
            "%s: realistic R2=%.4f RMSE=%.2f | perfect-future-weather R2=%.4f "
            "(leakage gap=%.4f) | naive R2=%.4f | test %s..%s, %d features",
            label,
            metrics["r2"],
            metrics["rmse"],
            metrics_leaky_eval["r2"],
            summary[label]["leakage_r2_gap"],
            naive["r2"],
            summary[label]["test_start"],
            summary[label]["test_end"],
            len(model_cols),
        )

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved v6 models to %s", out_dir)
    print(f"\nSaved v6 models to {out_dir}")


if __name__ == "__main__":
    main()