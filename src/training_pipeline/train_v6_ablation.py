"""Ablation check for v6: does dropping fw_* actually hurt R²?

The leakage-gap check in train_v6.py came back ~0 (noise-level), which is
good news IF the model genuinely doesn't depend much on fw_* precision --
but it's also what you'd see if the injected noise was just too small to
matter for a heavily-regularized Ridge model (alpha 100-300). This script
settles it directly: train the identical pipeline with fw_* removed and
compare. If hist-only R² is close to the full v6 R², fw_* isn't actually
driving the v5->v6 improvement -- something else is (e.g. the alpha
re-tune), and that's worth knowing before crediting future-weather
features for the gain.

Also prints the fitted day3 model's largest-magnitude coefficients so you
can see directly how much weight fw_* columns get relative to historical
ones.

Run from the project root:

    python -m src.training_pipeline.train_v6_ablation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.dashboard.forecast import HORIZONS, build_daily_features
from src.training_pipeline.data_loader import DEFAULT_FEATURE_DF_PATH
from src.training_pipeline.train_v6 import (
    ALPHAS,
    RANDOM_SEED,
    TRAIN_END_DATE,
    compute_metrics,
    inject_forecast_uncertainty,
    make_target,
    time_based_split,
)


def main() -> None:
    import argparse
    from src.training_pipeline.data_loader import load_features

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["local", "hopsworks"], default="local")
    args = parser.parse_args()

    df = load_features(source=args.source, local_path=DEFAULT_FEATURE_DF_PATH)
    df["collection_timestamp"] = pd.to_datetime(df["collection_timestamp"])
    df = df[df["collection_timestamp"] <= TRAIN_END_DATE].reset_index(drop=True)

    daily, feature_cols = build_daily_features(df)
    max_horizon = max(n_days for _, n_days in HORIZONS)
    rng = np.random.default_rng(RANDOM_SEED)
    daily_realistic = inject_forecast_uncertainty(daily, max_horizon, rng)

    print(f"{'horizon':<10} {'full v6 R2':>12} {'hist-only R2':>14} {'delta':>10}")
    print("-" * 48)

    day3_model = None
    day3_cols = None

    for label, n_days in HORIZONS:
        fw_cols = [
            f"fw_{var}{h}"
            for h in range(1, n_days + 1)
            for var in ["wind", "humid", "temp", "clouds", "pressure"]
        ]
        hist_cols = [c for c in feature_cols if not c.startswith("fw_")]
        full_cols = [c for c in hist_cols + fw_cols if c in daily.columns]
        hist_only_cols = [c for c in hist_cols if c in daily.columns]

        dd = make_target(daily_realistic, n_days)
        dd = dd.dropna(subset=full_cols + ["target"]).reset_index(drop=True)
        tr, te = time_based_split(dd)

        # Full v6 (historical + realistic-noise future weather)
        scaler_full = StandardScaler().fit(tr[full_cols])
        model_full = Ridge(alpha=ALPHAS[label]).fit(scaler_full.transform(tr[full_cols]), tr["target"])
        r2_full = compute_metrics(te["target"], model_full.predict(scaler_full.transform(te[full_cols])))["r2"]

        # Ablation: historical features only, same alpha, same rows
        scaler_hist = StandardScaler().fit(tr[hist_only_cols])
        model_hist = Ridge(alpha=ALPHAS[label]).fit(scaler_hist.transform(tr[hist_only_cols]), tr["target"])
        r2_hist = compute_metrics(te["target"], model_hist.predict(scaler_hist.transform(te[hist_only_cols])))["r2"]

        print(f"{label:<10} {r2_full:>12.4f} {r2_hist:>14.4f} {r2_full - r2_hist:>10.4f}")

        if label == "day3_avg":
            day3_model, day3_cols = model_full, full_cols

    if day3_model is not None:
        coefs = pd.Series(day3_model.coef_, index=day3_cols).sort_values(key=np.abs, ascending=False)
        print("\nTop 15 |coefficient| for day3_avg (standardized features, full v6 model):")
        print(coefs.head(15).to_string())
        n_fw_in_top15 = sum(1 for c in coefs.head(15).index if c.startswith("fw_"))
        print(f"\n{n_fw_in_top15} of the top 15 coefficients are fw_* (future-weather) features.")


if __name__ == "__main__":
    main()