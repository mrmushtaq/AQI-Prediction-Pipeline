"""Streamlit dashboard: real-time next-3-day AQI forecasts for Sukkur.

Run with:
    streamlit run src/dashboard/app.py

Loads the most recent hourly engineered features (`data/processed/feature_df.csv`),
rebuilds the v5 daily features, scores the three daily-average Ridge models, and
displays the mean-AQI forecasts for the next 1, 2, and 3 days with color-coded
AQI categories and historical context.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Streamlit runs app.py as a script with only its own directory on sys.path,
# so add the project root (two levels up) to make `src` importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.logger import get_logger
from src.dashboard.explainability import explain_forecast
from src.dashboard.forecast import (
    DEFAULT_REGISTRY_DIR,
    HORIZONS,
    fetch_future_weather,
    make_forecast,
    load_daily_avg_models,
)
from src.training_pipeline.data_loader import DEFAULT_FEATURE_DF_PATH

logger = get_logger(__name__)

# US EPA AQI breakpoints for colour-coded health categories.
AQI_CATEGORIES = [
    (50, "Good", "#2ECC71"),
    (100, "Moderate", "#F1C40F"),
    (150, "Unhealthy for Sensitive Groups", "#E67E22"),
    (200, "Unhealthy", "#E74C3C"),
    (300, "Very Unhealthy", "#8E44AD"),
    (float("inf"), "Hazardous", "#641E16"),
]

CITY = "Sukkur"


@st.cache_data(show_spinner=False)
def load_latest_features() -> pd.DataFrame:
    """Load and sort the latest hourly engineered feature dataframe."""
    df = pd.read_csv(DEFAULT_FEATURE_DF_PATH)
    df["collection_timestamp"] = pd.to_datetime(df["collection_timestamp"])
    return df.sort_values("collection_timestamp").reset_index(drop=True)


def aqi_category(value: float) -> tuple[str, str]:
    """Return (label, hex colour) for a US EPA AQI value."""
    for upper, label, colour in AQI_CATEGORIES:
        if value <= upper:
            return label, colour
    return AQI_CATEGORIES[-1][1], AQI_CATEGORIES[-1][2]


def main() -> None:
    st.set_page_config(page_title="AQI Predictor — Sukkur", page_icon="🌬️", layout="wide")
    st.title(f"Air Quality Index Forecast — {CITY}")
    st.caption(
        "Daily-average AQI forecast for the next 3 days (Ridge models incl. future weather, "
        "R² ≈ 0.79–0.80). Data source: hourly observations in feature_df.csv + live OpenWeather forecast."
    )

    try:
        df = load_latest_features()
    except FileNotFoundError as exc:
        st.error(f"Missing feature data at {DEFAULT_FEATURE_DF_PATH}. Run feature engineering first.")
        logger.error("Dashboard failed to load features: %s", exc)
        return

    # Live future weather (OpenWeather 5-day forecast) drives the v6 features.
    # On any failure the forecast still works using archived weather where
    # available, so the dashboard never breaks on a transient API error.
    try:
        future_weather = fetch_future_weather(df)
        st.sidebar.success("Live future weather loaded (OpenWeather).")
    except Exception as exc:  # noqa: BLE001 -- forecast must not break on API hiccups
        future_weather = None
        st.sidebar.warning(
            "Future-weather fetch failed; using archived weather only: "
            f"{type(exc).__name__}. R² may be slightly lower."
        )
        logger.warning("Future-weather fetch failed: %s", exc)

    st.sidebar.header("About")
    st.sidebar.markdown(
        "- Forecasts the **mean AQI** over the next 1, 2, and 3 days.\n"
        "- Trained on ~2.5 years of hourly Sukkur data (2024–2026).\n"
        "- Models: Ridge regression on daily features incl. **future weather** (scaled).\n"
        "- Evaluation (20% chronological holdout): R² ≈ 0.79 / 0.79 / 0.80."
    )

    try:
        forecast = make_forecast(df, future_weather=future_weather)
        models, metadata = load_daily_avg_models()
    except Exception as exc:  # noqa: BLE001 -- surface any model/feature error to the user
        st.error(f"Could not generate forecast: {exc}")
        logger.exception("Dashboard forecast failed")
        return

    # Hazardous-level alert (project requirement).
    hazardous_threshold = 200  # US EPA: "Unhealthy" and above warrant an alert
    peak = max(forecast.day1_avg, forecast.day2_avg, forecast.day3_avg)
    if peak >= hazardous_threshold:
        st.error(
            f"⚠️ **Hazardous AQI alert** — peak predicted AQI of **{peak:.0f}** "
            f"within the next 3 days. Limit outdoor exposure and follow local "
            f"health guidance."
        )
    elif peak >= 150:
        st.warning(
            f"⚠️ **Elevated AQI alert** — peak predicted AQI of **{peak:.0f}** "
            f"within the next 3 days. Sensitive groups should reduce exertion."
        )
    else:
        st.success(f"✅ No hazardous AQI conditions predicted within the next 3 days (peak {peak:.0f}).")

    st.subheader("Next 3 Days — Mean AQI Forecast")
    st.markdown(f"Forecast as of **{forecast.forecast_date}** (latest fully-featured day).")

    cols = st.columns(3)
    labels = {"day1_avg": "Day 1", "day2_avg": "Day 2", "day3_avg": "Day 3"}
    for col, (label, _n_days) in zip(cols, HORIZONS):
        value = getattr(forecast, label)
        cat, colour = aqi_category(value)
        met = forecast.model_metadata.get(label, {})
        r2 = met.get("test_metrics", {}).get("r2")
        with col:
            st.markdown(
                f"<div style='background:{colour};border-radius:10px;padding:18px;color:#fff'>"
                f"<div style='font-size:18px'>{labels[label]}</div>"
                f"<div style='font-size:44px;font-weight:bold'>{value:.0f}</div>"
                f"<div style='font-size:16px'>{cat}</div>"
                f"<div style='font-size:13px;opacity:0.85'>test R² = {r2:.3f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.subheader("Historical Context")
    daily = df.set_index("collection_timestamp").resample("D")["aqi"].mean()
    recent = daily.tail(30)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(recent.index, recent.values, label="Observed daily AQI", color="#2980B9")
    horizon_dates = pd.date_range(forecast.forecast_date, periods=3, freq="D")
    forecast_values = [forecast.day1_avg, forecast.day2_avg, forecast.day3_avg]
    ax.plot(
        horizon_dates,
        forecast_values,
        label="Forecast (mean over next N days)",
        color="#E67E22",
        marker="o",
        linestyle="--",
    )
    ax.axvline(pd.Timestamp(forecast.forecast_date), color="#95A5A6", linestyle=":")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily AQI")
    ax.set_title(f"Recent daily AQI and 3-day forecast — {CITY}")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    with st.expander("Model details"):
        for label, _n_days in HORIZONS:
            met = forecast.model_metadata.get(label, {})
            st.markdown(
                f"**{labels[label]}** — {met.get('target', '')} "
                f"(alpha={met.get('alpha')}, n_train={met.get('n_train')}, n_test={met.get('n_test')}). "
                f"Test R²={met.get('test_metrics', {}).get('r2', float('nan')):.4f} "
                f"(naive persistence R²={met.get('naive_baseline', {}).get('r2', float('nan')):.4f})."
            )
        st.markdown(f"**Model registry:** `{forecast.run_dir}`")
        st.caption(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")

    st.subheader("Why did the model predict this? (SHAP / LIME)")
    try:
        explanations = explain_forecast(
            df, use_lime=st.checkbox("Also compute LIME (slower)", value=False),
            future_weather=future_weather,
        )
    except Exception as exc:  # noqa: BLE001 -- explanation is best-effort
        st.warning(f"Could not compute explanations: {exc}")
        logger.exception("Dashboard explanation failed")
        explanations = []

    if explanations:
        exp_cols = st.columns(3)
        for col, exp in zip(exp_cols, explanations):
            with col:
                st.markdown(f"**{labels[exp.label]}** — predicted **{exp.prediction:.1f}**")
                order = np.argsort(np.abs(exp.shap_values))[::-1]
                top_n = order[:8]
                fig_ex, ax_ex = plt.subplots(figsize=(4.2, 3.2))
                names = [exp.feature_cols[i] for i in top_n][::-1]
                vals = exp.shap_values[top_n][::-1]
                ax_ex.barh(names, vals, color=["#E74C3C" if v < 0 else "#2ECC71" for v in vals])
                ax_ex.axvline(0, color="k", lw=0.8)
                ax_ex.set_xlabel("SHAP contribution to AQI")
                ax_ex.set_title("Top contributing features")
                ax_ex.tick_params(axis="y", labelsize=8)
                st.pyplot(fig_ex)
                if exp.lime_values is not None:
                    lime_top = np.argsort(np.abs(exp.lime_values))[::-1][:5]
                    st.caption(
                        "LIME top: "
                        + ", ".join(
                            f"{exp.feature_cols[i]} ({exp.lime_values[i]:+.1f})" for i in lime_top
                        )
                    )
        with st.expander("SHAP methodology"):
            st.markdown(
                "SHAP values decompose the prediction into the model's base rate plus a "
                "per-feature contribution. A negative value (green) pushed AQI *down* "
                "from the base; a positive value (red) pushed it *up*. Because the "
                "models are linear, SHAP is exact and additive: base + Σ(SHAP) = prediction. "
                "LIME provides a complementary local linear approximation."
            )


if __name__ == "__main__":
    main()