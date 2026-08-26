"""Production dashboard for the real Sukkur AQI Predictor artifacts."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.alerts import AQI_BANDS, alert_level, classify_aqi, trend_delta
from src.dashboard.data_service import dataset_quality, latest_metadata, load_historical_data, load_hopsworks_data, model_runs, source_summary
from src.dashboard.explainability import explain_forecast
from src.dashboard.forecast import HORIZONS, fetch_future_weather, load_daily_avg_models, make_forecast
from src.dashboard.styles import apply_styles


@st.cache_data(ttl=300, show_spinner=False)
def load_data() -> pd.DataFrame:
    try:
        frame = load_hopsworks_data()
    except Exception:
        frame, _ = load_historical_data()
    frame.attrs["live_status"] = "unavailable"
    try:
        from src.feature_pipeline.live_pipeline import run_live_pipeline

        current = pd.DataFrame([run_live_pipeline()])
        current["collection_timestamp"] = pd.to_datetime(current["collection_timestamp"], utc=True, errors="coerce")
        frame = pd.concat([frame, current], ignore_index=True)
        frame = frame.drop_duplicates(subset=["city", "collection_timestamp"], keep="last")
        frame = frame.sort_values("collection_timestamp").reset_index(drop=True)
        frame.attrs["live_status"] = "success"
    except Exception:
        frame.attrs["live_status"] = "failed"
    return frame


@st.cache_data(ttl=300, show_spinner=False)
def load_models() -> tuple[dict, dict]:
    return load_daily_avg_models()


def number(value: object, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "Data unavailable"
    return f"{float(value):,.{decimals}f}"


def metric(label: str, value: object, detail: str = "") -> None:
    st.markdown(f'<div class="card"><div class="card-label">{label}</div><div class="card-value">{value}</div><div class="muted">{detail}</div></div>', unsafe_allow_html=True)


def chart(fig: go.Figure, title: str) -> None:
    fig.update_layout(title=title, template="plotly_white", height=360, margin=dict(l=10, r=10, t=50, b=10), font=dict(family="DM Sans"), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")


def overview(df: pd.DataFrame, forecast, metadata: dict, live: bool) -> None:
    latest = df.iloc[-1]
    current = pd.to_numeric(latest.get("aqi"), errors="coerce")
    category = classify_aqi(float(current))[0] if pd.notna(current) else "Data unavailable"
    status = "LIVE" if live else "LAST KNOWN DATA"
    updated = pd.Timestamp(latest["collection_timestamp"]).strftime("%d %b %Y, %H:%M UTC")
    st.markdown(f'<div class="hero"><div class="eyebrow" style="color:#b7f2cf">{status} · {updated}</div><h1>Sukkur AQI Predictor</h1><p>AI-powered air quality intelligence for {latest.get("city", "the monitored location")}</p><div class="hero-stat"><div class="hero-value">{number(current, 0)}</div><div>{category}</div></div></div>', unsafe_allow_html=True)
    st.caption("Stored observations are never presented as a live API response.")
    st.subheader("Current conditions")
    st.caption("The latest readings from the monitored location")
    fields = [("AQI", "aqi", ""), ("PM2.5", "pm2_5", "ug/m3"), ("Temperature", "temperature", "C"), ("Feels like", "feels_like", "C"), ("Humidity", "humidity", "%"), ("Pressure", "pressure", "hPa"), ("Wind speed", "wind_speed", "m/s"), ("Cloud cover", "clouds", "%")]
    cols = st.columns(4)
    for col, (label, field, unit) in zip(cols * 2, fields):
        with col:
            value = latest.get(field)
            suffix = f" {unit}" if pd.notna(value) and unit else ""
            metric(label, f"{number(value)}{suffix}", "Latest available observation")
    st.subheader("Three-day forecast")
    st.caption("Daily mean AQI predicted by the current model")
    cards = st.columns(3)
    values = [getattr(forecast, label) for label, _ in HORIZONS]
    baseline = float(current) if pd.notna(current) else None
    for card, (label, days), value in zip(cards, HORIZONS, values):
        cat = classify_aqi(value)[0]
        direction, delta = trend_delta(value, baseline) if baseline is not None else ("Unavailable", float("nan"))
        target_date = pd.Timestamp(forecast.forecast_date) + pd.Timedelta(days=days - 1)
        with card:
            metric(f"Day {days} · {target_date.strftime('%A, %d %b %Y')}", number(value), f"{cat} · {direction} ({number(delta)})")
            st.badge("Forecast", icon=":material/insights:", color="blue")
    daily = df.set_index("collection_timestamp")["aqi"].resample("D").mean().dropna()
    dates = pd.date_range(forecast.forecast_date, periods=3, freq="D")
    fig = go.Figure(go.Scatter(x=daily.tail(90).index, y=daily.tail(90), name="Observed", mode="lines", line=dict(color="#0f766e")))
    fig.add_trace(go.Scatter(x=dates, y=values, name="Forecast", mode="lines+markers", line=dict(color="#ea580c", dash="dash")))
    for upper, label, band_color in AQI_BANDS[:-1]:
        fig.add_hline(y=upper, line_dash="dot", line_color=band_color, opacity=.35, annotation_text=label)
    chart(fig, "Observed daily AQI and model forecast")
    peak = max(values)
    if alert_level(peak):
        st.warning(f"Air quality alert: the forecast reaches {peak:.1f} AQI ({classify_aqi(peak)[0]}).")
    else:
        st.success("No elevated forecast alert was triggered by the current predictions.")
    with st.expander("Forecast model details"):
        st.json(metadata)


def historical(df: pd.DataFrame) -> None:
    st.header("Historical analysis")
    stats = df["aqi"].describe() if "aqi" in df else pd.Series(dtype=float)
    cols = st.columns(4)
    values = [len(df), df.get("city", pd.Series(dtype=str)).nunique(), stats.get("mean"), stats.get("50%")]
    for col, label, value in zip(cols, ["Records", "Cities", "AQI mean", "AQI median"], values):
        with col:
            metric(label, number(value, 0 if label in {"Records", "Cities"} else 1))
    st.caption(f"Coverage: {df['collection_timestamp'].min():%Y-%m-%d} to {df['collection_timestamp'].max():%Y-%m-%d}")
    period = st.selectbox("History window", ["24 hours", "7 days", "30 days", "90 days", "1 year", "All data"], index=2)
    days = {"24 hours": 1, "7 days": 7, "30 days": 30, "90 days": 90, "1 year": 365}.get(period)
    shown = df[df["collection_timestamp"] >= df["collection_timestamp"].max() - pd.Timedelta(days=days)] if days else df
    chart(go.Figure(go.Scatter(x=shown["collection_timestamp"], y=shown["aqi"], mode="lines", name="AQI", line=dict(color="#0f766e"))), "Historical AQI observations")
    st.subheader("Source coverage")
    coverage = source_summary(df)
    if coverage.empty:
        st.info("AQI source metadata is unavailable.")
    else:
        st.dataframe(coverage, hide_index=True, width="stretch")
        st.caption("OpenAQ-v3 means ground-sensor measurements. OpenMeteo-AirQuality means CAMS model estimates.")
    candidates = [c for c in ["temperature", "humidity", "wind_speed", "pressure", "pm2_5"] if c in df]
    if candidates:
        selected = st.selectbox("Compare AQI with", candidates)
        pair = df[[selected, "aqi"]].apply(pd.to_numeric, errors="coerce").dropna()
        correlation = pair[selected].corr(pair["aqi"]) if len(pair) > 1 else None
        st.caption(f"Pearson correlation: {number(correlation, 3)}. Correlation does not imply causation.")
        chart(go.Figure(go.Scattergl(x=pair[selected], y=pair["aqi"], mode="markers", marker=dict(color="#0f766e", opacity=.45))), f"{selected} versus AQI")


def quality(df: pd.DataFrame) -> None:
    st.header("Data quality and lineage")
    result = dataset_quality(df)
    cols = st.columns(4)
    for col, label, value in zip(cols, ["Rows", "Duplicate keys", "Invalid AQI", "Columns"], [result["rows"], result["duplicates"], result["invalid_aqi"], len(df.columns)]):
        with col:
            metric(label, f"{value:,}")
    missing = result["missing"].head(15).rename("missing_fraction").reset_index().rename(columns={"index": "column"})
    missing["missing_percent"] = missing["missing_fraction"] * 100
    st.dataframe(missing, hide_index=True, width="stretch")
    st.markdown('<div class="source-note">Null means unavailable from the source; it is not a zero measurement.</div>', unsafe_allow_html=True)
    st.subheader("Latest pipeline metadata")
    st.json(latest_metadata() or {"status": "Data unavailable"})


def models() -> None:
    st.header("Model intelligence")
    runs = model_runs()
    if not runs:
        st.info("No model registry metadata is available.")
        return
    for run in runs:
        st.subheader(run.get("run_name", "Model run"))
        rows = []
        for label, _ in HORIZONS:
            item, metrics, baseline = run.get(label, {}), run.get(label, {}).get("test_metrics", {}), run.get(label, {}).get("naive_baseline", {})
            rows.append({"horizon": label, "target": item.get("target", "N/A"), "features": len(item.get("feature_cols", [])), "RMSE": metrics.get("rmse", "N/A"), "MAE": metrics.get("mae", "N/A"), "R2": metrics.get("r2", "N/A"), "beats baseline": metrics.get("rmse", float("inf")) < baseline.get("rmse", float("inf"))})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Registry path: {run.get('run_dir')}")


def explain(df: pd.DataFrame) -> None:
    st.header("Explainable AI")
    st.caption("Contributions explain the model prediction; they are not causal relationships.")
    try:
        with st.spinner("Computing SHAP explanations from the loaded model..."):
            explanations = explain_forecast(df, use_lime=False)
    except Exception as exc:
        st.warning("No explanation is currently available for this forecast.")
        with st.expander("Technical details"):
            st.code(str(exc))
        return
    selected = st.selectbox("Forecast horizon", [item.label for item in explanations])
    item = next(item for item in explanations if item.label == selected)
    values = pd.Series(item.shap_values, index=item.feature_cols)
    values = values.loc[values.abs().sort_values(ascending=False).head(12).index].sort_values()
    fig = go.Figure(go.Bar(x=values.values, y=values.index, orientation="h", marker_color=["#dc2626" if value > 0 else "#0f766e" for value in values]))
    chart(fig, f"SHAP contributions · prediction {number(item.prediction)}")


@st.fragment(run_every="5m")
def render_dashboard() -> None:
    with st.sidebar:
        st.markdown("## Sukkur AQI Predictor")
        view = st.radio("Navigate", ["Overview", "Historical analysis", "Model intelligence", "Explainability", "Data quality"], label_visibility="collapsed")
        live = False
        if st.button("Refresh current data", icon=":material/refresh:", width="stretch"):
            load_data.clear()
            st.rerun()
        st.caption("Historical AQI sources are labeled as ground sensor or CAMS model estimate.")
    df = load_data()
    live = df.attrs.get("live_status") == "success"
    if df.empty:
        st.error("No project data is available. Run ingestion or feature engineering first.")
        return
    try:
        try:
            future_weather = fetch_future_weather(df)
        except Exception:
            future_weather = None
        forecast = make_forecast(df, future_weather=future_weather)
        _, metadata = load_models()
    except Exception as exc:
        forecast, metadata = None, {}
        st.warning("Forecast temporarily unavailable. Other views remain available.")
        with st.expander("Technical details"):
            st.code(str(exc))
    if view == "Overview":
        overview(df, forecast, metadata, live) if forecast else st.info("Forecast unavailable; displaying stored observations only.")
    elif view == "Historical analysis":
        historical(df)
    elif view == "Model intelligence":
        models()
    elif view == "Explainability":
        explain(df)
    else:
        quality(df)


def main() -> None:
    st.set_page_config(page_title="Sukkur AQI Predictor", page_icon=":material/air:", layout="wide", initial_sidebar_state="expanded")
    apply_styles()
    render_dashboard()


if __name__ == "__main__":
    main()
