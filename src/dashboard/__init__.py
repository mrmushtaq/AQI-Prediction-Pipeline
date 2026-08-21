"""AQI forecast dashboard (Phase 6).

Loads the three daily-average Ridge models from the local model registry,
rebuilds the exact daily features the v5 training notebook used, and serves
next-3-day AQI forecasts through a Streamlit dashboard.
"""