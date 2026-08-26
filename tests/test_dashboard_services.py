from __future__ import annotations

import json

import pandas as pd
import pytest

from src.dashboard.alerts import alert_level, classify_aqi, trend_delta
from src.dashboard.data_service import dataset_quality, load_historical_data, model_runs, source_summary


def test_aqi_classification_and_trend_are_centralized():
    assert classify_aqi(50)[0] == "Good"
    assert classify_aqi(151)[0] == "Unhealthy"
    assert trend_delta(101, 100) == ("Stable", 1)
    assert trend_delta(120, 100)[0] == "Increasing"
    assert alert_level(149) is None
    assert alert_level(150) == "Unhealthy for sensitive groups"


def test_real_historical_data_is_selected_when_available():
    frame, source = load_historical_data()
    assert not frame.empty
    assert source == "Historical processed data"
    assert len(frame) > 0
    assert frame["collection_timestamp"].is_monotonic_increasing


def test_quality_and_source_summary_use_loaded_values():
    frame = pd.DataFrame(
        {
            "city": ["A", "A", "B"],
            "collection_timestamp": pd.to_datetime(["2026-01-01"] * 2 + ["2026-01-02"], utc=True),
            "aqi": [25, 25, 175],
            "aqi_source": ["OpenAQ-v3", "OpenAQ-v3", "OpenMeteo-AirQuality"],
        }
    )
    result = dataset_quality(frame)
    assert result["rows"] == 3
    assert result["duplicates"] == 1
    summary = source_summary(frame).set_index("source")
    assert summary.loc["OpenAQ-v3", "records"] == 2
    assert summary.loc["OpenMeteo-AirQuality", "percent"] == pytest.approx(100 / 3)


def test_model_runs_reads_registry_metadata(tmp_path):
    run = tmp_path / "ridge_run"
    run.mkdir()
    (run / "metadata.json").write_text(json.dumps({"day1_avg": {"test_metrics": {"r2": 0.5}}}), encoding="utf-8")
    import src.dashboard.data_service as service

    original = service.REGISTRY_DIR
    service.REGISTRY_DIR = tmp_path
    try:
        found = model_runs()
    finally:
        service.REGISTRY_DIR = original
    assert found[0]["day1_avg"]["test_metrics"]["r2"] == 0.5
