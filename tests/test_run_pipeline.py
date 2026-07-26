"""Unit tests for `src.feature_pipeline.run_pipeline`, covering the
merge logic, run-metadata persistence (success and failure paths), and
end-to-end orchestration with mocked fetchers.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.common.exceptions import APIError, ValidationError
from src.feature_pipeline.run_pipeline import (
    merge_weather_and_aqi,
    run_pipeline,
    save_run_metadata,
)

SAMPLE_WEATHER = {
    "timestamp": "2026-07-26T00:00:00+00:00",
    "collection_timestamp": "2026-07-26T00:00:00+00:00",
    "city": "Sukkur",
    "latitude": 27.7052,
    "longitude": 68.8574,
    "temperature": 33.2,
    "feels_like": 35.0,
    "humidity": 47,
    "pressure": 999,
    "visibility": 10000,
    "wind_speed": 3.7,
    "wind_direction": 261,
    "clouds": 100,
    "sunrise": 1785026642,
    "sunset": 1785075472,
    "weather": "light rain",
    "source": "OpenWeather",
}

SAMPLE_AQI = {
    "timestamp": "2026-07-26T00:00:00+00:00",
    "city": "Sukkur",
    "latitude": 27.7052,
    "longitude": 68.8574,
    "aqi": 156,
    "pm2_5": 156,
    "pm10": None,
    "co": None,
    "so2": None,
    "no2": None,
    "o3": None,
    "source": "AQICN-geo",
}


def test_merge_weather_and_aqi_produces_expected_schema():
    merged = merge_weather_and_aqi(SAMPLE_WEATHER, SAMPLE_AQI)

    assert merged["city"] == "Sukkur"
    assert merged["temperature"] == 33.2
    assert merged["aqi"] == 156
    assert merged["pm2_5"] == 156
    assert merged["pm10"] is None
    assert merged["weather_source"] == "OpenWeather"
    assert merged["aqi_source"] == "AQICN-geo"
    assert merged["collection_timestamp"] == merged["timestamp"]


def test_save_run_metadata_writes_success_record(settings):
    save_run_metadata(
        settings,
        status="SUCCESS",
        weather_source="OpenWeather",
        aqi_source="AQICN-geo",
        city="Sukkur",
        records=1,
    )

    files = list(settings.metadata_dir.glob("run_*.json"))
    assert len(files) == 1

    data = json.loads(files[0].read_text())
    assert data["status"] == "SUCCESS"
    assert data["city"] == "Sukkur"
    assert data["weather_source"] == "OpenWeather"
    assert data["aqi_source"] == "AQICN-geo"
    assert data["records"] == 1
    assert data["error"] is None


def test_save_run_metadata_writes_failure_record_with_error(settings):
    save_run_metadata(
        settings,
        status="FAILED",
        weather_source="OpenWeather",
        aqi_source=None,
        city="Sukkur",
        records=0,
        error="Both AQI sources failed.",
    )

    files = list(settings.metadata_dir.glob("run_*.json"))
    assert len(files) == 1

    data = json.loads(files[0].read_text())
    assert data["status"] == "FAILED"
    assert data["aqi_source"] is None
    assert data["records"] == 0
    assert "Both AQI sources failed" in data["error"]


def test_save_run_metadata_never_raises_on_storage_failure(settings, monkeypatch):
    """Metadata is diagnostic only -- a storage failure while writing it must
    never bubble up and mask the pipeline's real outcome. `save_json` always
    wraps underlying I/O errors into `StorageError` (its documented contract),
    so that's what we simulate here.
    """
    from src.common.exceptions import StorageError

    with patch(
        "src.feature_pipeline.run_pipeline.save_json",
        side_effect=StorageError("disk full"),
    ):
        # Should not raise, despite save_json raising internally.
        save_run_metadata(settings, status="SUCCESS", records=1)


def test_run_pipeline_success_writes_success_metadata(settings):
    with patch(
        "src.feature_pipeline.run_pipeline.WeatherFetcher.fetch_current_weather",
        return_value=dict(SAMPLE_WEATHER),
    ), patch(
        "src.feature_pipeline.run_pipeline.AirQualityFetcher.fetch_air_quality",
        return_value=dict(SAMPLE_AQI),
    ):
        record = run_pipeline(settings)

    assert record["city"] == "Sukkur"

    metadata_files = list(settings.metadata_dir.glob("run_*.json"))
    assert len(metadata_files) == 1
    metadata = json.loads(metadata_files[0].read_text())
    assert metadata["status"] == "SUCCESS"
    assert metadata["weather_source"] == "OpenWeather"
    assert metadata["aqi_source"] == "AQICN-geo"

    assert metadata["records"] == 1

    processed_csv = settings.processed_data_dir / "aqi_dataset.csv"
    assert processed_csv.exists()


def test_run_pipeline_failure_still_writes_failed_metadata(settings):
    """If AQI fetching fails entirely, the pipeline should propagate the
    error but still leave a FAILED metadata record with whatever partial
    context (e.g. weather_source) was gathered before the failure.
    """
    with patch(
        "src.feature_pipeline.run_pipeline.WeatherFetcher.fetch_current_weather",
        return_value=dict(SAMPLE_WEATHER),
    ), patch(
        "src.feature_pipeline.run_pipeline.AirQualityFetcher.fetch_air_quality",
        side_effect=APIError("All AQI sources failed.", source="AQI-all-sources"),
    ):
        with pytest.raises(APIError):
            run_pipeline(settings)

    metadata_files = list(settings.metadata_dir.glob("run_*.json"))
    assert len(metadata_files) == 1
    metadata = json.loads(metadata_files[0].read_text())
    assert metadata["status"] == "FAILED"
    assert metadata["weather_source"] == "OpenWeather"
    assert metadata["aqi_source"] is None
    assert "All AQI sources failed" in metadata["error"]


def test_run_pipeline_validation_failure_writes_failed_metadata(settings):
    with patch(
        "src.feature_pipeline.run_pipeline.WeatherFetcher.fetch_current_weather",
        return_value=dict(SAMPLE_WEATHER),
    ), patch(
        "src.feature_pipeline.run_pipeline.AirQualityFetcher.fetch_air_quality",
        return_value=dict(SAMPLE_AQI),
    ), patch(
        "src.feature_pipeline.run_pipeline.validate_record",
        side_effect=ValidationError("Missing required field(s): ['aqi']"),
    ):
        with pytest.raises(ValidationError):
            run_pipeline(settings)

    metadata_files = list(settings.metadata_dir.glob("run_*.json"))
    assert len(metadata_files) == 1
    metadata = json.loads(metadata_files[0].read_text())
    assert metadata["status"] == "FAILED"
    assert metadata["aqi_source"] == "AQICN-geo"
