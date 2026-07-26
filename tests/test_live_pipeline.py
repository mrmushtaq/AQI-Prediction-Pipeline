from __future__ import annotations

from unittest.mock import patch

import pytest

from src.common.exceptions import AQIPipelineError, PipelineError
from src.feature_pipeline.live_pipeline import run_live_pipeline

SAMPLE_WEATHER = {
    "timestamp": "2026-07-26T12:00:00+00:00",
    "collection_timestamp": "2026-07-26T12:00:00+00:00",
    "city": "Sukkur",
    "latitude": 27.7052,
    "longitude": 68.8574,
    "temperature": 34.2,
    "feels_like": 36.0,
    "humidity": 40,
    "pressure": 1008,
    "wind_speed": 3.1,
    "wind_direction": 180,
    "clouds": 10,
    "visibility": 10000,
    "weather": "clear sky",
    "source": "OpenWeather",
}

SAMPLE_AQI = {
    "timestamp": "2026-07-26T12:00:00+00:00",
    "city": "Sukkur",
    "latitude": 27.7052,
    "longitude": 68.8574,
    "aqi": 152,
    "pm2_5": 90,
    "pm10": 120,
    "co": 4,
    "so2": 2,
    "no2": 10,
    "o3": 15,
    "source": "AQICN-city",
}


def test_run_live_pipeline_success(settings, tmp_path):
    settings = settings.__class__(
        **{**settings.__dict__, "raw_data_dir": tmp_path / "raw",
           "processed_data_dir": tmp_path / "processed"}
    )

    with (
        patch(
            "src.feature_pipeline.live_pipeline.WeatherFetcher.fetch_current_weather",
            return_value=SAMPLE_WEATHER,
        ),
        patch(
            "src.feature_pipeline.live_pipeline.AirQualityFetcher.fetch_air_quality",
            return_value=SAMPLE_AQI,
        ),
    ):
        result = run_live_pipeline(settings)

    assert result["city"] == "Sukkur"
    assert result["aqi"] == 152
    assert result["weather_source"] == "OpenWeather"
    assert result["aqi_source"] == "AQICN-city"

    raw_files = list(tmp_path.glob("raw/*.json"))
    assert len(raw_files) == 1

    processed_files = list(tmp_path.glob("processed/*.csv"))
    assert len(processed_files) >= 1


def test_run_live_pipeline_propagates_weather_failure(settings):
    with patch(
        "src.feature_pipeline.live_pipeline.WeatherFetcher.fetch_current_weather",
        side_effect=AQIPipelineError("Weather API down"),
    ):
        with pytest.raises(AQIPipelineError):
            run_live_pipeline(settings)


def test_run_live_pipeline_propagates_aqi_failure(settings):
    with (
        patch(
            "src.feature_pipeline.live_pipeline.WeatherFetcher.fetch_current_weather",
            return_value=SAMPLE_WEATHER,
        ),
        patch(
            "src.feature_pipeline.live_pipeline.AirQualityFetcher.fetch_air_quality",
            side_effect=AQIPipelineError("AQI API down"),
        ),
    ):
        with pytest.raises(AQIPipelineError):
            run_live_pipeline(settings)


def test_run_live_pipeline_wraps_unexpected_error(settings):
    with (
        patch(
            "src.feature_pipeline.live_pipeline.WeatherFetcher.fetch_current_weather",
            side_effect=RuntimeError("unexpected"),
        ),
    ):
        with pytest.raises(PipelineError):
            run_live_pipeline(settings)
