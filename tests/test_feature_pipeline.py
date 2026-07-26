from __future__ import annotations

import pytest

from src.common.exceptions import PipelineError
from src.feature_pipeline.feature_pipeline import merge_weather_and_aqi, process_features


WEATHER_SAMPLE = {
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

AQI_SAMPLE = {
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


def test_merge_weather_and_aqi():
    merged = merge_weather_and_aqi(WEATHER_SAMPLE, AQI_SAMPLE)

    assert merged["city"] == "Sukkur"
    assert merged["temperature"] == 34.2
    assert merged["aqi"] == 152
    assert merged["pm2_5"] == 90
    assert merged["weather_source"] == "OpenWeather"
    assert merged["aqi_source"] == "AQICN-city"
    assert merged["timestamp"] == "2026-07-26T12:00:00+00:00"
    assert merged["collection_timestamp"] == "2026-07-26T12:00:00+00:00"


def test_merge_weather_and_aqi_defaults_collection_timestamp():
    """When weather dict lacks a collection_timestamp, it should default to
    the API timestamp (backward-compatible with live pipeline).
    """
    weather_no_ct = dict(WEATHER_SAMPLE)
    del weather_no_ct["collection_timestamp"]
    merged = merge_weather_and_aqi(weather_no_ct, AQI_SAMPLE)
    assert merged["collection_timestamp"] == merged["timestamp"]


def test_merge_weather_and_aqi_missing_weather_key():
    bad_weather = {"temperature": 25.0}
    with pytest.raises(KeyError):
        merge_weather_and_aqi(bad_weather, AQI_SAMPLE)


def test_process_features_success():
    result = process_features(WEATHER_SAMPLE, AQI_SAMPLE)

    assert result["city"] == "Sukkur"
    assert result["temperature"] == 34.2
    assert result["aqi"] == 152
    assert result["weather_source"] == "OpenWeather"
    assert result["aqi_source"] == "AQICN-city"


def test_process_features_raises_on_missing_required_field():
    bad_weather = {k: v for k, v in WEATHER_SAMPLE.items() if k != "temperature"}
    bad_weather["temperature"] = None
    with pytest.raises(PipelineError):
        process_features(bad_weather, AQI_SAMPLE)


def test_process_features_raises_on_invalid_timestamp():
    bad_weather = dict(WEATHER_SAMPLE)
    bad_weather["timestamp"] = "not-a-timestamp"
    with pytest.raises(PipelineError):
        process_features(bad_weather, AQI_SAMPLE)


def test_process_features_raises_on_wrong_datatype():
    bad_weather = dict(WEATHER_SAMPLE)
    bad_weather["humidity"] = "forty percent"
    with pytest.raises(PipelineError):
        process_features(bad_weather, AQI_SAMPLE)


def test_process_features_works_with_partial_pollutants():
    partial_aqi = {k: v for k, v in AQI_SAMPLE.items()}
    partial_aqi["pm10"] = None
    partial_aqi["co"] = None
    result = process_features(WEATHER_SAMPLE, partial_aqi)
    assert result["aqi"] == 152
    assert result["pm10"] is None
