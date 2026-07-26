"""Unit tests for `src.feature_pipeline.fetch_weather`."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.common.exceptions import APIError
from src.feature_pipeline.fetch_weather import WeatherFetcher

SAMPLE_OPENWEATHER_RESPONSE = {
    "coord": {"lat": 27.7052, "lon": 68.8574},
    "weather": [{"description": "clear sky"}],
    "main": {
        "temp": 34.2,
        "feels_like": 36.0,
        "humidity": 40,
        "pressure": 1008,
    },
    "visibility": 10000,
    "wind": {"speed": 3.1, "deg": 180},
    "clouds": {"all": 10},
    "sys": {"sunrise": 1690000000, "sunset": 1690040000},
    "name": "Sukkur",
}

SAMPLE_OPENMETEO_HISTORICAL_RESPONSE = {
    "latitude": 27.7052,
    "longitude": 68.8574,
    "generationtime_ms": 0.5,
    "utc_offset_seconds": 0,
    "timezone": "GMT",
    "timezone_abbreviation": "GMT",
    "elevation": 72.0,
    "hourly_units": {
        "time": "iso8601",
        "temperature_2m": "\u00b0C",
        "relative_humidity_2m": "%",
        "surface_pressure": "hPa",
        "wind_speed_10m": "km/h",
        "wind_direction_10m": "\u00b0",
        "cloud_cover": "%",
        "precipitation": "mm",
        "visibility": "m",
        "weather_code": "wmo code",
    },
    "hourly": {
        "time": [
            "2025-01-01T00:00",
            "2025-01-01T01:00",
            "2025-01-01T02:00",
            "2025-01-01T03:00",
        ],
        "temperature_2m": [20.0, 19.5, 19.0, 18.5],
        "relative_humidity_2m": [50, 52, 54, 56],
        "surface_pressure": [1013.0, 1012.8, 1012.5, 1012.3],
        "wind_speed_10m": [2.0, 2.5, 3.0, 3.5],
        "wind_direction_10m": [90, 95, 100, 105],
        "cloud_cover": [20, 25, 30, 35],
        "precipitation": [0.0, 0.0, 0.1, 0.2],
        "visibility": [10000, 10000, 8000, 6000],
        "weather_code": [2, 2, 3, 3],
    },
}


def _mock_response(status_code: int, json_data: dict | None = None, text: str = "") -> MagicMock:
    mock = MagicMock()
    mock.status_code = status_code
    mock.text = text
    if json_data is not None:
        mock.json.return_value = json_data
    return mock


def test_fetch_current_weather_success(settings):
    fetcher = WeatherFetcher(settings)
    with patch.object(
        fetcher.session, "get", return_value=_mock_response(200, SAMPLE_OPENWEATHER_RESPONSE)
    ):
        result = fetcher.fetch_current_weather()

    assert result["city"] == "Sukkur"
    assert result["temperature"] == 34.2
    assert result["humidity"] == 40
    assert result["weather"] == "clear sky"
    assert result["source"] == "OpenWeather"
    assert "timestamp" in result


def test_fetch_current_weather_retries_then_succeeds(settings):
    settings = settings.__class__(**{**settings.__dict__, "max_retries": 2, "retry_backoff_seconds": 0.0})
    fetcher = WeatherFetcher(settings)

    responses = [
        _mock_response(503, text="service unavailable"),
        _mock_response(200, SAMPLE_OPENWEATHER_RESPONSE),
    ]
    with patch.object(fetcher.session, "get", side_effect=responses):
        result = fetcher.fetch_current_weather()

    assert result["city"] == "Sukkur"


def test_fetch_current_weather_raises_api_error_on_non_retryable_status(settings):
    fetcher = WeatherFetcher(settings)
    with patch.object(fetcher.session, "get", return_value=_mock_response(401, text="unauthorized")):
        with pytest.raises(APIError):
            fetcher.fetch_current_weather()


def test_fetch_current_weather_raises_api_error_on_timeout(settings):
    fetcher = WeatherFetcher(settings)
    with patch.object(fetcher.session, "get", side_effect=requests.exceptions.Timeout("timed out")):
        with pytest.raises(APIError):
            fetcher.fetch_current_weather()


def test_fetch_current_weather_raises_api_error_on_malformed_payload(settings):
    fetcher = WeatherFetcher(settings)
    malformed = {"main": {"temp": 30}}  # missing "coord", "weather", etc.
    with patch.object(fetcher.session, "get", return_value=_mock_response(200, malformed)):
        with pytest.raises(APIError):
            fetcher.fetch_current_weather()


def test_fetch_historical_weather_success(settings):
    fetcher = WeatherFetcher(settings)
    with patch.object(
        fetcher.session, "get", return_value=_mock_response(200, SAMPLE_OPENMETEO_HISTORICAL_RESPONSE)
    ):
        result = fetcher.fetch_historical_weather(
            latitude=27.7052, longitude=68.8574, timestamp=1735689600
        )

    assert result["city"] == "Sukkur"
    assert result["temperature"] == 20.0
    assert result["humidity"] == 50
    assert result["weather"] == "partly cloudy"
    assert result["source"] == "Open-Meteo Archive"


def test_fetch_historical_weather_uses_defaults(settings, monkeypatch):
    from datetime import datetime as dt_mod, timezone

    today = dt_mod.now(timezone.utc).strftime("%Y-%m-%d")
    sample = dict(SAMPLE_OPENMETEO_HISTORICAL_RESPONSE)
    sample["hourly"] = {k: list(v) for k, v in sample["hourly"].items()}
    sample["hourly"]["time"].insert(0, f"{today}T00:00")
    sample["hourly"]["temperature_2m"].insert(0, 20.0)
    sample["hourly"]["relative_humidity_2m"].insert(0, 50)
    sample["hourly"]["surface_pressure"].insert(0, 1013.0)
    sample["hourly"]["wind_speed_10m"].insert(0, 2.0)
    sample["hourly"]["wind_direction_10m"].insert(0, 90)
    sample["hourly"]["cloud_cover"].insert(0, 20)
    sample["hourly"]["precipitation"].insert(0, 0.0)
    sample["hourly"]["visibility"].insert(0, 10000)
    sample["hourly"]["weather_code"].insert(0, 2)

    fetcher = WeatherFetcher(settings)
    with patch.object(
        fetcher.session, "get", return_value=_mock_response(200, sample)
    ):
        result = fetcher.fetch_historical_weather()

    assert result["temperature"] == 20.0
    assert result["source"] == "Open-Meteo Archive"


def test_fetch_historical_weather_retries_then_succeeds(settings):
    settings = settings.__class__(**{**settings.__dict__, "max_retries": 2, "retry_backoff_seconds": 0.0})
    fetcher = WeatherFetcher(settings)

    responses = [
        _mock_response(503, text="unavailable"),
        _mock_response(200, SAMPLE_OPENMETEO_HISTORICAL_RESPONSE),
    ]
    with patch.object(fetcher.session, "get", side_effect=responses):
        result = fetcher.fetch_historical_weather(latitude=27.7052, longitude=68.8574, timestamp=1735689600)

    assert result["temperature"] == 20.0


def test_fetch_historical_weather_raises_on_malformed(settings):
    fetcher = WeatherFetcher(settings)
    malformed = {"hourly": {"time": []}}  # missing temperature_2m, etc.
    with patch.object(fetcher.session, "get", return_value=_mock_response(200, malformed)):
        with pytest.raises(APIError):
            fetcher.fetch_historical_weather(latitude=27.7052, longitude=68.8574, timestamp=1735689600)


def test_fetch_historical_weather_accepts_datetime(settings):
    """fetch_historical_weather should accept a datetime object and
    auto-convert to the date string Open-Meteo expects.
    """
    from datetime import datetime, timezone

    fetcher = WeatherFetcher(settings)
    dt = datetime(2025, 1, 1, tzinfo=timezone.utc)

    with patch.object(fetcher.session, "get") as mock_get:
        mock_get.return_value = _mock_response(200, SAMPLE_OPENMETEO_HISTORICAL_RESPONSE)
        fetcher.fetch_historical_weather(latitude=27.7052, longitude=68.8574, timestamp=dt)

    call_url = mock_get.call_args[0][0]
    call_params = mock_get.call_args[1]["params"]
    assert "archive-api.open-meteo.com" in call_url
    assert call_params["start_date"] == "2025-01-01"
    assert call_params["end_date"] == "2025-01-01"


def test_fetch_weather_exponential_backoff_used_between_retries(settings):
    """Verify the weather fetcher uses exponential backoff
    (base * 2**(attempt-1)) matching the AQI fetcher's retry strategy.
    """
    settings = settings.__class__(
        **{**settings.__dict__, "max_retries": 3, "retry_backoff_seconds": 1.0}
    )
    fetcher = WeatherFetcher(settings)

    with patch.object(fetcher.session, "get", return_value=_mock_response(503, text="unavailable")):
        with patch("src.feature_pipeline.fetch_weather.time.sleep") as mock_sleep:
            with pytest.raises(APIError):
                fetcher.fetch_current_weather()

    sleep_durations = [call.args[0] for call in mock_sleep.call_args_list]
    assert sleep_durations[:2] == [1.0, 2.0]
