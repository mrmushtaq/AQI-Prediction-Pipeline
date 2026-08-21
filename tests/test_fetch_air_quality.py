"""Unit tests for `src.feature_pipeline.fetch_air_quality`, covering the
three-stage fallback chain: AQICN-by-city -> AQICN-by-geo -> OpenWeather,
plus the OpenAQ v3 historical AQI pipeline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.common.exceptions import APIError
from src.feature_pipeline.fetch_air_quality import AirQualityFetcher

SAMPLE_AQICN_RESPONSE = {
    "status": "ok",
    "data": {
        "aqi": 152,
        "city": {"name": "Sukkur", "geo": [27.7052, 68.8574]},
        "iaqi": {
            "pm25": {"v": 90},
            "pm10": {"v": 120},
            "co": {"v": 4},
            "so2": {"v": 2},
            "no2": {"v": 10},
            "o3": {"v": 15},
        },
    },
}

SAMPLE_AQICN_UNKNOWN_STATION_RESPONSE = {"status": "error", "data": "Unknown station"}

SAMPLE_OPENWEATHER_AIR_POLLUTION_RESPONSE = {
    "coord": {"lon": 68.8574, "lat": 27.7052},
    "list": [
        {
            "main": {"aqi": 3},
            "components": {
                "co": 300.5,
                "no2": 12.1,
                "o3": 55.2,
                "so2": 8.4,
                "pm2_5": 40.1,
                "pm10": 60.3,
            },
        }
    ],
}

# --- OpenAQ v3 test data ---

SAMPLE_V3_LOCATIONS_RESPONSE = {
    "meta": {"name": "openaq-api", "page": 1, "limit": 100, "found": 1},
    "results": [
        {
            "id": 1234,
            "name": "Station A",
            "sensors": [
                {"id": 1001, "name": "PM2.5 Sensor", "parameter": {"id": 2, "name": "pm25", "units": "µg/m³"}},
                {"id": 1002, "name": "PM10 Sensor", "parameter": {"id": 1, "name": "pm10", "units": "µg/m³"}},
                {"id": 1003, "name": "CO Sensor", "parameter": {"id": 8, "name": "co", "units": "µg/m³"}},
                {"id": 1004, "name": "NO2 Sensor", "parameter": {"id": 7, "name": "no2", "units": "ppb"}},
                {"id": 1005, "name": "SO2 Sensor", "parameter": {"id": 6, "name": "so2", "units": "ppb"}},
                {"id": 1006, "name": "O3 Sensor", "parameter": {"id": 10, "name": "o3", "units": "ppb"}},
            ],
            "coordinates": {"latitude": 27.7, "longitude": 68.86},
        }
    ],
}

SAMPLE_V3_EMPTY_LOCATIONS_RESPONSE = {"meta": {"name": "openaq-api", "page": 1, "limit": 100, "found": 0}, "results": []}

SAMPLE_V3_LOCATIONS_NO_SENSORS_RESPONSE = {
    "meta": {"name": "openaq-api", "page": 1, "limit": 100, "found": 1},
    "results": [
        {
            "id": 1234,
            "name": "Station A",
            "sensors": [],
            "coordinates": {"latitude": 27.7, "longitude": 68.86},
        }
    ],
}

SAMPLE_V3_MEASUREMENTS_BY_SENSOR = {
    1001: {  # pm25 sensor
        "meta": {"name": "openaq-api", "page": 1, "limit": 100, "found": 1},
        "results": [
            {
                "value": 25.0,
                "parameter": {"id": 2, "name": "pm25", "units": "µg/m³"},
                "period": {
                    "label": "measurement",
                    "interval": "01:00:00",
                    "datetimeFrom": {"utc": "2025-01-01T00:00:00Z", "local": "2025-01-01T05:00:00+05:00"},
                    "datetimeTo": {"utc": "2025-01-01T01:00:00Z", "local": "2025-01-01T06:00:00+05:00"},
                },
                "coordinates": None,
                "summary": None,
                "coverage": None,
            }
        ],
    },
    1002: {  # pm10 sensor
        "meta": {"name": "openaq-api", "page": 1, "limit": 100, "found": 1},
        "results": [
            {
                "value": 45.0,
                "parameter": {"id": 1, "name": "pm10", "units": "µg/m³"},
                "period": {
                    "label": "measurement",
                    "interval": "01:00:00",
                    "datetimeFrom": {"utc": "2025-01-01T00:00:00Z", "local": "2025-01-01T05:00:00+05:00"},
                    "datetimeTo": {"utc": "2025-01-01T01:00:00Z", "local": "2025-01-01T06:00:00+05:00"},
                },
                "coordinates": None,
                "summary": None,
                "coverage": None,
            }
        ],
    },
    1003: {  # co sensor
        "meta": {"name": "openaq-api", "page": 1, "limit": 100, "found": 1},
        "results": [
            {
                "value": 200.0,
                "parameter": {"id": 8, "name": "co", "units": "µg/m³"},
                "period": {
                    "label": "measurement",
                    "interval": "01:00:00",
                    "datetimeFrom": {"utc": "2025-01-01T00:00:00Z", "local": "2025-01-01T05:00:00+05:00"},
                    "datetimeTo": {"utc": "2025-01-01T01:00:00Z", "local": "2025-01-01T06:00:00+05:00"},
                },
                "coordinates": None,
                "summary": None,
                "coverage": None,
            }
        ],
    },
    1004: {  # no2 sensor
        "meta": {"name": "openaq-api", "page": 1, "limit": 100, "found": 1},
        "results": [
            {
                "value": 8.0,
                "parameter": {"id": 7, "name": "no2", "units": "ppb"},
                "period": {
                    "label": "measurement",
                    "interval": "01:00:00",
                    "datetimeFrom": {"utc": "2025-01-01T00:00:00Z", "local": "2025-01-01T05:00:00+05:00"},
                    "datetimeTo": {"utc": "2025-01-01T01:00:00Z", "local": "2025-01-01T06:00:00+05:00"},
                },
                "coordinates": None,
                "summary": None,
                "coverage": None,
            }
        ],
    },
    1005: {  # so2 sensor
        "meta": {"name": "openaq-api", "page": 1, "limit": 100, "found": 1},
        "results": [
            {
                "value": 5.0,
                "parameter": {"id": 6, "name": "so2", "units": "ppb"},
                "period": {
                    "label": "measurement",
                    "interval": "01:00:00",
                    "datetimeFrom": {"utc": "2025-01-01T00:00:00Z", "local": "2025-01-01T05:00:00+05:00"},
                    "datetimeTo": {"utc": "2025-01-01T01:00:00Z", "local": "2025-01-01T06:00:00+05:00"},
                },
                "coordinates": None,
                "summary": None,
                "coverage": None,
            }
        ],
    },
    1006: {  # o3 sensor
        "meta": {"name": "openaq-api", "page": 1, "limit": 100, "found": 1},
        "results": [
            {
                "value": 30.0,
                "parameter": {"id": 10, "name": "o3", "units": "ppb"},
                "period": {
                    "label": "measurement",
                    "interval": "01:00:00",
                    "datetimeFrom": {"utc": "2025-01-01T00:00:00Z", "local": "2025-01-01T05:00:00+05:00"},
                    "datetimeTo": {"uc": "2025-01-01T01:00:00Z", "local": "2025-01-01T06:00:00+05:00"},
                },
                "coordinates": None,
                "summary": None,
                "coverage": None,
            }
        ],
    },
}

SAMPLE_V3_AQI_TEST_LOCATIONS = {
    "meta": {"name": "openaq-api", "page": 1, "limit": 100, "found": 1},
    "results": [
        {
            "id": 1234,
            "name": "Station A",
            "sensors": [
                {"id": 2001, "name": "PM2.5 Sensor", "parameter": {"id": 2, "name": "pm25", "units": "µg/m³"}},
                {"id": 2002, "name": "PM10 Sensor", "parameter": {"id": 1, "name": "pm10", "units": "µg/m³"}},
                {"id": 2003, "name": "CO Sensor", "parameter": {"id": 8, "name": "co", "units": "ppm"}},
            ],
            "coordinates": {"latitude": 27.7, "longitude": 68.86},
        }
    ],
}

SAMPLE_V3_AQI_TEST_MEASUREMENTS = {
    2001: {  # pm25 sensor - 35.0 µg/m³ -> AQI ~99
        "meta": {"name": "openaq-api", "page": 1, "limit": 100, "found": 1},
        "results": [
            {
                "value": 35.0,
                "parameter": {"id": 2, "name": "pm25", "units": "µg/m³"},
                "period": {
                    "label": "measurement",
                    "interval": "01:00:00",
                    "datetimeFrom": {"utc": "2025-06-15T00:00:00Z", "local": "2025-06-15T05:00:00+05:00"},
                    "datetimeTo": {"utc": "2025-06-15T01:00:00Z", "local": "2025-06-15T06:00:00+05:00"},
                },
                "coordinates": None,
                "summary": None,
                "coverage": None,
            }
        ],
    },
    2002: {  # pm10 sensor - 85.0 µg/m³ -> AQI ~51
        "meta": {"name": "openaq-api", "page": 1, "limit": 100, "found": 1},
        "results": [
            {
                "value": 85.0,
                "parameter": {"id": 1, "name": "pm10", "units": "µg/m³"},
                "period": {
                    "label": "measurement",
                    "interval": "01:00:00",
                    "datetimeFrom": {"utc": "2025-06-15T00:00:00Z", "local": "2025-06-15T05:00:00+05:00"},
                    "datetimeTo": {"utc": "2025-06-15T01:00:00Z", "local": "2025-06-15T06:00:00+05:00"},
                },
                "coordinates": None,
                "summary": None,
                "coverage": None,
            }
        ],
    },
    2003: {  # co sensor - 5.0 ppm -> AQI ~56
        "meta": {"name": "openaq-api", "page": 1, "limit": 100, "found": 1},
        "results": [
            {
                "value": 5.0,
                "parameter": {"id": 8, "name": "co", "units": "ppm"},
                "period": {
                    "label": "measurement",
                    "interval": "01:00:00",
                    "datetimeFrom": {"utc": "2025-06-15T00:00:00Z", "local": "2025-06-15T05:00:00+05:00"},
                    "datetimeTo": {"utc": "2025-06-15T01:00:00Z", "local": "2025-06-15T06:00:00+05:00"},
                },
                "coordinates": None,
                "summary": None,
                "coverage": None,
            }
        ],
    },
}

SAMPLE_V3_UNIT_CONVERSION_LOCATIONS = {
    "meta": {"name": "openaq-api", "page": 1, "limit": 100, "found": 1},
    "results": [
        {
            "id": 1234,
            "name": "Station A",
            "sensors": [
                {"id": 3001, "name": "CO Sensor", "parameter": {"id": 8, "name": "co", "units": "µg/m³"}},
            ],
            "coordinates": {"latitude": 27.7, "longitude": 68.86},
        }
    ],
}

SAMPLE_V3_UNIT_CONVERSION_MEASUREMENTS = {
    3001: {  # co sensor - 11460 µg/m³
        "meta": {"name": "openaq-api", "page": 1, "limit": 100, "found": 1},
        "results": [
            {
                "value": 11460.0,
                "parameter": {"id": 8, "name": "co", "units": "µg/m³"},
                "period": {
                    "label": "measurement",
                    "interval": "01:00:00",
                    "datetimeFrom": {"utc": "2025-06-15T00:00:00Z", "local": "2025-06-15T05:00:00+05:00"},
                    "datetimeTo": {"utc": "2025-06-15T01:00:00Z", "local": "2025-06-15T06:00:00+05:00"},
                },
                "coordinates": None,
                "summary": None,
                "coverage": None,
            }
        ],
    },
}

SAMPLE_V3_UNSUPPORTED_PARAM_LOCATIONS = {
    "meta": {"name": "openaq-api", "page": 1, "limit": 100, "found": 1},
    "results": [
        {
            "id": 1234,
            "name": "Station A",
            "sensors": [
                {"id": 4001, "name": "BC Sensor", "parameter": {"id": 11, "name": "bc", "units": "µg/m³"}},
            ],
            "coordinates": {"latitude": 27.7, "longitude": 68.86},
        }
    ],
}

SAMPLE_V3_EMPTY_MEASUREMENTS = {
    "meta": {"name": "openaq-api", "page": 1, "limit": 100, "found": 0},
    "results": [],
}

# --- Open-Meteo Air Quality (historical fallback) test data ---

SAMPLE_OPENMETEO_AIR_QUALITY_RESPONSE = {
    "latitude": 27.7052,
    "longitude": 68.8574,
    "hourly": {
        "time": [
            "2025-01-01T00:00",
            "2025-01-01T01:00",
            "2025-01-01T02:00",
            "2025-01-01T03:00",
        ],
        "pm10": [45.0, 46.0, 47.0, 48.0],
        "pm2_5": [25.0, 26.0, 27.0, 28.0],
        "carbon_monoxide": [300.0, 310.0, 320.0, 330.0],
        "nitrogen_dioxide": [8.0, 9.0, 10.0, 11.0],
        "sulphur_dioxide": [5.0, 6.0, 7.0, 8.0],
        "ozone": [30.0, 31.0, 32.0, 33.0],
        "us_aqi": [120, 121, 122, 123],
    },
}


def _mock_response(status_code: int, json_data: dict | None = None, text: str = "") -> MagicMock:
    mock = MagicMock()
    mock.status_code = status_code
    mock.text = text
    if json_data is not None:
        mock.json.return_value = json_data
    return mock


def _is_city_query(url: str) -> bool:
    return "waqi.info/feed/" in url and "geo:" not in url


def _is_geo_query(url: str) -> bool:
    return "waqi.info/feed/geo:" in url


def _is_v3_locations_query(url: str) -> bool:
    return "openaq.org/v3/locations" in url


def _is_v3_sensor_query(url: str) -> bool:
    return "openaq.org/v3/sensors/" in url and "/measurements" in url


def _is_openmeteo_query(url: str) -> bool:
    return "air-quality-api.open-meteo.com" in url


def _build_v3_sensor_side_effect(
    locations_response: dict,
    measurements_by_sensor: dict[int, dict],
    openmeteo_response: MagicMock | None = None,
):
    """Build a side-effect function that returns the appropriate mock response
    for each step of the OpenAQ v3 lookup chain -- and, optionally, for the
    Open-Meteo Air Quality fallback that is reached whenever OpenAQ raises an
    ``APIError``.

    Args:
        locations_response: The JSON response for the initial locations query.
        measurements_by_sensor: Dict mapping sensor_id -> JSON response for
            that sensor's measurements endpoint.
        openmeteo_response: Mock response to return for the Open-Meteo
            Air Quality fallback endpoint. If ``None``, hitting that endpoint
            raises ``AssertionError`` (so tests that expect OpenAQ to succeed
            can prove the fallback was never invoked).
    """
    call_count = [0]

    def side_effect(url, params=None, headers=None, timeout=None):
        if _is_v3_locations_query(url):
            return _mock_response(200, locations_response)
        if _is_v3_sensor_query(url):
            # Extract sensor ID from URL.
            parts = url.split("/")
            try:
                sensor_idx = parts.index("sensors") + 1
                sensor_id = int(parts[sensor_idx])
            except (ValueError, IndexError):
                raise AssertionError(f"Cannot parse sensor ID from URL: {url}")
            resp = measurements_by_sensor.get(sensor_id, SAMPLE_V3_EMPTY_MEASUREMENTS)
            call_count[0] += 1
            return _mock_response(200, resp)
        if _is_openmeteo_query(url):
            if openmeteo_response is None:
                raise AssertionError(
                    f"Unexpected Open-Meteo fallback URL: {url} (OpenAQ did not fail)"
                )
            return openmeteo_response
        raise AssertionError(f"Unexpected URL: {url}")

    return side_effect


def test_stage1_aqicn_city_succeeds_without_further_stages(settings):
    fetcher = AirQualityFetcher(settings)
    with patch.object(fetcher.session, "get", return_value=_mock_response(200, SAMPLE_AQICN_RESPONSE)):
        result = fetcher.fetch_air_quality()

    assert result["source"] == "AQICN-city"
    assert result["aqi"] == 152
    assert result["pm2_5"] == 90


def test_stage2_aqicn_geo_used_when_city_query_fails(settings):
    fetcher = AirQualityFetcher(settings)

    def get_side_effect(url, **kwargs):
        if _is_city_query(url):
            return _mock_response(200, SAMPLE_AQICN_UNKNOWN_STATION_RESPONSE)
        if _is_geo_query(url):
            return _mock_response(200, SAMPLE_AQICN_RESPONSE)
        raise AssertionError(f"Unexpected URL requested: {url}")

    with patch.object(fetcher.session, "get", side_effect=get_side_effect):
        result = fetcher.fetch_air_quality()

    assert result["source"] == "AQICN-geo"
    assert result["aqi"] == 152


def test_stage3_openweather_used_when_both_aqicn_stages_fail(settings):
    fetcher = AirQualityFetcher(settings)

    def get_side_effect(url, **kwargs):
        if _is_city_query(url) or _is_geo_query(url):
            return _mock_response(500, text="internal error")
        return _mock_response(200, SAMPLE_OPENWEATHER_AIR_POLLUTION_RESPONSE)

    with patch.object(fetcher.session, "get", side_effect=get_side_effect):
        result = fetcher.fetch_air_quality()

    assert result["source"] == "OpenWeather-AirPollution"
    assert result["aqi"] == 3
    assert result["pm2_5"] == 40.1


def test_stage3_openweather_used_when_both_aqicn_stages_return_non_ok(settings):
    fetcher = AirQualityFetcher(settings)

    def get_side_effect(url, **kwargs):
        if _is_city_query(url) or _is_geo_query(url):
            return _mock_response(200, SAMPLE_AQICN_UNKNOWN_STATION_RESPONSE)
        return _mock_response(200, SAMPLE_OPENWEATHER_AIR_POLLUTION_RESPONSE)

    with patch.object(fetcher.session, "get", side_effect=get_side_effect):
        result = fetcher.fetch_air_quality()

    assert result["source"] == "OpenWeather-AirPollution"


def test_raises_when_all_three_stages_fail(settings):
    fetcher = AirQualityFetcher(settings)
    with patch.object(fetcher.session, "get", return_value=_mock_response(500, text="down")):
        with pytest.raises(APIError):
            fetcher.fetch_air_quality()


# --- OpenAQ v3 Historical Tests ---


def test_fetch_historical_aqi_success(settings):
    """fetch_historical_aqi uses OpenAQ v3 and computes AQI correctly."""
    fetcher = AirQualityFetcher(settings)
    side_effect = _build_v3_sensor_side_effect(
        SAMPLE_V3_LOCATIONS_RESPONSE,
        SAMPLE_V3_MEASUREMENTS_BY_SENSOR,
    )
    with patch.object(fetcher.session, "get", side_effect=side_effect):
        result = fetcher.fetch_historical_aqi(
            latitude=27.7052, longitude=68.8574, timestamp=1735689600
        )
    assert result["source"] == "OpenAQ-v3"
    assert result["aqi"] is not None
    assert result["pm2_5"] == 25.0
    assert result["pm10"] == 45.0
    assert result["co"] == 200.0
    assert result["no2"] == 8.0
    assert result["so2"] == 5.0
    assert result["o3"] == 30.0


def test_fetch_historical_aqi_raises_on_empty_locations(settings):
    """When OpenAQ v3 returns no locations AND the Open-Meteo fallback also
    fails, raise a clear error mentioning both sources."""
    fetcher = AirQualityFetcher(settings)
    side_effect = _build_v3_sensor_side_effect(
        SAMPLE_V3_EMPTY_LOCATIONS_RESPONSE,
        {},
        openmeteo_response=_mock_response(500, text="down"),
    )
    with patch.object(fetcher.session, "get", side_effect=side_effect):
        with pytest.raises(APIError, match="All historical AQI sources failed"):
            fetcher.fetch_historical_aqi(
                latitude=27.7052, longitude=68.8574, timestamp=1735689600
            )


def test_fetch_historical_aqi_raises_on_locations_without_sensors(settings):
    """When OpenAQ v3 returns locations but no target sensors AND the
    Open-Meteo fallback also fails, raise a clear error."""
    fetcher = AirQualityFetcher(settings)
    side_effect = _build_v3_sensor_side_effect(
        SAMPLE_V3_LOCATIONS_NO_SENSORS_RESPONSE,
        {},
        openmeteo_response=_mock_response(500, text="down"),
    )
    with patch.object(fetcher.session, "get", side_effect=side_effect):
        with pytest.raises(APIError, match="All historical AQI sources failed"):
            fetcher.fetch_historical_aqi(
                latitude=27.7052, longitude=68.8574, timestamp=1735689600
            )


def test_fetch_historical_aqi_raises_on_api_error(settings):
    """When the OpenAQ v3 API returns a 5xx, raise an APIError."""
    fetcher = AirQualityFetcher(settings)
    with patch.object(fetcher.session, "get", return_value=_mock_response(500, text="down")):
        with pytest.raises(APIError):
            fetcher.fetch_historical_aqi(
                latitude=27.7052, longitude=68.8574, timestamp=1735689600
            )


def test_fetch_historical_aqi_uses_defaults(settings):
    """Defaults for latitude/longitude should fall back to settings."""
    fetcher = AirQualityFetcher(settings)
    from datetime import datetime, timezone
    side_effect = _build_v3_sensor_side_effect(
        SAMPLE_V3_LOCATIONS_RESPONSE,
        SAMPLE_V3_MEASUREMENTS_BY_SENSOR,
    )
    with patch.object(fetcher.session, "get", side_effect=side_effect):
        result = fetcher.fetch_historical_aqi(
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc)
        )
    assert result["source"] == "OpenAQ-v3"
    assert result["pm2_5"] == 25.0


def test_fetch_historical_aqi_accepts_datetime(settings):
    """fetch_historical_aqi should accept a datetime object."""
    from datetime import datetime, timezone

    fetcher = AirQualityFetcher(settings)
    dt = datetime(2025, 1, 1, tzinfo=timezone.utc)

    side_effect = _build_v3_sensor_side_effect(
        SAMPLE_V3_LOCATIONS_RESPONSE,
        SAMPLE_V3_MEASUREMENTS_BY_SENSOR,
    )
    with patch.object(fetcher.session, "get", side_effect=side_effect) as mock_get:
        fetcher.fetch_historical_aqi(
            latitude=27.7052, longitude=68.8574, timestamp=dt
        )

    calls = mock_get.call_args_list
    # First call should be to the locations endpoint.
    locations_call_url = calls[0][0][0]
    assert "openaq.org/v3/locations" in locations_call_url
    assert "coordinates" in calls[0][1]["params"]
    # Subsequent calls should be to sensor measurements with datetime_from/to.
    sensor_call_params = calls[1][1]["params"]
    assert "datetime_from" in sensor_call_params
    assert "datetime_to" in sensor_call_params


def test_fetch_historical_aqi_computes_aqi_correctly(settings):
    """Verify the AQI value is computed from the EPA formula, not faked."""
    fetcher = AirQualityFetcher(settings)
    side_effect = _build_v3_sensor_side_effect(
        SAMPLE_V3_AQI_TEST_LOCATIONS,
        SAMPLE_V3_AQI_TEST_MEASUREMENTS,
    )
    with patch.object(fetcher.session, "get", side_effect=side_effect):
        result = fetcher.fetch_historical_aqi(
            latitude=27.7052, longitude=68.8574, timestamp=1735689600
        )
        # PM2.5 of 35.0 -> AQI 99 (EPA formula)
        # PM10 of 85 -> AQI ~51
        # CO 5.0 ppm -> AQI ~56
        # Overall AQI should be max = 99
        assert result["aqi"] == 99
        assert result["source"] == "OpenAQ-v3"


def test_fetch_historical_aqi_handles_unit_conversion(settings):
    """Verify CO µg/m³ is correctly converted to ppm for AQI calculation."""
    fetcher = AirQualityFetcher(settings)
    side_effect = _build_v3_sensor_side_effect(
        SAMPLE_V3_UNIT_CONVERSION_LOCATIONS,
        SAMPLE_V3_UNIT_CONVERSION_MEASUREMENTS,
    )
    with patch.object(fetcher.session, "get", side_effect=side_effect):
        result = fetcher.fetch_historical_aqi(
            latitude=27.7052, longitude=68.8574, timestamp=1735689600
        )
    # 11460 µg/m³ / 1146 = 10 ppm -> AQI ~107
    assert result["aqi"] is not None
    assert result["co"] == 11460.0


def test_fetch_historical_aqi_raises_on_no_supported_sensors(settings):
    """When OpenAQ v3 returns locations with only unsupported sensors AND the
    Open-Meteo fallback also fails, raise a clear error."""
    fetcher = AirQualityFetcher(settings)
    side_effect = _build_v3_sensor_side_effect(
        SAMPLE_V3_UNSUPPORTED_PARAM_LOCATIONS,
        {},
        openmeteo_response=_mock_response(500, text="down"),
    )
    with patch.object(fetcher.session, "get", side_effect=side_effect):
        with pytest.raises(APIError, match="All historical AQI sources failed"):
            fetcher.fetch_historical_aqi(
                latitude=27.7052, longitude=68.8574, timestamp=1735689600
            )


def test_fetch_historical_aqi_falls_back_to_openmeteo_on_empty_locations(settings):
    """When OpenAQ v3 has no ground sensors for the location, the pipeline
    transparently falls back to the Open-Meteo Air Quality (model) API."""
    fetcher = AirQualityFetcher(settings)
    side_effect = _build_v3_sensor_side_effect(
        SAMPLE_V3_EMPTY_LOCATIONS_RESPONSE,
        {},
        openmeteo_response=_mock_response(200, SAMPLE_OPENMETEO_AIR_QUALITY_RESPONSE),
    )
    with patch.object(fetcher.session, "get", side_effect=side_effect):
        result = fetcher.fetch_historical_aqi(
            latitude=27.7052, longitude=68.8574, timestamp=1735689600
        )

    assert result["source"] == "OpenMeteo-AirQuality"
    assert result["aqi"] == 120
    assert result["pm2_5"] == 25.0
    # Pollutant fields are deliberately nulled for the model-estimate era to
    # keep the schema consistent across the full historical range.
    assert result["pm10"] is None
    assert result["co"] is None
    assert result["so2"] is None
    assert result["no2"] is None
    assert result["o3"] is None


def test_fetch_historical_aqi_does_not_fall_back_when_openaq_succeeds(settings):
    """The Open-Meteo fallback must NOT be invoked when OpenAQ v3 succeeds."""
    fetcher = AirQualityFetcher(settings)
    side_effect = _build_v3_sensor_side_effect(
        SAMPLE_V3_LOCATIONS_RESPONSE,
        SAMPLE_V3_MEASUREMENTS_BY_SENSOR,
    )
    with patch.object(fetcher.session, "get", side_effect=side_effect) as mock_get:
        result = fetcher.fetch_historical_aqi(
            latitude=27.7052, longitude=68.8574, timestamp=1735689600
        )

    assert result["source"] == "OpenAQ-v3"
    assert all(not _is_openmeteo_query(call.args[0]) for call in mock_get.call_args_list)


def test_fetch_historical_aqi_passes_auth_header(settings):
    """Verify the X-API-Key header is passed with every v3 request."""
    fetcher = AirQualityFetcher(settings)
    side_effect = _build_v3_sensor_side_effect(
        SAMPLE_V3_LOCATIONS_RESPONSE,
        SAMPLE_V3_MEASUREMENTS_BY_SENSOR,
    )
    with patch.object(fetcher.session, "get", side_effect=side_effect) as mock_get:
        fetcher.fetch_historical_aqi(
            latitude=27.7052, longitude=68.8574, timestamp=1735689600
        )

    for call_args in mock_get.call_args_list:
        headers = call_args[1].get("headers", {})
        assert headers.get("X-API-Key") == "test-openaq-key"


def test_exponential_backoff_used_between_retries(settings):
    """Verify the retry helper sleeps with exponentially increasing backoff
    (base * 2**(attempt-1)) rather than a linear schedule.
    """
    settings = settings.__class__(**{**settings.__dict__, "max_retries": 3, "retry_backoff_seconds": 1.0})
    fetcher = AirQualityFetcher(settings)

    # Every attempt across every stage fails with a transient (retryable) error.
    with patch.object(fetcher.session, "get", return_value=_mock_response(503, text="unavailable")):
        with patch("src.feature_pipeline.fetch_air_quality.time.sleep") as mock_sleep:
            with pytest.raises(APIError):
                fetcher.fetch_air_quality()

    sleep_durations = [call.args[0] for call in mock_sleep.call_args_list]
    # Within a single stage's 3 attempts, backoff should be 1s, then 2s (2 sleeps per stage).
    assert sleep_durations[:2] == [1.0, 2.0]
