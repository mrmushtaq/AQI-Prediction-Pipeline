"""Unit tests for `src.utils.data_validator`."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.common.exceptions import ValidationError
from src.utils.data_validator import validate_record


def _valid_record(**overrides) -> dict:
    base = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "collection_timestamp": datetime.now(timezone.utc).isoformat(),
        "city": "Sukkur",
        "latitude": 27.7052,
        "longitude": 68.8574,
        "temperature": 34.2,
        "humidity": 40,
        "pressure": 1008,
        "wind_speed": 3.1,
        "visibility": 10000,
        "weather": "clear sky",
        "aqi": 152,
        "pm2_5": 90,
        "pm10": 120,
        "co": 4,
        "so2": 2,
        "no2": 10,
        "o3": 15,
    }
    base.update(overrides)
    return base


def test_validate_record_accepts_valid_record():
    record = _valid_record()
    validate_record(record)  # should not raise


def test_validate_record_accepts_partial_pollutant_data():
    """Some AQICN stations only report AQI + PM2.5; other pollutant fields
    should be allowed to be None as long as the key is present.
    """
    record = _valid_record(pm10=None, co=None, so2=None, no2=None, o3=None)
    validate_record(record)  # should not raise


def test_validate_record_rejects_missing_required_field():
    record = _valid_record()
    del record["aqi"]
    with pytest.raises(ValidationError):
        validate_record(record)


def test_validate_record_rejects_optional_field_absent_as_key():
    record = _valid_record()
    del record["pm10"]  # key must still exist, even if value would be None
    with pytest.raises(ValidationError):
        validate_record(record)


def test_validate_record_rejects_none_required_field():
    record = _valid_record(humidity=None)
    with pytest.raises(ValidationError):
        validate_record(record)


def test_validate_record_rejects_wrong_datatype():
    record = _valid_record(humidity="forty percent")
    with pytest.raises(ValidationError):
        validate_record(record)


def test_validate_record_rejects_negative_aqi():
    record = _valid_record(aqi=-5)
    with pytest.raises(ValidationError):
        validate_record(record)


def test_validate_record_rejects_negative_humidity():
    record = _valid_record(humidity=-10)
    with pytest.raises(ValidationError):
        validate_record(record)


def test_validate_record_rejects_invalid_latitude():
    record = _valid_record(latitude=120.0)
    with pytest.raises(ValidationError):
        validate_record(record)


def test_validate_record_rejects_invalid_longitude():
    record = _valid_record(longitude=200.0)
    with pytest.raises(ValidationError):
        validate_record(record)


def test_validate_record_rejects_malformed_timestamp():
    record = _valid_record(timestamp="not-a-timestamp")
    with pytest.raises(ValidationError):
        validate_record(record)


def test_validate_record_rejects_far_future_timestamp():
    future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    record = _valid_record(timestamp=future)
    with pytest.raises(ValidationError):
        validate_record(record)
