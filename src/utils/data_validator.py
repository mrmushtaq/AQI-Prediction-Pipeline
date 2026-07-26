"""
Validation rules applied to fetched weather and air-quality records before
they are merged and persisted.

Every function raises `ValidationError` (rather than returning booleans) so
that the pipeline fails loudly and early instead of silently persisting bad
data -- data quality bugs found downstream (in an ML model, weeks later) are
far more expensive than a pipeline that fails fast at ingestion time.
"""

from __future__ import annotations

from datetime import datetime, timezone
from numbers import Number
from typing import Any, Mapping, Sequence

from src.common.exceptions import ValidationError
from src.common.logger import get_logger

logger = get_logger(__name__)

# Fields that must be present and non-null in every merged record. These are
# always reported by both weather providers and by every AQICN station (AQI
# and PM2.5 are the two most universally available pollutant readings).
REQUIRED_FIELDS: Sequence[str] = (
    "timestamp",
    "collection_timestamp",
    "city",
    "latitude",
    "longitude",
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "weather",
    "aqi",
    "pm2_5",
)

# Pollutant fields that must be *present as a key* but MAY be null. Not every
# AQICN monitoring station is equipped with sensors for every pollutant --
# many report only AQI + PM2.5 -- so treating these as strictly required would
# reject perfectly valid station data. When present, they are still validated
# for correct datatype and non-negativity below.
OPTIONAL_NULLABLE_FIELDS: Sequence[str] = (
    "pm10",
    "co",
    "so2",
    "no2",
    "o3",
)

# Fields that must be numeric (int or float).
NUMERIC_FIELDS: Sequence[str] = (
    "latitude",
    "longitude",
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_direction",
    "visibility",
    "clouds",
    "aqi",
    "pm2_5",
    "pm10",
    "co",
    "so2",
    "no2",
    "o3",
)

# Fields that must never be negative.
NON_NEGATIVE_FIELDS: Sequence[str] = (
    "humidity",
    "pressure",
    "wind_speed",
    "visibility",
    "aqi",
    "pm2_5",
    "pm10",
    "co",
    "so2",
    "no2",
    "o3",
)


def validate_required_fields(record: Mapping[str, Any]) -> None:
    """Ensure all strictly-required fields are present and not None, and that
    optional pollutant fields at least exist as keys (their value may be
    `None` when a monitoring station doesn't measure that pollutant).

    Raises:
        ValidationError: If any strictly-required field is missing/None, or
            an optional pollutant field is absent as a key entirely.
    """
    missing = [field for field in REQUIRED_FIELDS if field not in record or record[field] is None]
    if missing:
        raise ValidationError(f"Missing required field(s): {missing}")

    absent_optional_keys = [field for field in OPTIONAL_NULLABLE_FIELDS if field not in record]
    if absent_optional_keys:
        raise ValidationError(
            f"Optional pollutant field(s) missing from record entirely "
            f"(expected key present, value may be null): {absent_optional_keys}"
        )


def validate_datatypes(record: Mapping[str, Any]) -> None:
    """Ensure numeric fields hold numeric (non-boolean) values.

    Raises:
        ValidationError: If a numeric field holds a non-numeric value.
    """
    bad_fields = []
    for field in NUMERIC_FIELDS:
        value = record.get(field)
        if value is None:
            continue  # already caught by validate_required_fields
        if isinstance(value, bool) or not isinstance(value, Number):
            bad_fields.append((field, type(value).__name__))
    if bad_fields:
        raise ValidationError(f"Field(s) with incorrect datatype (expected numeric): {bad_fields}")


def validate_non_negative_values(record: Mapping[str, Any]) -> None:
    """Ensure fields that are physically non-negative are not negative.

    Raises:
        ValidationError: If any field in NON_NEGATIVE_FIELDS is negative.
    """
    negatives = []
    for field in NON_NEGATIVE_FIELDS:
        value = record.get(field)
        if isinstance(value, Number) and not isinstance(value, bool) and value < 0:
            negatives.append((field, value))
    if negatives:
        raise ValidationError(f"Field(s) with invalid negative value: {negatives}")


def validate_coordinates(record: Mapping[str, Any]) -> None:
    """Ensure latitude is within [-90, 90] and longitude within [-180, 180].

    Raises:
        ValidationError: If latitude or longitude is out of range or not numeric.
    """
    lat = record.get("latitude")
    lon = record.get("longitude")

    if not isinstance(lat, Number) or isinstance(lat, bool) or not (-90 <= lat <= 90):
        raise ValidationError(f"Invalid latitude: {lat!r} (must be a number in [-90, 90])")

    if not isinstance(lon, Number) or isinstance(lon, bool) or not (-180 <= lon <= 180):
        raise ValidationError(f"Invalid longitude: {lon!r} (must be a number in [-180, 180])")


def validate_timestamp(record: Mapping[str, Any]) -> None:
    """Ensure `timestamp` is a valid ISO-8601 string parseable to a datetime,
    and is not absurdly in the future (which would indicate a unit bug, e.g.
    seconds vs. milliseconds).

    Raises:
        ValidationError: If the timestamp is missing, malformed, or invalid.
    """
    timestamp = record.get("timestamp")
    if not isinstance(timestamp, str):
        raise ValidationError(f"Invalid timestamp type: {type(timestamp).__name__} (expected str)")

    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(f"Invalid timestamp format: {timestamp!r} ({exc})") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    # Allow a small forward skew for clock drift, but reject anything wildly
    # in the future (e.g. more than 1 day ahead), which usually signals a
    # units bug (seconds vs. milliseconds) rather than a legitimate forecast.
    if (parsed - now).total_seconds() > 86400:
        raise ValidationError(f"Timestamp is implausibly far in the future: {timestamp!r}")


def validate_record(record: Mapping[str, Any]) -> None:
    """Run the full validation suite against a single merged weather+AQI record.

    Args:
        record: A merged dict as produced by the pipeline before persistence.

    Raises:
        ValidationError: If any validation rule fails. The message describes
            exactly which rule failed.
    """
    validate_required_fields(record)
    validate_datatypes(record)
    validate_non_negative_values(record)
    validate_coordinates(record)
    validate_timestamp(record)
    logger.info("Record validation passed for city=%s timestamp=%s", record.get("city"), record.get("timestamp"))
