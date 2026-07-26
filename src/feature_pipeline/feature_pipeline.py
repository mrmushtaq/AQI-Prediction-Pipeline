from __future__ import annotations

from typing import Any

from src.common.exceptions import PipelineError
from src.common.logger import get_logger
from src.utils.data_validator import validate_record

logger = get_logger(__name__)


OUTPUT_FIELDS = (
    "timestamp",
    "collection_timestamp",
    "city",
    "latitude",
    "longitude",
    "temperature",
    "feels_like",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_direction",
    "clouds",
    "visibility",
    "weather",
    "aqi",
    "pm2_5",
    "pm10",
    "co",
    "so2",
    "no2",
    "o3",
    "weather_source",
    "aqi_source",
)


def merge_weather_and_aqi(weather: dict[str, Any], aqi: dict[str, Any]) -> dict[str, Any]:
    """Merge standardized weather and AQI dictionaries into a single flat
    record matching the pipeline's output dataset schema.

    Weather fields take precedence for ``city`` / ``latitude`` / ``longitude``
    metadata (they are considered the primary source of truth for location),
    while all AQI-specific measurements come from the AQI record.

    Args:
        weather: Standardized weather dictionary (see
            :mod:`~src.feature_pipeline.fetch_weather`).
        aqi:     Standardized air-quality dictionary (see
            :mod:`~src.feature_pipeline.fetch_air_quality`).

    Returns:
        A single flat dict containing all fields required by the output
        dataset schema.
    """
    return {
        "timestamp": weather["timestamp"],
        "collection_timestamp": weather.get("collection_timestamp", weather["timestamp"]),
        "city": weather["city"],
        "latitude": weather["latitude"],
        "longitude": weather["longitude"],
        "temperature": weather["temperature"],
        "feels_like": weather.get("feels_like"),
        "humidity": weather["humidity"],
        "pressure": weather["pressure"],
        "wind_speed": weather.get("wind_speed"),
        "wind_direction": weather.get("wind_direction"),
        "clouds": weather.get("clouds"),
        "visibility": weather.get("visibility"),
        "weather": weather["weather"],
        "aqi": aqi["aqi"],
        "pm2_5": aqi["pm2_5"],
        "pm10": aqi["pm10"],
        "co": aqi["co"],
        "so2": aqi["so2"],
        "no2": aqi["no2"],
        "o3": aqi["o3"],
        "weather_source": weather.get("source"),
        "aqi_source": aqi.get("source"),
    }


def process_features(
    weather: dict[str, Any],
    aqi: dict[str, Any],
) -> dict[str, Any]:
    """Core reusable feature-processing pipeline.

    This function is the heart of the project.  It is called **identically**
    by both :mod:`~src.feature_pipeline.live_pipeline` and
    :mod:`~src.feature_pipeline.historical_backfill`, so that feature
    processing logic never needs to be duplicated.

    Workflow:

        1. Merge weather + AQI data into a single record.
        2. Validate the merged record (required fields, datatypes, ranges,
           timestamps, coordinates).
        3. Return the processed feature dict ready for storage.

    Args:
        weather: Standardized weather dictionary.
        aqi:     Standardized air-quality dictionary.

    Returns:
        A validated, normalized feature dict.

    Raises:
        PipelineError: If merging or validation fails.
    """
    logger.info("Feature pipeline: merging weather and AQI data")

    try:
        merged = merge_weather_and_aqi(weather, aqi)
    except KeyError as exc:
        raise PipelineError(
            f"Failed to merge weather and AQI records: missing key {exc}"
        ) from exc

    logger.info(
        "Feature pipeline: validating merged record for city=%s",
        merged.get("city"),
    )
    try:
        validate_record(merged)
    except Exception as exc:
        raise PipelineError(
            f"Feature pipeline: validation failed for merged record: {exc}"
        ) from exc

    logger.info(
        "Feature pipeline: processing complete for timestamp=%s",
        merged.get("timestamp"),
    )
    return merged
