"""
Main entry point for the data ingestion pipeline.

Workflow:
    Load configuration
        -> Fetch weather
        -> Fetch AQI
        -> Validate merged record
        -> Save raw JSON
        -> Save processed CSV
        -> Save run metadata
        -> Log completion

A metadata file is written to `data/metadata/` for *every* run attempt --
whether it succeeds or fails -- so each ingestion run leaves an auditable
record behind (timestamp, city, which provider actually served weather/AQI
data, record count, and status). This is invaluable for debugging failures
after the fact without having to dig through the full log file.

Run directly with:
    python -m src.feature_pipeline.run_pipeline
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any

from configs.config import Settings, load_settings
from src.common.exceptions import AQIPipelineError, StorageError
from src.common.logger import get_logger
from src.feature_pipeline.fetch_air_quality import AirQualityFetcher
from src.feature_pipeline.fetch_weather import WeatherFetcher
from src.utils.data_validator import validate_record
from src.utils.storage import save_csv, save_json

logger = get_logger(__name__)


def merge_weather_and_aqi(weather: dict[str, Any], aqi: dict[str, Any]) -> dict[str, Any]:
    """Merge standardized weather and AQI dictionaries into a single flat
    record matching the pipeline's output dataset schema.

    Weather fields take precedence for `city`/`latitude`/`longitude` metadata
    (they are considered the primary source of truth for location), while all
    AQI-specific measurements come from the AQI record.

    Args:
        weather: Standardized weather dictionary (see `fetch_weather.py`).
        aqi: Standardized air-quality dictionary (see `fetch_air_quality.py`).

    Returns:
        A single flat dict containing the fields required by
        `data_validator.REQUIRED_FIELDS` plus extra descriptive metadata.
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
        "sunrise": weather.get("sunrise"),
        "sunset": weather.get("sunset"),
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


def save_run_metadata(
    settings: Settings,
    *,
    status: str,
    weather_source: str | None = None,
    aqi_source: str | None = None,
    city: str | None = None,
    records: int = 0,
    error: str | None = None,
) -> None:
    """Persist a small JSON metadata record describing this pipeline run.

    Written to `data/metadata/run_<timestamp>.json` for every run attempt --
    success or failure -- so each ingestion run leaves an auditable trail
    (which provider served the data, how many records were written, and
    whether/why it failed) without needing to dig through the full log file.

    Failures while writing metadata are deliberately swallowed (logged only):
    metadata is a diagnostic nice-to-have, and a metadata-write failure must
    never mask or override the pipeline's real success/failure outcome.

    Args:
        settings: Loaded pipeline settings (used for `metadata_dir` and the
            configured default city as a fallback).
        status: `"SUCCESS"` or `"FAILED"`.
        weather_source: Which provider served weather data, if known.
        aqi_source: Which provider/stage served AQI data, if known.
        city: City the run was for. Falls back to `settings.default_city`.
        records: Number of records written to the processed dataset this run.
        error: Human-readable error message, if the run failed.
    """
    run_time = datetime.now(timezone.utc).isoformat()
    metadata = {
        "run_time": run_time,
        "city": city or settings.default_city,
        "weather_source": weather_source,
        "aqi_source": aqi_source,
        "records": records,
        "status": status,
        "error": error,
    }

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metadata_path = settings.metadata_dir / f"run_{run_timestamp}.json"

    try:
        save_json(metadata, metadata_path)
        logger.info("Saved run metadata: %s", metadata_path)
    except StorageError as exc:
        # Metadata is diagnostic, not load-bearing -- never let a metadata
        # write failure mask the pipeline's actual success/failure result.
        logger.warning("Failed to save run metadata (non-fatal): %s", exc)


def run_pipeline(settings: Settings | None = None) -> dict[str, Any]:
    """Execute the full ingestion pipeline end-to-end, once.

    A metadata record is written to `data/metadata/` regardless of outcome:
    on success it captures which providers served the data and how many
    records were written; on failure it captures whatever partial context
    (city, weather/AQI source) was available before the failure occurred,
    plus the error message.

    Args:
        settings: Optional pre-loaded `Settings`. If omitted, configuration is
            loaded from the environment via `load_settings()`.

    Returns:
        The final merged and validated record that was persisted.

    Raises:
        AQIPipelineError: If any stage of the pipeline fails (configuration,
            fetching, validation, or storage). The specific subclass indicates
            which stage failed. A metadata file recording the failure is
            written before this exception propagates.
    """
    logger.info("=" * 60)
    logger.info("Pipeline started")

    # Track whatever context we've gathered so far, so a failure metadata
    # record can still report as much as is known (e.g. weather succeeded
    # but AQI fetching failed).
    weather_source: str | None = None
    aqi_source: str | None = None
    run_city: str | None = None

    try:
        settings = settings or load_settings()
        run_city = settings.default_city
        logger.info(
            "Configuration loaded: city=%s lat=%s lon=%s",
            settings.default_city,
            settings.latitude,
            settings.longitude,
        )

        weather_fetcher = WeatherFetcher(settings)
        weather_data = weather_fetcher.fetch_current_weather()
        weather_source = weather_data.get("source")
        logger.info("Weather data fetched successfully from %s", weather_source)

        aqi_fetcher = AirQualityFetcher(settings)
        aqi_data = aqi_fetcher.fetch_air_quality()
        aqi_source = aqi_data.get("source")
        logger.info("AQI data fetched successfully from %s", aqi_source)

        merged_record = merge_weather_and_aqi(weather_data, aqi_data)
        run_city = merged_record.get("city", run_city)

        validate_record(merged_record)
        logger.info("Validation succeeded for merged record")

        run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        city_slug = str(merged_record["city"]).lower().replace(" ", "_")

        raw_path = settings.raw_data_dir / f"raw_{city_slug}_{run_timestamp}.json"
        save_json(
            {"weather_raw": weather_data, "aqi_raw": aqi_data, "merged": merged_record},
            raw_path,
        )

        processed_path = settings.processed_data_dir / "aqi_dataset.csv"
        save_csv([merged_record], processed_path)

        logger.info("Pipeline completed successfully")
        logger.info("=" * 60)

        save_run_metadata(
            settings,
            status="SUCCESS",
            weather_source=weather_source,
            aqi_source=aqi_source,
            city=run_city,
            records=1,
        )

        return merged_record

    except AQIPipelineError as exc:
        # `settings` may not have loaded successfully (e.g. ConfigurationError),
        # in which case there is nowhere meaningful to write metadata -- skip
        # it rather than raising a second, unrelated exception during failure
        # handling.
        if settings is not None:
            save_run_metadata(
                settings,
                status="FAILED",
                weather_source=weather_source,
                aqi_source=aqi_source,
                city=run_city,
                records=0,
                error=str(exc),
            )
        raise


def main() -> int:
    """CLI entry point. Returns a process exit code (0 = success, 1 = failure)."""
    try:
        record = run_pipeline()
        print("Pipeline completed successfully. Sample record:")
        print(record)
        return 0
    except AQIPipelineError as exc:
        logger.error("Pipeline failed: %s", exc)
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level safety net for unexpected errors
        logger.exception("Pipeline failed with an unexpected error")
        print(f"Pipeline failed with an unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
