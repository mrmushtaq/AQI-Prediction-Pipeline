from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any

from configs.config import Settings, load_settings
from src.common.exceptions import AQIPipelineError, PipelineError
from src.common.logger import get_logger
from src.feature_pipeline.fetch_air_quality import AirQualityFetcher
from src.feature_pipeline.feature_pipeline import process_features
from src.feature_pipeline.fetch_weather import WeatherFetcher
from src.utils.storage import save_csv, save_json, save_metadata

logger = get_logger(__name__)


def run_live_pipeline(settings: Settings | None = None) -> dict[str, Any]:
    """Run the live (current-time) data ingestion pipeline end-to-end.

    Workflow:

        1. Load configuration.
        2. Fetch current weather.
        3. Fetch current AQI (via three-stage fallback).
        4. Run the reusable feature pipeline (merge + validate).
        5. Save raw API responses as JSON.
        6. Append processed record to CSV dataset.
        7. Save pipeline metadata.

    Args:
        settings: Optional pre-loaded ``Settings``.  If omitted, configuration
            is loaded from the environment.

    Returns:
        The final merged and validated record that was persisted.

    Raises:
        AQIPipelineError: If any stage fails.
    """
    logger.info("=" * 60)
    logger.info("Live pipeline started")

    settings = settings or load_settings()
    logger.info(
        "Configuration loaded: city=%s lat=%s lon=%s",
        settings.default_city,
        settings.latitude,
        settings.longitude,
    )

    try:
        weather_fetcher = WeatherFetcher(settings)
        weather_data = weather_fetcher.fetch_current_weather()
        logger.info("Weather data fetched successfully from %s", weather_data.get("source"))

        aqi_fetcher = AirQualityFetcher(settings)
        aqi_data = aqi_fetcher.fetch_air_quality()
        logger.info("AQI data fetched successfully from %s", aqi_data.get("source"))

        merged_record = process_features(weather_data, aqi_data)
        logger.info("Feature pipeline completed successfully")

        run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        city_slug = str(merged_record["city"]).lower().replace(" ", "_")

        raw_path = settings.raw_data_dir / f"live_raw_{city_slug}_{run_timestamp}.json"
        save_json(
            {"weather_raw": weather_data, "aqi_raw": aqi_data, "merged": merged_record},
            raw_path,
        )

        processed_path = settings.processed_data_dir / "aqi_dataset.csv"
        save_csv([merged_record], processed_path)

        metadata = {
            "run_time": datetime.now(timezone.utc).isoformat(),
            "pipeline_type": "live",
            "city": merged_record["city"],
            "weather_source": merged_record.get("weather_source"),
            "aqi_source": merged_record.get("aqi_source"),
            "records": 1,
            "status": "SUCCESS",
        }
        save_metadata(metadata, pipeline_type="live")

        logger.info("Live pipeline completed successfully")
        logger.info("=" * 60)

        return merged_record

    except AQIPipelineError:
        raise
    except Exception as exc:
        raise PipelineError(f"Live pipeline failed with unexpected error: {exc}") from exc


def main() -> int:
    """CLI entry point. Returns a process exit code (0 = success, 1 = failure)."""
    try:
        record = run_live_pipeline()
        print("Live pipeline completed successfully. Sample record:")
        print(record)
        return 0
    except AQIPipelineError as exc:
        logger.error("Live pipeline failed: %s", exc)
        print(f"Live pipeline failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Live pipeline failed with an unexpected error")
        print(f"Live pipeline failed with an unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
