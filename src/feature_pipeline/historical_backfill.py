from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from configs.config import Settings, load_settings
from src.common.exceptions import AQIPipelineError, PipelineError, APIError
from src.common.logger import get_logger
from src.feature_pipeline.fetch_air_quality import AirQualityFetcher
from src.feature_pipeline.feature_pipeline import process_features
from src.feature_pipeline.fetch_weather import WeatherFetcher
from src.utils.storage import save_csv, save_json, save_metadata, load_csv

logger = get_logger(__name__)

HISTORICAL_INTERVAL_HOURS = 3
HISTORICAL_INTERVAL_SECONDS = HISTORICAL_INTERVAL_HOURS * 3600
HISTORICAL_BATCH_SIZE = 1


def _align_to_interval(dt: datetime, interval_hours: int) -> datetime:
    """Round *dt* down to the nearest *interval_hours* boundary.

    Example with ``interval_hours=3``::

        2024-01-01T01:23:45  ->  2024-01-01T00:00:00
        2024-01-01T03:00:00  ->  2024-01-01T03:00:00
        2024-01-01T04:59:59  ->  2024-01-01T03:00:00
    """
    aligned_hour = (dt.hour // interval_hours) * interval_hours
    return dt.replace(hour=aligned_hour, minute=0, second=0, microsecond=0)


def _build_seen_keys(processed_path: Path) -> set[tuple[str, str]]:
    """Load existing ``(collection_timestamp, city)`` pairs from a historical
    CSV file so that re-running the backfill does not produce duplicate rows.

    Args:
        processed_path: Path to the historical CSV file.

    Returns:
        A set of ``(collection_timestamp_string, city)`` tuples for rows that
        have already been persisted.
    """
    if not processed_path.exists():
        return set()
    try:
        rows = load_csv(processed_path)
        keys: set[tuple[str, str]] = set()
        for row in rows:
            ct = row.get("collection_timestamp", row.get("timestamp", ""))
            city = row.get("city", "")
            if ct and city:
                keys.add((ct, city))
        return keys
    except Exception:
        logger.warning("Could not parse existing CSV for dedup; starting fresh")
        return set()


def _generate_interval_timestamps(
    start: datetime,
    end: datetime,
    interval_hours: int,
) -> list[datetime]:
    """Yield UTC datetimes at fixed *interval_hours* boundaries from
    *start* through *end* (inclusive).

    All returned datetimes are aligned to interval boundaries (e.g. for
    interval=3: 00:00, 03:00, 06:00, ...) and are timezone-aware (UTC).
    """
    aligned = _align_to_interval(start, interval_hours)
    timestamps: list[datetime] = []
    current = aligned
    while current <= end:
        timestamps.append(current)
        current += timedelta(hours=interval_hours)
    return timestamps


def run_historical_backfill(
    start_date: str,
    end_date: str,
    settings: Settings | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Run the historical data backfill for a given date range.

    Workflow
    --------
    1. Parse and validate the date range.
    2. Normalise the start date to the nearest interval boundary and generate
       evenly-spaced timestamps (every ``HISTORICAL_INTERVAL_HOURS``) through
       the end date.
    3. For each timestamp:
       a. Fetch real historical weather via the One Call API 3.0 time-machine.
       b. Fetch real historical AQI via OpenWeather Air Pollution History API
          (AQICN stages are **skipped** -- they return current data, not
          historical, and would silently fabricate history).
       c. Merge and validate through the shared ``process_features()``.
       d. Save to the processed CSV (incremental, never overwrites).
       e. Save raw API payloads as JSON.
    4. Write run metadata to ``data/metadata/``.
    5. Return the list of new records collected this run.

    Every row in the output CSV corresponds to **one real API request** per
    data source.

    Args:
        start_date: ISO-8601 date (e.g. ``"2024-01-01"``) or datetime string.
        end_date:   ISO-8601 date or datetime string.
        settings:   Optional pre-loaded ``Settings``.
        force:      If ``True``, ignore existing rows in the processed CSV
                    and re-fetch/re-save every timestamp in the requested
                    range, even if it was already collected in a previous
                    run. Defaults to ``False`` (idempotent, dedup-by-key
                    behaviour).

    Returns:
        List of processed feature dicts collected during **this** run.

    Raises:
        PipelineError: If the date range is invalid.
    """
    run_start = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info("Historical backfill started: %s to %s", start_date, end_date)
    if force:
        logger.info("Force mode enabled: duplicate check will be bypassed")

    settings = settings or load_settings()

    try:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
    except ValueError as exc:
        raise PipelineError(
            f"Invalid date format: start={start_date!r} end={end_date!r} ({exc})"
        ) from exc

    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    if start > end:
        raise PipelineError(
            f"start_date ({start_date}) must not be after end_date ({end_date})"
        )

    weather_fetcher = WeatherFetcher(settings)
    aqi_fetcher = AirQualityFetcher(settings)

    processed_path = settings.processed_data_dir / "aqi_dataset_historical.csv"
    seen_keys = set() if force else _build_seen_keys(processed_path)

    timestamps = _generate_interval_timestamps(start, end, HISTORICAL_INTERVAL_HOURS)
    if not timestamps:
        logger.warning("No timestamps generated for range %s to %s", start_date, end_date)
        return []

    logger.info(
        "Generated %d collection timestamps (interval=%dh)",
        len(timestamps),
        HISTORICAL_INTERVAL_HOURS,
    )

    current: list[dict[str, Any]] = []
    batch_buffer: list[dict[str, Any]] = []
    step_count = 0
    error_count = 0
    skipped_count = 0
    new_record_count = 0
    api_errors: list[str] = []

    actual_weather_sources: set[str] = set()
    actual_aqi_sources: set[str] = set()

    def _flush_batch() -> None:
        if batch_buffer:
            save_csv(batch_buffer, processed_path)
            batch_buffer.clear()

    for current_ts in timestamps:
        collection_ts_iso = current_ts.isoformat()
        logger.info(
            "Backfill step %d/%d: fetching data for %s",
            step_count + 1,
            len(timestamps),
            collection_ts_iso,
        )

        try:
            weather_data = weather_fetcher.fetch_historical_weather(
                latitude=settings.latitude,
                longitude=settings.longitude,
                timestamp=current_ts,
            )
            weather_data["collection_timestamp"] = collection_ts_iso

            aqi_data = aqi_fetcher.fetch_historical_aqi(
                latitude=settings.latitude,
                longitude=settings.longitude,
                timestamp=current_ts,
            )

            merged_record = process_features(weather_data, aqi_data)

            key = (merged_record["collection_timestamp"], merged_record["city"])
            if key in seen_keys:
                skipped_count += 1
                logger.info(
                    "Backfill step %d: duplicate skipped for %s",
                    step_count + 1,
                    collection_ts_iso,
                )
            else:
                seen_keys.add(key)

                actual_weather_sources.add(merged_record.get("weather_source", "Unknown"))
                actual_aqi_sources.add(merged_record.get("aqi_source", "Unknown"))

                current.append(merged_record)
                batch_buffer.append(merged_record)
                new_record_count += 1

                if len(batch_buffer) >= HISTORICAL_BATCH_SIZE:
                    _flush_batch()

                run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                city_slug = str(merged_record["city"]).lower().replace(" ", "_")

                raw_path = (
                    settings.raw_data_dir
                    / f"historical_raw_{city_slug}_{run_timestamp}_step{step_count:05d}.json"
                )
                save_json(
                    {
                        "weather_raw": weather_data,
                        "aqi_raw": aqi_data,
                        "merged": merged_record,
                    },
                    raw_path,
                )

                logger.info(
                    "Backfill step %d: success for %s",
                    step_count + 1,
                    collection_ts_iso,
                )

        except APIError as exc:
            error_count += 1
            api_errors.append(f"[{collection_ts_iso}] {exc}")
            logger.warning(
                "Backfill step %d: API error for %s - %s. Skipping.",
                step_count + 1,
                collection_ts_iso,
                exc,
            )
        except PipelineError as exc:
            error_count += 1
            api_errors.append(f"[{collection_ts_iso}] {exc}")
            logger.warning(
                "Backfill step %d: pipeline error for %s - %s. Skipping.",
                step_count + 1,
                collection_ts_iso,
                exc,
            )

        step_count += 1

    _flush_batch()

    run_end = datetime.now(timezone.utc)
    execution_time_seconds = (run_end - run_start).total_seconds()

    metadata: dict[str, Any] = {
        "run_start": run_start.isoformat(),
        "run_end": run_end.isoformat(),
        "execution_time_seconds": round(execution_time_seconds, 2),
        "pipeline_type": "historical",
        "city": settings.default_city,
        "weather_source": ", ".join(sorted(actual_weather_sources)) if actual_weather_sources else "Unknown",
        "aqi_source": ", ".join(sorted(actual_aqi_sources)) if actual_aqi_sources else "Unknown",
        "interval_hours": HISTORICAL_INTERVAL_HOURS,
        "total_requested_steps": len(timestamps),
        "successful_records": new_record_count,
        "skipped_duplicates": skipped_count,
        "error_count": error_count,
        "start_date": start_date,
        "end_date": end_date,
        "force_mode": force,
        "errors": api_errors[:20],
        "status": "SUCCESS" if error_count == 0 else "PARTIAL",
    }
    save_metadata(metadata, pipeline_type="historical")

    logger.info(
        "Historical backfill completed: %d/%d steps, %d new, %d skipped, %d errors (%.1fs)",
        new_record_count + skipped_count + error_count,
        len(timestamps),
        new_record_count,
        skipped_count,
        error_count,
        execution_time_seconds,
    )
    logger.info("=" * 60)

    return current


def main() -> int:
    """CLI entry point.

    Expects two positional arguments: ``start_date`` and ``end_date``, plus
    an optional ``--force`` flag.

    Usage::

        python -m src.feature_pipeline.historical_backfill 2024-01-01 2024-06-30
        python -m src.feature_pipeline.historical_backfill 2024-01-01 2024-06-30 --force

    The ``--force`` flag bypasses the duplicate-timestamp check, so rows for
    dates already present in ``aqi_dataset_historical.csv`` are re-fetched
    and re-saved (appended) instead of being skipped.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    args = [a for a in sys.argv[1:] if a != "--force"]
    force = "--force" in sys.argv[1:]

    if len(args) < 2:
        print("Usage: python -m src.feature_pipeline.historical_backfill "
              "<start_date> <end_date> [--force]")
        print("Example: python -m src.feature_pipeline.historical_backfill "
              "2024-01-01 2024-06-30")
        print("Force re-fetch of already-collected dates: add --force")
        return 1

    start_date = args[0]
    end_date = args[1]

    try:
        records = run_historical_backfill(start_date, end_date, force=force)
        print(f"Historical backfill completed: {len(records)} new records collected.")
        return 0
    except AQIPipelineError as exc:
        logger.error("Historical backfill failed: %s", exc)
        print(f"Historical backfill failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Historical backfill failed with an unexpected error")
        print(
            f"Historical backfill failed with an unexpected error: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())