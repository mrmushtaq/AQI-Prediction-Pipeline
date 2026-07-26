from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.common.exceptions import PipelineError, APIError
from src.feature_pipeline.historical_backfill import run_historical_backfill


def _make_weather(dt: datetime) -> dict:
    ts = dt.isoformat()
    return {
        "timestamp": ts,
        "city": "Sukkur",
        "latitude": 27.7052,
        "longitude": 68.8574,
        "temperature": 20.0,
        "feels_like": None,
        "humidity": 50,
        "pressure": 1013,
        "wind_speed": 2.0,
        "wind_direction": 90,
        "clouds": 20,
        "visibility": 10000,
        "precipitation": 0.0,
        "weather": "partly cloudy",
        "source": "Open-Meteo Archive",
    }


def _make_aqi(dt: datetime) -> dict:
    ts = dt.isoformat()
    return {
        "timestamp": ts,
        "city": "Sukkur",
        "latitude": 27.7052,
        "longitude": 68.8574,
        "aqi": 100,
        "pm2_5": 50,
        "pm10": 80,
        "co": 3,
        "so2": 1,
        "no2": 8,
        "o3": 12,
        "source": "OpenAQ-v3",
    }


def _gen_side_effect(start_ts_str: str, interval_hours: int = 3):
    """Return side-effect closures for weather and AQI mocks that generate
    per-step data with the correct timestamp based on call count.
    """
    base = datetime.fromisoformat(start_ts_str).replace(tzinfo=timezone.utc)
    call_index = [0]

    def weather_side(*args, **kwargs):
        dt = base + __import__("datetime").timedelta(
            hours=call_index[0] * interval_hours
        )
        return _make_weather(dt)

    def aqi_side(*args, **kwargs):
        dt = base + __import__("datetime").timedelta(
            hours=call_index[0] * interval_hours
        )
        call_index[0] += 1
        return _make_aqi(dt)

    return weather_side, aqi_side


def _gen_weather_side_for(start_ts_str: str, interval_hours: int = 3):
    base = datetime.fromisoformat(start_ts_str).replace(tzinfo=timezone.utc)
    call_index = [0]

    def weather_side(*args, **kwargs):
        dt = base + __import__("datetime").timedelta(
            hours=call_index[0] * interval_hours
        )
        call_index[0] += 1
        return _make_weather(dt)

    return weather_side


def test_backfill_single_step(settings, tmp_path):
    settings = settings.__class__(
        **{**settings.__dict__, "raw_data_dir": tmp_path / "raw",
           "processed_data_dir": tmp_path / "processed"}
    )

    w, a = _gen_side_effect("2025-01-01T00:00:00")

    with (
        patch(
            "src.feature_pipeline.historical_backfill.WeatherFetcher.fetch_historical_weather",
            side_effect=w,
        ),
        patch(
            "src.feature_pipeline.historical_backfill.AirQualityFetcher.fetch_historical_aqi",
            side_effect=a,
        ),
    ):
        records = run_historical_backfill(
            "2025-01-01T00:00:00", "2025-01-01T00:00:00", settings
        )

    assert len(records) == 1
    assert records[0]["city"] == "Sukkur"
    assert records[0]["aqi"] == 100

    csv_path = tmp_path / "processed" / "aqi_dataset_historical.csv"
    assert csv_path.exists()


def test_backfill_multiple_steps(settings, tmp_path):
    settings = settings.__class__(
        **{**settings.__dict__, "raw_data_dir": tmp_path / "raw",
           "processed_data_dir": tmp_path / "processed"}
    )

    w, a = _gen_side_effect("2025-01-01T00:00:00")

    with (
        patch(
            "src.feature_pipeline.historical_backfill.WeatherFetcher.fetch_historical_weather",
            side_effect=w,
        ),
        patch(
            "src.feature_pipeline.historical_backfill.AirQualityFetcher.fetch_historical_aqi",
            side_effect=a,
        ),
    ):
        records = run_historical_backfill(
            "2025-01-01T00:00:00", "2025-01-01T06:00:00", settings
        )

    assert len(records) == 3


def test_backfill_skips_on_api_error(settings, tmp_path):
    settings = settings.__class__(
        **{**settings.__dict__, "raw_data_dir": tmp_path / "raw",
           "processed_data_dir": tmp_path / "processed"}
    )

    w_base = _gen_weather_side_for("2025-01-01T00:00:00")
    call_count = [0]

    def weather_side(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 2:
            raise APIError("Transient API failure", source="test")
        return w_base(*args, **kwargs)

    aqi_data = _make_aqi(datetime(2025, 1, 1, tzinfo=timezone.utc))

    with (
        patch(
            "src.feature_pipeline.historical_backfill.WeatherFetcher.fetch_historical_weather",
            side_effect=weather_side,
        ),
        patch(
            "src.feature_pipeline.historical_backfill.AirQualityFetcher.fetch_historical_aqi",
            return_value=aqi_data,
        ),
    ):
        records = run_historical_backfill(
            "2025-01-01T00:00:00", "2025-01-01T06:00:00", settings
        )

    assert len(records) == 2


def test_backfill_dedup_skips_existing_timestamps(settings, tmp_path):
    settings = settings.__class__(
        **{**settings.__dict__, "raw_data_dir": tmp_path / "raw",
           "processed_data_dir": tmp_path / "processed"}
    )

    w1, a1 = _gen_side_effect("2025-01-01T00:00:00")
    w2, a2 = _gen_side_effect("2025-01-01T00:00:00")

    with (
        patch(
            "src.feature_pipeline.historical_backfill.WeatherFetcher.fetch_historical_weather",
            side_effect=w1,
        ),
        patch(
            "src.feature_pipeline.historical_backfill.AirQualityFetcher.fetch_historical_aqi",
            side_effect=a1,
        ),
    ):
        first = run_historical_backfill(
            "2025-01-01T00:00:00", "2025-01-01T06:00:00", settings
        )
    assert len(first) == 3

    with (
        patch(
            "src.feature_pipeline.historical_backfill.WeatherFetcher.fetch_historical_weather",
            side_effect=w2,
        ),
        patch(
            "src.feature_pipeline.historical_backfill.AirQualityFetcher.fetch_historical_aqi",
            side_effect=a2,
        ),
    ):
        second = run_historical_backfill(
            "2025-01-01T00:00:00", "2025-01-01T06:00:00", settings
        )
    assert len(second) == 0


def test_backfill_dedup_partial_overlap(settings, tmp_path):
    settings = settings.__class__(
        **{**settings.__dict__, "raw_data_dir": tmp_path / "raw",
           "processed_data_dir": tmp_path / "processed"}
    )

    w1, a1 = _gen_side_effect("2025-01-01T00:00:00")
    w2, a2 = _gen_side_effect("2025-01-01T03:00:00")

    with (
        patch(
            "src.feature_pipeline.historical_backfill.WeatherFetcher.fetch_historical_weather",
            side_effect=w1,
        ),
        patch(
            "src.feature_pipeline.historical_backfill.AirQualityFetcher.fetch_historical_aqi",
            side_effect=a1,
        ),
    ):
        first = run_historical_backfill(
            "2025-01-01T00:00:00", "2025-01-01T03:00:00", settings
        )
    assert len(first) == 2

    with (
        patch(
            "src.feature_pipeline.historical_backfill.WeatherFetcher.fetch_historical_weather",
            side_effect=w2,
        ),
        patch(
            "src.feature_pipeline.historical_backfill.AirQualityFetcher.fetch_historical_aqi",
            side_effect=a2,
        ),
    ):
        second = run_historical_backfill(
            "2025-01-01T03:00:00", "2025-01-01T09:00:00", settings
        )
    assert len(second) == 2


def test_backfill_rejects_invalid_date_range(settings):
    with pytest.raises(PipelineError, match="must not be after"):
        run_historical_backfill("2025-06-01", "2025-01-01", settings)


def test_backfill_rejects_malformed_dates(settings):
    with pytest.raises(PipelineError):
        run_historical_backfill("not-a-date", "2025-01-01", settings)
