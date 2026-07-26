"""Shared pytest fixtures for the AQI pipeline test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the project root is importable as `configs.*` / `src.*` regardless of
# the directory pytest is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import Settings  # noqa: E402


@pytest.fixture
def settings(tmp_path) -> Settings:
    """A fully-populated, fast-failing Settings instance for tests.

    Retries are set to 1 with zero backoff so failure-path tests run instantly.
    """
    return Settings(
        aqicn_token="test-aqicn-token",
        openweather_api_key="test-openweather-key",
        openaq_api_key="test-openaq-key",
        default_city="Sukkur",
        latitude=27.7052,
        longitude=68.8574,
        log_level="DEBUG",
        request_timeout_seconds=5.0,
        max_retries=1,
        retry_backoff_seconds=0.0,
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        metadata_dir=tmp_path / "metadata",
    )
