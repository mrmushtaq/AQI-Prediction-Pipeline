"""
Centralized configuration loader for the AQI Prediction data ingestion pipeline.

Every configurable value is loaded from environment variables (populated from
a `.env` file via `python-dotenv`). No secrets or environment-specific values
should ever be hardcoded elsewhere in the codebase -- import `settings` from
this module instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from src.common.exceptions import ConfigurationError

# Resolve project root (two levels up from this file: configs/config.py -> root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load variables from a .env file at the project root, if present. This does
# NOT override variables already set in the real process environment, which
# keeps behaviour predictable in CI/production where env vars are injected
# directly.
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
LOGS_DIR = PROJECT_ROOT / "logs"

# HTTP behaviour shared by all external API clients
DEFAULT_REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "10"))
DEFAULT_MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
DEFAULT_RETRY_BACKOFF_SECONDS = float(os.getenv("RETRY_BACKOFF_SECONDS", "2"))


def _get_env(key: str, *, required: bool = True, default: str | None = None) -> str | None:
    """Fetch an environment variable, optionally enforcing that it is present
    and non-empty.

    Args:
        key: Environment variable name.
        required: If True, raises ConfigurationError when missing/empty.
        default: Fallback value used when the variable is absent and not required.

    Returns:
        The environment variable's string value, or `default` if not required
        and unset.

    Raises:
        ConfigurationError: If `required` is True and the variable is missing
            or empty.
    """
    value = os.getenv(key)
    if value is None:
        try:
            import streamlit as st

            value = st.secrets.get(key, default)
        except Exception:
            value = default
    if required and (value is None or value.strip() == ""):
        raise ConfigurationError(
            f"Missing required environment variable '{key}'. "
            f"Please set it in your .env file (see .env.example)."
        )
    return value


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of all pipeline configuration.

    Attributes:
        aqicn_token: API token for the AQICN (World Air Quality Index) API.
        openweather_api_key: API key for OpenWeather (current weather,
            forecast, and Air Pollution API fallback).
        default_city: Human-readable city name used for AQICN lookups and
            dataset labeling.
        latitude: Latitude used for OpenWeather lookups.
        longitude: Longitude used for OpenWeather lookups.
        log_level: Root logging level (e.g. INFO, DEBUG).
        request_timeout_seconds: Timeout applied to every outbound HTTP request.
        max_retries: Maximum retry attempts per external API call.
        retry_backoff_seconds: Base backoff (seconds) between retries; grows
            linearly with attempt number.
    """

    aqicn_token: str
    openweather_api_key: str
    openaq_api_key: str
    default_city: str
    latitude: float
    longitude: float
    log_level: str = "INFO"
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS
    raw_data_dir: Path = field(default_factory=lambda: DATA_RAW_DIR)
    processed_data_dir: Path = field(default_factory=lambda: DATA_PROCESSED_DIR)
    metadata_dir: Path = field(default_factory=lambda: DATA_METADATA_DIR)


def load_settings() -> Settings:
    """Load and validate all configuration values from the environment.

    Returns:
        A populated, validated `Settings` instance.

    Raises:
        ConfigurationError: If any required variable is missing, or if
            latitude/longitude cannot be parsed as floats or are out of range.
    """
    aqicn_token = _get_env("AQICN_TOKEN")
    openweather_api_key = _get_env("OPENWEATHER_API_KEY")
    openaq_api_key = _get_env("OPENAQ_API_KEY")
    default_city = _get_env("DEFAULT_CITY")
    lat_raw = _get_env("OPENWEATHER_LAT")
    lon_raw = _get_env("OPENWEATHER_LON")
    log_level = _get_env("LOG_LEVEL", required=False, default="INFO")

    try:
        latitude = float(lat_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"OPENWEATHER_LAT must be a valid float, got: {lat_raw!r}"
        ) from exc

    try:
        longitude = float(lon_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"OPENWEATHER_LON must be a valid float, got: {lon_raw!r}"
        ) from exc

    if not (-90.0 <= latitude <= 90.0):
        raise ConfigurationError(f"OPENWEATHER_LAT out of range [-90, 90]: {latitude}")
    if not (-180.0 <= longitude <= 180.0):
        raise ConfigurationError(f"OPENWEATHER_LON out of range [-180, 180]: {longitude}")

    return Settings(
        aqicn_token=aqicn_token,
        openweather_api_key=openweather_api_key,
        openaq_api_key=openaq_api_key,
        default_city=default_city,
        latitude=latitude,
        longitude=longitude,
        log_level=log_level or "INFO",
    )

@dataclass(frozen=True)
class HopsworksSettings:
    """Configuration needed to connect to a Hopsworks Feature Store.

    Kept as a separate, standalone settings object (rather than folded into
    `Settings`) so that the existing live/historical ingestion pipelines --
    and every test that constructs a `Settings` -- are completely unaffected
    by Phase 3. Hopsworks credentials are only required when actually running
    the feature-store sync (`src.feature_pipeline.feature_store`).

    Attributes:
        api_key: Hopsworks API key (Account Settings -> API Keys on
            https://app.hopsworks.ai).
        project_name: Name of the Hopsworks project to connect to.
    """

    api_key: str
    project_name: str


def load_hopsworks_settings() -> HopsworksSettings:
    """Load and validate Hopsworks connection settings from the environment.

    Returns:
        A populated `HopsworksSettings` instance.

    Raises:
        ConfigurationError: If `HOPSWORKS_API_KEY` or `HOPSWORKS_PROJECT_NAME`
            is missing or empty.
    """
    api_key = _get_env("HOPSWORKS_API_KEY")
    project_name = _get_env("HOPSWORKS_PROJECT_NAME")
    return HopsworksSettings(api_key=api_key, project_name=project_name)