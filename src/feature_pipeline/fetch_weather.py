"""
Weather data fetcher backed by OpenWeather (live) and Open-Meteo (historical).

Downloads current weather and forecast data from OpenWeather.
Downloads historical weather from the Open-Meteo Archive API, which is free
and does not require a paid subscription (unlike OpenWeather One Call 3.0).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import requests

from configs.config import Settings
from src.common.exceptions import APIError
from src.common.logger import get_logger

logger = get_logger(__name__)

CURRENT_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
HISTORICAL_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

SOURCE_NAME = "OpenWeather"
SOURCE_HISTORICAL = "Open-Meteo Archive"


class WeatherFetcher:
    """Fetches current weather and forecast data from the OpenWeather API.

    Attributes:
        settings: Loaded pipeline settings (API key, coordinates, retry/timeout config).
        session: A `requests.Session` reused across calls for connection pooling.
    """

    def __init__(self, settings: Settings, session: requests.Session | None = None):
        self.settings = settings
        self.session = session or requests.Session()

    def _request_with_retry(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        """Issue a GET request with retry-on-failure and a fixed timeout.

        Args:
            url: Endpoint URL.
            params: Query parameters (API key is added by the caller).

        Returns:
            The parsed JSON response body.

        Raises:
            APIError: If every retry attempt fails, or the response is not
                valid JSON, or the HTTP status indicates failure.
        """
        last_exception: Exception | None = None

        for attempt in range(1, self.settings.max_retries + 1):
            try:
                logger.info(
                    "Requesting %s (attempt %d/%d) params=%s",
                    url,
                    attempt,
                    self.settings.max_retries,
                    {k: v for k, v in params.items() if k != "appid"},
                )
                response = self.session.get(
                    url, params=params, timeout=self.settings.request_timeout_seconds
                )
                logger.info("Response status from %s: %d", url, response.status_code)

                if response.status_code == 200:
                    return response.json()

                # Non-200: decide whether it's worth retrying (5xx / 429) or fatal (4xx).
                if response.status_code in (429, 500, 502, 503, 504):
                    last_exception = APIError(
                        f"Transient error from {url}",
                        source=SOURCE_NAME,
                        status_code=response.status_code,
                    )
                else:
                    raise APIError(
                        f"Non-retryable error from {url}: {response.text[:200]}",
                        source=SOURCE_NAME,
                        status_code=response.status_code,
                    )

            except requests.exceptions.Timeout as exc:
                last_exception = APIError(f"Request timed out: {exc}", source=SOURCE_NAME)
            except requests.exceptions.RequestException as exc:
                last_exception = APIError(f"Request failed: {exc}", source=SOURCE_NAME)
            except ValueError as exc:
                last_exception = APIError(f"Invalid JSON response: {exc}", source=SOURCE_NAME)

            if attempt < self.settings.max_retries:
                backoff = self.settings.retry_backoff_seconds * (2 ** (attempt - 1))
                logger.info("Retrying %s in %.1fs...", url, backoff)
                time.sleep(backoff)

        logger.error("All %d attempts failed for %s", self.settings.max_retries, url)
        raise last_exception or APIError(f"Unknown failure calling {url}", source=SOURCE_NAME)

    def fetch_historical_weather(
        self,
        latitude: float | None = None,
        longitude: float | None = None,
        timestamp: datetime | int | None = None,
    ) -> dict[str, Any]:
        """Fetch historical weather conditions for a given location and time.

        Uses the **Open-Meteo Archive API** (free, no API key required).
        Accepts a ``datetime``, a Unix epoch ``int``, or ``None`` (defaults to
        now).  The API returns hourly data for the full day containing the
        requested timestamp; the hour closest to the requested time is
        extracted and standardised.

        Args:
            latitude:  Latitude (defaults to ``settings.latitude``).
            longitude: Longitude (defaults to ``settings.longitude``).
            timestamp: ``datetime`` (auto-converted) or Unix epoch ``int`` for
                the desired observation time.

        Returns:
            Standardized weather dictionary (same schema as
            :meth:`fetch_current_weather`).

        Raises:
            APIError: If the request fails after all retries, or the response
                is missing expected fields.
        """
        lat = latitude if latitude is not None else self.settings.latitude
        lon = longitude if longitude is not None else self.settings.longitude

        if isinstance(timestamp, datetime):
            dt = timestamp
        elif timestamp is not None:
            dt = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
        else:
            dt = datetime.now(timezone.utc)

        date_str = dt.strftime("%Y-%m-%d")
        params: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date_str,
            "end_date": date_str,
            "hourly": (
                "temperature_2m,apparent_temperature,relative_humidity_2m,surface_pressure,"
                "wind_speed_10m,wind_direction_10m,cloud_cover,"
                "precipitation,visibility,weather_code"
            ),
            "timezone": "GMT",
        }
        raw = self._request_with_retry(HISTORICAL_WEATHER_URL, params)
        return self._standardize_historical_weather(raw, lat, lon, dt)

    @staticmethod
    def _map_wmo_weather_code(code: int) -> str:
        """Map a WMO weather interpretation code to a human-readable description.

        See https://open-meteo.com/en/docs#weathervariables for the full table.
        """
        mapping: dict[int, str] = {
            0: "clear sky",
            1: "mainly clear",
            2: "partly cloudy",
            3: "overcast",
            45: "foggy",
            48: "depositing rime fog",
            51: "light drizzle",
            53: "moderate drizzle",
            55: "dense drizzle",
            56: "light freezing drizzle",
            57: "dense freezing drizzle",
            61: "slight rain",
            63: "moderate rain",
            65: "heavy rain",
            66: "light freezing rain",
            67: "heavy freezing rain",
            71: "slight snow fall",
            73: "moderate snow fall",
            75: "heavy snow fall",
            77: "snow grains",
            80: "slight rain showers",
            81: "moderate rain showers",
            82: "violent rain showers",
            85: "slight snow showers",
            86: "heavy snow showers",
            95: "thunderstorm",
            96: "thunderstorm with slight hail",
            99: "thunderstorm with heavy hail",
        }
        return mapping.get(code, "Unknown")

    def _standardize_historical_weather(
        self, raw: dict[str, Any], lat: float, lon: float, requested_dt: datetime
    ) -> dict[str, Any]:
        """Normalize a raw Open-Meteo Archive response into the pipeline's
        standard schema.

        The Open-Meteo Archive returns hourly data for a full day.  This method
        extracts the hourly entry nearest to *requested_dt* and maps every
        available variable into the pipeline's field names.

        Args:
            raw:          The raw JSON body returned by the Open-Meteo API.
            lat:          Latitude used for the request (fallback).
            lon:          Longitude used for the request (fallback).
            requested_dt: The exact ``datetime`` the caller asked for.

        Returns:
            A flat dictionary with standardized field names.

        Raises:
            APIError: If required keys are missing from the payload or the
                requested time cannot be found in the returned data.
        """
        try:
            hourly = raw["hourly"]
            times: list[str] = hourly["time"]

            requested_iso = requested_dt.strftime("%Y-%m-%dT%H:%M")
            idx: int | None = None
            for i, t in enumerate(times):
                if t.startswith(requested_iso):
                    idx = i
                    break

            if idx is None:
                requested_ts = requested_dt.timestamp()
                closest_diff = float("inf")
                for i, t in enumerate(times):
                    try:
                        t_dt = datetime.fromisoformat(t)
                        diff = abs(t_dt.replace(tzinfo=timezone.utc).timestamp() - requested_ts)
                        if diff < closest_diff:
                            closest_diff = diff
                            idx = i
                    except ValueError:
                        continue

            if idx is None:
                raise APIError(
                    "Could not find matching hourly entry in Open-Meteo response",
                    source=SOURCE_HISTORICAL,
                )

            weather_codes = hourly.get("weather_code", [])
            weather_code = weather_codes[idx] if idx < len(weather_codes) else None
            description = self._map_wmo_weather_code(weather_code) if weather_code is not None else "N/A"

            return {
                "timestamp": requested_dt.isoformat(),
                "city": self.settings.default_city,
                "latitude": raw.get("latitude", lat),
                "longitude": raw.get("longitude", lon),
                "temperature": hourly["temperature_2m"][idx],
                "feels_like": (
                    hourly.get("apparent_temperature", [None])[idx]
                    if idx < len(hourly.get("apparent_temperature", []))
                    else None
                ),
                "humidity": hourly["relative_humidity_2m"][idx],
                "pressure": hourly["surface_pressure"][idx],
                "visibility": hourly.get("visibility", [None])[idx] if idx < len(hourly.get("visibility", [])) else None,
                "wind_speed": hourly["wind_speed_10m"][idx],
                "wind_direction": hourly["wind_direction_10m"][idx],
                "clouds": hourly["cloud_cover"][idx],
                "precipitation": hourly.get("precipitation", [None])[idx] if idx < len(hourly.get("precipitation", [])) else None,
                "weather": description,
                "source": SOURCE_HISTORICAL,
            }
        except (KeyError, IndexError, TypeError) as exc:
            raise APIError(
                f"Unexpected historical weather response structure: {exc}",
                source=SOURCE_HISTORICAL,
            ) from exc

    def fetch_current_weather(self) -> dict[str, Any]:
        """Fetch current weather conditions for the configured coordinates.

        Returns:
            Standardized weather dictionary. See module-level schema in
            `_standardize_current_weather`.

        Raises:
            APIError: If the request fails after all retries, or the response
                is missing expected fields.
        """
        params = {
            "lat": self.settings.latitude,
            "lon": self.settings.longitude,
            "appid": self.settings.openweather_api_key,
            "units": "metric",
        }
        raw = self._request_with_retry(CURRENT_WEATHER_URL, params)
        return self._standardize_current_weather(raw)

    def fetch_forecast(self) -> dict[str, Any]:
        """Fetch 5-day/3-hour forecast data for the configured coordinates.

        Returns:
            A dict with the raw forecast payload alongside a `fetched_at`
            timestamp and normalized top-level `city` metadata. Forecast data
            is inherently a list of many entries, so it is not flattened into
            the single-record schema used for current weather.

        Raises:
            APIError: If the request fails after all retries.
        """
        params = {
            "lat": self.settings.latitude,
            "lon": self.settings.longitude,
            "appid": self.settings.openweather_api_key,
            "units": "metric",
        }
        raw = self._request_with_retry(FORECAST_URL, params)

        try:
            return {
                "city": raw["city"]["name"],
                "latitude": raw["city"]["coord"]["lat"],
                "longitude": raw["city"]["coord"]["lon"],
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "forecast_entries": raw.get("list", []),
            }
        except KeyError as exc:
            raise APIError(
                f"Unexpected forecast response structure, missing key: {exc}",
                source=SOURCE_NAME,
            ) from exc

    def _standardize_current_weather(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Normalize a raw OpenWeather 'current weather' payload into the
        pipeline's standard schema.

        Args:
            raw: The raw JSON body returned by OpenWeather.

        Returns:
            A flat dictionary with standardized field names.

        Raises:
            APIError: If required keys are missing from the payload.
        """
        try:
            main = raw["main"]
            wind = raw.get("wind", {})
            clouds = raw.get("clouds", {})
            sys_ = raw.get("sys", {})
            weather_list = raw.get("weather", [])
            description = weather_list[0]["description"] if weather_list else None

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "city": raw.get("name", self.settings.default_city),
                "latitude": raw["coord"]["lat"],
                "longitude": raw["coord"]["lon"],
                "temperature": main["temp"],
                "feels_like": main.get("feels_like"),
                "humidity": main["humidity"],
                "pressure": main["pressure"],
                "visibility": raw.get("visibility"),
                "wind_speed": wind.get("speed"),
                "wind_direction": wind.get("deg"),
                "clouds": clouds.get("all"),
                "sunrise": sys_.get("sunrise"),
                "sunset": sys_.get("sunset"),
                "weather": description,
                "source": SOURCE_NAME,
            }
        except KeyError as exc:
            raise APIError(
                f"Unexpected weather response structure, missing key: {exc}",
                source=SOURCE_NAME,
            ) from exc


def fetch_weather_data(settings: Settings) -> dict[str, Any]:
    """Convenience function: fetch current weather using a fresh `WeatherFetcher`.

    Args:
        settings: Loaded pipeline settings.

    Returns:
        Standardized current-weather dictionary.
    """
    fetcher = WeatherFetcher(settings)
    return fetcher.fetch_current_weather()
