"""
Air-quality data fetcher.

Live (current) data uses a three-stage fallback strategy:
    1. AQICN, queried by city name.
    2. AQICN, queried by latitude/longitude (geo-based station lookup).
    3. OpenWeather Air Pollution API -- used only if both AQICN attempts fail.

Historical AQI is fetched from the **OpenAQ API v3**.  The legacy OpenAQ v2
endpoint (``/v2/measurements``) has been retired and returns HTTP 410.  The
v3 API requires an ``X-API-Key`` header and uses a hierarchical lookup:
locations -> sensors -> measurements.

Since OpenAQ returns raw pollutant concentrations rather than a ready-made
AQI value, AQI is computed from concentrations using the **US EPA AQI
breakpoint formula**.

For sparsely-instrumented locations (smaller cities, fewer nearby stations)
the historical lookup widens its search window around the target timestamp
and, if the nearest sensor for a pollutant has no data in that window, falls
through to try other nearby sensors reporting the same pollutant before
giving up on that pollutant entirely. Among all candidate readings found,
the one closest in time to the requested timestamp is used.

Every stage implements retry logic with exponential backoff, a request
timeout, and structured logging. All stages are normalized into the
exact same output schema, so downstream code never needs to know which
provider (or which AQICN query strategy) actually served the data.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from typing import Any

import requests

from configs.config import Settings
from src.common.exceptions import APIError
from src.common.logger import get_logger

logger = get_logger(__name__)

AQICN_CITY_URL_TEMPLATE = "https://api.waqi.info/feed/{city}/"
AQICN_GEO_URL_TEMPLATE = "https://api.waqi.info/feed/geo:{lat};{lon}/"
OPENWEATHER_AIR_POLLUTION_URL = "https://api.openweathermap.org/data/2.5/air_pollution"
OPENAQ_V3_LOCATIONS_URL = "https://api.openaq.org/v3/locations"
OPENAQ_V3_SENSORS_MEASUREMENTS_TEMPLATE = "https://api.openaq.org/v3/sensors/{sensor_id}/measurements"

SOURCE_AQICN_CITY = "AQICN-city"
SOURCE_AQICN_GEO = "AQICN-geo"
SOURCE_OPENWEATHER_FALLBACK = "OpenWeather-AirPollution"
SOURCE_OPENAQ_HISTORICAL = "OpenAQ-v3"
SOURCE_OPENMETEO_HISTORICAL = "OpenMeteo-AirQuality"

# OpenWeather AQI is on a 1-5 scale; AQICN uses the US EPA 0-500 scale.
# We keep both sources' native AQI value as-is and record `source` so
# downstream feature engineering can normalize/convert if needed.

# How far (in hours, each direction) to widen the OpenAQ v3 measurement
# search window around the requested historical timestamp. Sparsely-
# reporting sensors (common outside major cities) may not have a reading
# in a narrow +1h window but will often have one within a wider window.
# Total search window is therefore up to 2 * this value.
OPENAQ_MEASUREMENT_WINDOW_HOURS = 6

# Maximum number of candidate sensors to try per pollutant parameter before
# giving up on that pollutant for this timestamp. Trying more than one
# sensor lets us route around individual sensors with local data gaps.
OPENAQ_MAX_SENSORS_PER_PARAM = 5

# Open-Meteo's Air Quality API (CAMS-model-based, no API key required). Used
# as a fallback when OpenAQ v3 has no ground-sensor data for the requested
# location/time (e.g. before a station's activation date, or for cities
# with no nearby OpenAQ coverage at all). Unlike OpenAQ, this is a MODEL
# ESTIMATE (~40km global resolution), not a direct ground measurement -- see
# the module docstring and README Section 11 for the distinction.
OPENMETEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


class AirQualityFetcher:
    """Fetches air-quality data using a three-stage fallback strategy:
    AQICN by city name -> AQICN by lat/lon -> OpenWeather Air Pollution API.

    Attributes:
        settings: Loaded pipeline settings (API tokens, coordinates, retry/timeout config).
        session: A `requests.Session` reused across calls for connection pooling.
    """

    def __init__(self, settings: Settings, session: requests.Session | None = None):
        self.settings = settings
        self.session = session or requests.Session()

    def _request_with_retry(
        self, url: str, params: dict[str, Any], source: str,
        headers: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Issue a GET request with retry-on-failure (exponential backoff)
        and a fixed timeout.

        Backoff grows exponentially with the attempt number:
        `retry_backoff_seconds * 2 ** (attempt - 1)` (e.g. with a 2s base:
        2s, 4s, 8s, ...), which spaces out retries more aggressively than a
        linear schedule and is kinder to rate-limited/degraded upstreams.

        Args:
            url: Endpoint URL.
            params: Query parameters.
            source: Human-readable source name, used in structured log
                messages/errors so each stage of the fallback chain is
                individually traceable.
            headers: Optional HTTP headers (e.g. ``X-API-Key`` for OpenAQ v3).

        Returns:
            The parsed JSON response body.

        Raises:
            APIError: If every retry attempt fails.
        """
        last_exception: Exception | None = None

        for attempt in range(1, self.settings.max_retries + 1):
            try:
                logger.info(
                    "stage=%s event=request_started url=%s attempt=%d/%d",
                    source,
                    url,
                    attempt,
                    self.settings.max_retries,
                )
                response = self.session.get(
                    url, params=params, headers=headers,
                    timeout=self.settings.request_timeout_seconds,
                )
                logger.info(
                    "stage=%s event=response_received url=%s status_code=%d",
                    source,
                    url,
                    response.status_code,
                )

                if response.status_code == 200:
                    return response.json()

                if response.status_code in (429, 500, 502, 503, 504):
                    last_exception = APIError(
                        f"Transient error from {url}", source=source, status_code=response.status_code
                    )
                else:
                    raise APIError(
                        f"Non-retryable error from {url}: {response.text[:200]}",
                        source=source,
                        status_code=response.status_code,
                    )

            except requests.exceptions.Timeout as exc:
                last_exception = APIError(f"Request timed out: {exc}", source=source)
            except requests.exceptions.RequestException as exc:
                last_exception = APIError(f"Request failed: {exc}", source=source)
            except ValueError as exc:
                last_exception = APIError(f"Invalid JSON response: {exc}", source=source)

            if attempt < self.settings.max_retries:
                backoff = self.settings.retry_backoff_seconds * (2 ** (attempt - 1))
                logger.info(
                    "stage=%s event=retry_scheduled url=%s attempt=%d backoff_seconds=%.1f",
                    source,
                    url,
                    attempt,
                    backoff,
                )
                time.sleep(backoff)

        logger.error(
            "stage=%s event=all_attempts_failed url=%s max_retries=%d",
            source,
            url,
            self.settings.max_retries,
        )
        raise last_exception or APIError(f"Unknown failure calling {url}", source=source)

    def _parse_aqicn_response(self, raw: dict[str, Any], source: str) -> dict[str, Any]:
        """Validate and normalize a raw AQICN payload into the standard schema.

        Shared by both the city-name and lat/lon AQICN query stages, since
        both return an identical response shape.

        Args:
            raw: The raw JSON body returned by AQICN.
            source: Which AQICN stage produced this payload
                (`SOURCE_AQICN_CITY` or `SOURCE_AQICN_GEO`), used for
                labeling and error attribution.

        Returns:
            Standardized air-quality dictionary.

        Raises:
            APIError: If AQICN reports a non-'ok' status, or the response is
                missing expected fields.
        """
        if raw.get("status") != "ok":
            raise APIError(
                f"AQICN returned non-ok status: {raw.get('status')} - {raw.get('data')}",
                source=source,
            )

        try:
            data = raw["data"]
            iaqi = data.get("iaqi", {})
            city_info = data.get("city", {})
            geo = city_info.get("geo", [self.settings.latitude, self.settings.longitude])

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "city": city_info.get("name", self.settings.default_city),
                "latitude": geo[0],
                "longitude": geo[1],
                "aqi": data["aqi"],
                "pm2_5": iaqi.get("pm25", {}).get("v"),
                "pm10": iaqi.get("pm10", {}).get("v"),
                "co": iaqi.get("co", {}).get("v"),
                "so2": iaqi.get("so2", {}).get("v"),
                "no2": iaqi.get("no2", {}).get("v"),
                "o3": iaqi.get("o3", {}).get("v"),
                "source": source,
            }
        except KeyError as exc:
            raise APIError(
                f"Unexpected AQICN response structure, missing key: {exc}",
                source=source,
            ) from exc

    def _fetch_from_aqicn_by_city(self) -> dict[str, Any]:
        """Stage 1: fetch and normalize AQI data from AQICN, queried by city name.

        Returns:
            Standardized air-quality dictionary.

        Raises:
            APIError: If the request fails, AQICN reports a non-'ok' status
                (e.g. an unrecognized city/"Unknown station"), or the response
                is missing expected fields.
        """
        url = AQICN_CITY_URL_TEMPLATE.format(city=self.settings.default_city)
        params = {"token": self.settings.aqicn_token}
        raw = self._request_with_retry(url, params, source=SOURCE_AQICN_CITY)
        return self._parse_aqicn_response(raw, source=SOURCE_AQICN_CITY)

    def _fetch_from_aqicn_by_geo(self) -> dict[str, Any]:
        """Stage 2: fetch and normalize AQI data from AQICN, queried by the
        configured latitude/longitude. AQICN resolves this to the nearest
        monitoring station, which succeeds in many cases where the city-name
        query fails because the exact city string isn't registered.

        Returns:
            Standardized air-quality dictionary.

        Raises:
            APIError: If the request fails, AQICN reports a non-'ok' status,
                or the response is missing expected fields.
        """
        url = AQICN_GEO_URL_TEMPLATE.format(
            lat=self.settings.latitude, lon=self.settings.longitude
        )
        params = {"token": self.settings.aqicn_token}
        raw = self._request_with_retry(url, params, source=SOURCE_AQICN_GEO)
        return self._parse_aqicn_response(raw, source=SOURCE_AQICN_GEO)

    def _fetch_from_openweather_fallback(self) -> dict[str, Any]:
        """Stage 3: fetch and normalize AQI data from the OpenWeather Air
        Pollution API. Used only after both AQICN stages have failed.

        Returns:
            Standardized air-quality dictionary.

        Raises:
            APIError: If the request fails or the response is missing expected
                fields.
        """
        params = {
            "lat": self.settings.latitude,
            "lon": self.settings.longitude,
            "appid": self.settings.openweather_api_key,
        }
        raw = self._request_with_retry(
            OPENWEATHER_AIR_POLLUTION_URL, params, source=SOURCE_OPENWEATHER_FALLBACK
        )

        try:
            entry = raw["list"][0]
            components = entry["components"]
            coord = raw.get("coord", {"lat": self.settings.latitude, "lon": self.settings.longitude})

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "city": self.settings.default_city,
                "latitude": coord.get("lat", self.settings.latitude),
                "longitude": coord.get("lon", self.settings.longitude),
                "aqi": entry["main"]["aqi"],
                "pm2_5": components.get("pm2_5"),
                "pm10": components.get("pm10"),
                "co": components.get("co"),
                "so2": components.get("so2"),
                "no2": components.get("no2"),
                "o3": components.get("o3"),
                "source": SOURCE_OPENWEATHER_FALLBACK,
            }
        except (KeyError, IndexError) as exc:
            raise APIError(
                f"Unexpected OpenWeather Air Pollution response structure, missing key: {exc}",
                source=SOURCE_OPENWEATHER_FALLBACK,
            ) from exc

    def fetch_historical_aqi(
        self,
        latitude: float | None = None,
        longitude: float | None = None,
        timestamp: datetime | int | None = None,
    ) -> dict[str, Any]:
        """Fetch historical AQI data for a given location and time.

        Uses the **OpenAQ API v3** (requires ``OPENAQ_API_KEY``).  The AQICN
        stages (used by :meth:`fetch_air_quality` for current data) are
        **skipped** here because they ignore the requested timestamp and
        return current data instead -- which would silently fabricate
        history.  OpenWeather's Air Pollution History API is also not used
        because it requires a paid subscription.

        The v3 API uses a hierarchical lookup:
            1. Find monitoring locations near the coordinates.
            2. Collect sensors grouped by pollutant parameter.
            3. Fetch measurements for each target parameter, trying multiple
               nearby sensors and a widened time window if the closest
               sensor has no data exactly at the requested time.

        Since OpenAQ returns raw pollutant concentrations rather than a
        ready-made AQI, the AQI is computed using the official **US EPA AQI
        breakpoint formula**.  Concentrations in µg/m³ are converted to ppm
        / ppb when needed.

        Args:
            latitude:  Latitude (defaults to ``settings.latitude``).
            longitude: Longitude (defaults to ``settings.longitude``).
            timestamp: ``datetime`` (auto-converted) or Unix epoch ``int`` for
                the desired observation time.

        Returns:
            Standardized AQI dictionary (``source`` will be ``OpenAQ-v3``
            on success).

        Raises:
            APIError: If the OpenAQ API call fails, no measurements are
                found near the requested location, or the response is
                malformed.
        """
        lat = latitude if latitude is not None else self.settings.latitude
        lon = longitude if longitude is not None else self.settings.longitude

        if isinstance(timestamp, datetime):
            dt = timestamp
        elif timestamp is not None:
            dt = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
        else:
            dt = datetime.now(timezone.utc)

        logger.info(
            "stage=%s event=attempt_started lat=%s lon=%s timestamp=%s",
            SOURCE_OPENAQ_HISTORICAL,
            lat,
            lon,
            dt.isoformat(),
        )
        try:
            return self._fetch_openaq_v3_measurements(lat, lon, dt)
        except APIError as openaq_error:
            logger.warning(
                "stage=%s event=stage_failed next_stage=%s reason=%s",
                SOURCE_OPENAQ_HISTORICAL,
                SOURCE_OPENMETEO_HISTORICAL,
                openaq_error,
            )
            logger.info("stage=%s event=attempt_started", SOURCE_OPENMETEO_HISTORICAL)
            try:
                return self._fetch_openmeteo_air_quality(lat, lon, dt)
            except APIError as openmeteo_error:
                logger.error(
                    "stage=%s event=all_stages_exhausted errors=%s",
                    "AQI-historical-all-sources",
                    f"{SOURCE_OPENAQ_HISTORICAL}: {openaq_error} | "
                    f"{SOURCE_OPENMETEO_HISTORICAL}: {openmeteo_error}",
                )
                raise APIError(
                    f"All historical AQI sources failed. {SOURCE_OPENAQ_HISTORICAL}: "
                    f"{openaq_error} | {SOURCE_OPENMETEO_HISTORICAL}: {openmeteo_error}",
                    source="AQI-historical-all-sources",
                ) from openmeteo_error

    def _build_auth_headers(self) -> dict[str, str]:
        """Build the ``X-API-Key`` header required by OpenAQ v3."""
        return {"X-API-Key": self.settings.openaq_api_key}

    def _fetch_openmeteo_air_quality(
        self, lat: float, lon: float, dt: datetime
    ) -> dict[str, Any]:
        """Fetch historical AQI from Open-Meteo's Air Quality API (CAMS
        atmospheric-composition model), used as a fallback when OpenAQ v3
        has no ground-sensor data for this location/time.

        Unlike OpenAQ, this is a **model estimate**, not a ground
        measurement -- see the module docstring. Open-Meteo computes the US
        EPA AQI (``us_aqi``) directly, so it is used as-is here rather than
        recomputed via :meth:`_calculate_aqi` (which expects OpenAQ's raw,
        per-parameter concentration units).

        Args:
            lat: Latitude.
            lon: Longitude.
            dt:  Requested observation datetime (UTC).

        Returns:
            Standardized AQI dictionary, ``source="OpenMeteo-AirQuality"``.

        Raises:
            APIError: If the request fails or no matching hourly entry is found.
        """
        date_str = dt.strftime("%Y-%m-%d")
        params: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date_str,
            "end_date": date_str,
            "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
            "timezone": "UTC",
        }
        raw = self._request_with_retry(
            OPENMETEO_AIR_QUALITY_URL, params, source=SOURCE_OPENMETEO_HISTORICAL,
        )

        try:
            hourly = raw["hourly"]
            times: list[str] = hourly["time"]
            requested_iso = dt.strftime("%Y-%m-%dT%H:00")

            idx = times.index(requested_iso) if requested_iso in times else None
            if idx is None:
                raise APIError(
                    f"No matching hourly entry for {requested_iso} in Open-Meteo "
                    f"Air Quality response.",
                    source=SOURCE_OPENMETEO_HISTORICAL,
                )

            return {
                "timestamp": dt.isoformat(),
                "city": self.settings.default_city,
                "latitude": raw.get("latitude", lat),
                "longitude": raw.get("longitude", lon),
                "aqi": hourly["us_aqi"][idx],
                "pm2_5": hourly["pm2_5"][idx],
                # pm10/co/so2/no2/o3 are deliberately set to None here, even
                # though Open-Meteo *does* provide model-estimated values for
                # them. Sukkur's real OpenAQ ground sensor only measures
                # AQI + PM2.5, so for the OpenAQ-covered era (~Nov 2025+)
                # these columns are always null (see README Section 10).
                # Populating them only for this Open-Meteo fallback era would
                # make column-availability itself a spurious proxy for
                # calendar time/season -- a model could learn "pm10 present"
                # as a stand-in for "this is old data" rather than any real
                # physical relationship. Nulling them keeps the schema
                # consistent across the full historical range; `aqi` and
                # `pm2_5` remain the two universally-reliable pollutant
                # fields either way.
                "pm10": None,
                "co": None,
                "so2": None,
                "no2": None,
                "o3": None,
                "source": SOURCE_OPENMETEO_HISTORICAL,
            }
        except (KeyError, IndexError, ValueError) as exc:
            raise APIError(
                f"Unexpected Open-Meteo Air Quality response structure: {exc}",
                source=SOURCE_OPENMETEO_HISTORICAL,
            ) from exc

    def _fetch_openaq_v3_locations(
        self, lat: float, lon: float
    ) -> list[dict[str, Any]]:
        """Find nearby monitoring locations via OpenAQ v3.

        Args:
            lat: Latitude.
            lon: Longitude.

        Returns:
            List of location dicts, each containing a ``sensors`` array.
            The API returns results ordered by proximity, which we rely on
            when picking which sensor to try first for each pollutant.

        Raises:
            APIError: If the request fails or no locations are found.
        """
        params: dict[str, Any] = {
            "coordinates": f"{lat},{lon}",
            "radius": 25000,
            "limit": 100,
        }
        headers = self._build_auth_headers()
        raw = self._request_with_retry(
            OPENAQ_V3_LOCATIONS_URL, params, headers=headers,
            source=SOURCE_OPENAQ_HISTORICAL,
        )
        results = raw.get("results", [])
        if not results:
            raise APIError(
                "No OpenAQ v3 monitoring locations found near the given coordinates. "
                "This may mean there is no monitoring station within 50 km of "
                f"({lat}, {lon}).",
                source=SOURCE_OPENAQ_HISTORICAL,
            )
        return results

    def _fetch_openaq_v3_measurements(
        self, lat: float, lon: float, dt: datetime
    ) -> dict[str, Any]:
        """Query OpenAQ v3 for pollutant concentrations near *lat*/*lon*
        around time *dt*, then compute AQI from the concentrations.

        Workflow:
            1. Find nearby locations via ``/v3/locations``.
            2. Collect sensor IDs grouped by parameter from all locations,
               in proximity order (closest location first), keeping up to
               ``OPENAQ_MAX_SENSORS_PER_PARAM`` candidate sensors per
               pollutant.
            3. For each target parameter, query candidate sensors in order
               over a widened time window (``OPENAQ_MEASUREMENT_WINDOW_HOURS``
               each side of the requested timestamp) until one returns
               data. Among all readings returned for that sensor, keep the
               one closest in time to the requested timestamp.
            4. Parse and compute AQI from whatever pollutants were found.

        This wider search is important for sparsely-instrumented locations
        (smaller cities with fewer/less frequently reporting stations),
        where the single nearest sensor may have a data gap at the exact
        requested hour even though it -- or a nearby sensor -- has data
        close by in time.

        Args:
            lat: Latitude.
            lon: Longitude.
            dt:  Requested observation datetime.

        Returns:
            Standardized AQI dictionary.

        Raises:
            APIError: If the request fails or no data is returned.
        """
        locations = self._fetch_openaq_v3_locations(lat, lon)

        # Collect candidate sensor IDs grouped by parameter name, in
        # proximity order, capped at OPENAQ_MAX_SENSORS_PER_PARAM per
        # pollutant so we don't fire off unbounded requests.
        target_params = {"pm25", "pm10", "co", "so2", "no2", "o3"}
        sensor_ids: dict[str, list[int]] = {}
        for loc in locations:
            for sensor in loc.get("sensors", []):
                param = sensor.get("parameter", {})
                param_name = param.get("name", "")
                if param_name not in target_params:
                    continue
                candidates = sensor_ids.setdefault(param_name, [])
                if len(candidates) < OPENAQ_MAX_SENSORS_PER_PARAM and sensor["id"] not in candidates:
                    candidates.append(sensor["id"])

        if not sensor_ids:
            raise APIError(
                "No OpenAQ v3 sensors found for any target pollutant parameter "
                f"(pm25, pm10, co, so2, no2, o3) near ({lat}, {lon}).",
                source=SOURCE_OPENAQ_HISTORICAL,
            )

        window = timedelta(hours=OPENAQ_MEASUREMENT_WINDOW_HOURS)
        date_from = (dt - window).strftime("%Y-%m-%dT%H:%M:%SZ")
        date_to = (dt + window).strftime("%Y-%m-%dT%H:%M:%SZ")
        headers = self._build_auth_headers()

        # Fetch measurements for each target parameter, trying candidate
        # sensors in proximity order until one yields data.
        measurements_raw: dict[str, dict[str, Any]] = {}
        for param_name, candidate_sensor_ids in sensor_ids.items():
            for sensor_id in candidate_sensor_ids:
                sensor_url = OPENAQ_V3_SENSORS_MEASUREMENTS_TEMPLATE.format(
                    sensor_id=sensor_id,
                )
                sensor_params: dict[str, Any] = {
                    "datetime_from": date_from,
                    "datetime_to": date_to,
                    "limit": 100,
                }
                try:
                    raw = self._request_with_retry(
                        sensor_url, sensor_params, headers=headers,
                        source=SOURCE_OPENAQ_HISTORICAL,
                    )
                    results = raw.get("results", [])
                    best = self._closest_measurement(results, dt)
                    if best is not None:
                        param_info = best.get("parameter", {})
                        measurements_raw[param_name] = {
                            "value": best.get("value"),
                            "unit": param_info.get("units", ""),
                        }
                        break  # Found usable data for this pollutant.
                    logger.info(
                        "stage=%s event=sensor_no_data_in_window sensor_id=%s "
                        "param=%s window_hours=%d",
                        SOURCE_OPENAQ_HISTORICAL, sensor_id, param_name,
                        OPENAQ_MEASUREMENT_WINDOW_HOURS,
                    )
                except APIError:
                    logger.warning(
                        "stage=%s event=sensor_fetch_failed sensor_id=%s param=%s",
                        SOURCE_OPENAQ_HISTORICAL, sensor_id, param_name,
                    )
            # If no candidate sensor produced data, param_name is simply
            # absent from measurements_raw and AQI is computed from
            # whichever pollutants were found.

        if not measurements_raw:
            raise APIError(
                "No OpenAQ v3 measurements found for any target pollutant near "
                f"({lat}, {lon}) within {OPENAQ_MEASUREMENT_WINDOW_HOURS}h of "
                f"{dt.isoformat()}, after trying up to "
                f"{OPENAQ_MAX_SENSORS_PER_PARAM} sensor(s) per pollutant.",
                source=SOURCE_OPENAQ_HISTORICAL,
            )

        return self._parse_openaq_v3_response(measurements_raw, lat, lon, dt)

    @staticmethod
    def _closest_measurement(
        results: list[dict[str, Any]], dt: datetime
    ) -> dict[str, Any] | None:
        """Pick the measurement result whose period is closest in time to
        the requested timestamp *dt*.

        Args:
            results: Raw ``results`` list from an OpenAQ v3 measurements
                response.
            dt: The requested observation datetime.

        Returns:
            The closest-in-time result dict, or ``None`` if *results* is
            empty or none have a parsable timestamp.
        """
        best_result: dict[str, Any] | None = None
        best_diff: timedelta | None = None

        for r in results:
            period = r.get("period", {})
            datetime_from = period.get("datetimeFrom", {})
            iso_ts = None
            if isinstance(datetime_from, dict):
                iso_ts = datetime_from.get("utc")
            elif isinstance(datetime_from, str):
                iso_ts = datetime_from

            if not iso_ts:
                continue

            try:
                ts = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
            except ValueError:
                continue

            diff = abs(ts - dt)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_result = r

        # Fall back to the first result if none had a parsable timestamp,
        # so a single measurement in the window still gets used.
        if best_result is None and results:
            return results[0]
        return best_result

    def _parse_openaq_v3_response(
        self,
        measurements: dict[str, dict[str, Any]],
        lat: float,
        lon: float,
        dt: datetime,
    ) -> dict[str, Any]:
        """Parse OpenAQ v3 measurements into the standard AQI output schema.

        Args:
            measurements: Dict mapping parameter name -> ``{"value", "unit"}``.
            lat: Latitude used for the request.
            lon: Longitude used for the request.
            dt:  Requested observation datetime.

        Returns:
            Standardized AQI dictionary.

        Raises:
            APIError: If parsing fails.
        """
        try:
            def _get_value(param: str) -> float | None:
                entry = measurements.get(param)
                return entry["value"] if entry else None

            def _get_unit(param: str) -> str | None:
                entry = measurements.get(param)
                return entry["unit"] if entry else None

            pm25 = _get_value("pm25")
            pm10 = _get_value("pm10")
            co_raw = _get_value("co")
            so2_raw = _get_value("so2")
            no2_raw = _get_value("no2")
            o3_raw = _get_value("o3")

            # Convert CO (µg/m³ -> ppm) if needed.
            co_unit = _get_unit("co")
            if co_raw is not None and co_unit and "µg" in co_unit:
                co_ppm = co_raw / 1146.0
            else:
                co_ppm = co_raw

            # Convert gas pollutants (µg/m³ -> ppb) if needed.
            so2_unit = _get_unit("so2")
            if so2_raw is not None and so2_unit and "µg" in so2_unit:
                so2_ppb = so2_raw / 2.62
            else:
                so2_ppb = so2_raw

            no2_unit = _get_unit("no2")
            if no2_raw is not None and no2_unit and "µg" in no2_unit:
                no2_ppb = no2_raw / 1.88
            else:
                no2_ppb = no2_raw

            o3_unit = _get_unit("o3")
            if o3_raw is not None and o3_unit and "µg" in o3_unit:
                o3_ppb = o3_raw / 1.96
            else:
                o3_ppb = o3_raw

            aqi = self._calculate_aqi(pm25, pm10, co_ppm, so2_ppb, no2_ppb, o3_ppb)

            return {
                "timestamp": dt.isoformat(),
                "city": self.settings.default_city,
                "latitude": lat,
                "longitude": lon,
                "aqi": aqi,
                "pm2_5": pm25,
                "pm10": pm10,
                "co": co_raw,
                "so2": so2_raw,
                "no2": no2_raw,
                "o3": o3_raw,
                "source": SOURCE_OPENAQ_HISTORICAL,
            }
        except Exception as exc:
            raise APIError(
                f"Failed to parse OpenAQ v3 response: {exc}",
                source=SOURCE_OPENAQ_HISTORICAL,
            ) from exc

    @staticmethod
    def _concentration_to_aqi(
        concentration: float,
        breakpoints: list[tuple[tuple[float, float], tuple[int, int]]],
    ) -> int | None:
        """Convert a pollutant concentration to an AQI sub-index using the
        US EPA breakpoint formula:

            AQI = ((I_high - I_low) / (C_high - C_low)) * (C - C_low) + I_low

        Args:
            concentration: Pollutant concentration in the expected unit.
            breakpoints: List of ``((c_low, c_high), (i_low, i_high))``
                tuples defining the EPA breakpoint ranges.

        Returns:
            The AQI sub-index rounded to the nearest integer, or ``None``
            if the concentration is outside all defined breakpoints.
        """
        for (c_low, c_high), (i_low, i_high) in breakpoints:
            if c_low <= concentration <= c_high:
                if c_high != c_low:
                    return int(
                        round(
                            ((i_high - i_low) / (c_high - c_low))
                            * (concentration - c_low)
                            + i_low
                        )
                    )
                return int(round((i_high + i_low) / 2))
        return None

    @staticmethod
    def _calculate_aqi(
        pm25: float | None,
        pm10: float | None,
        co_ppm: float | None,
        so2_ppb: float | None,
        no2_ppb: float | None,
        o3_ppb: float | None,
    ) -> int | None:
        """Calculate the overall US EPA AQI from individual pollutant
        concentrations.

        Each pollutant's concentration is mapped through the EPA breakpoint
        tables to produce a sub-index.  The overall AQI is the **maximum**
        of all sub-indices.

        Full EPA AQI table reference:
        https://www.epa.gov/system/files/documents/2024-02/epa-aqi-technical-assistance-document.pdf

        Averaging periods and units:
            PM2.5 — 24-hour average, µg/m³
            PM10  — 24-hour average, µg/m³
            CO    — 8-hour average, ppm
            SO2   — 1-hour average, ppb
            NO2   — 1-hour average, ppb
            O3    — 8-hour average, ppb

        Returns:
            Overall AQI (0-500) or ``None`` if no pollutant data is
            available.
        """
        indexes: list[int] = []

        if pm25 is not None:
            bp_pm25 = [
                ((0.0, 12.0), (0, 50)),
                ((12.1, 35.4), (51, 100)),
                ((35.5, 55.4), (101, 150)),
                ((55.5, 150.4), (151, 200)),
                ((150.5, 250.4), (201, 300)),
                ((250.5, 500.4), (301, 500)),
            ]
            val = AirQualityFetcher._concentration_to_aqi(pm25, bp_pm25)
            if val is not None:
                indexes.append(val)

        if pm10 is not None:
            bp_pm10 = [
                ((0, 54), (0, 50)),
                ((55, 154), (51, 100)),
                ((155, 254), (101, 150)),
                ((255, 354), (151, 200)),
                ((355, 424), (201, 300)),
                ((425, 604), (301, 500)),
            ]
            val = AirQualityFetcher._concentration_to_aqi(pm10, bp_pm10)
            if val is not None:
                indexes.append(val)

        if co_ppm is not None:
            bp_co = [
                ((0.0, 4.4), (0, 50)),
                ((4.5, 9.4), (51, 100)),
                ((9.5, 12.4), (101, 150)),
                ((12.5, 15.4), (151, 200)),
                ((15.5, 30.4), (201, 300)),
                ((30.5, 50.4), (301, 500)),
            ]
            val = AirQualityFetcher._concentration_to_aqi(co_ppm, bp_co)
            if val is not None:
                indexes.append(val)

        if so2_ppb is not None:
            bp_so2 = [
                ((0, 35), (0, 50)),
                ((36, 75), (51, 100)),
                ((76, 185), (101, 150)),
                ((186, 304), (151, 200)),
                ((305, 604), (201, 300)),
                ((605, 1004), (301, 500)),
            ]
            val = AirQualityFetcher._concentration_to_aqi(so2_ppb, bp_so2)
            if val is not None:
                indexes.append(val)

        if no2_ppb is not None:
            bp_no2 = [
                ((0, 53), (0, 50)),
                ((54, 100), (51, 100)),
                ((101, 360), (101, 150)),
                ((361, 649), (151, 200)),
                ((650, 1249), (201, 300)),
                ((1250, 2049), (301, 500)),
            ]
            val = AirQualityFetcher._concentration_to_aqi(no2_ppb, bp_no2)
            if val is not None:
                indexes.append(val)

        if o3_ppb is not None:
            bp_o3 = [
                ((0, 54), (0, 50)),
                ((55, 70), (51, 100)),
                ((71, 85), (101, 150)),
                ((86, 105), (151, 200)),
                ((106, 200), (201, 300)),
            ]
            val = AirQualityFetcher._concentration_to_aqi(o3_ppb, bp_o3)
            if val is not None:
                indexes.append(val)

        return max(indexes) if indexes else None

    def fetch_air_quality(self) -> dict[str, Any]:
        """Fetch air-quality data using the three-stage fallback strategy:

            1. AQICN by city name.
            2. AQICN by latitude/longitude (if stage 1 fails or is invalid).
            3. OpenWeather Air Pollution API (if both AQICN stages fail).

        Each stage's failure is logged with structured, stage-tagged messages
        before falling through to the next stage, so the exact failure point
        is always traceable from `logs/pipeline.log`.

        Returns:
            Standardized air-quality dictionary (see `_parse_aqicn_response` /
            `_fetch_from_openweather_fallback` for the shared schema). The
            `source` field indicates which provider/stage actually served
            the data (`AQICN-city`, `AQICN-geo`, or `OpenWeather-AirPollution`).

        Raises:
            APIError: If all three stages fail.
        """
        errors: list[str] = []

        logger.info("stage=%s event=attempt_started", SOURCE_AQICN_CITY)
        try:
            return self._fetch_from_aqicn_by_city()
        except APIError as city_error:
            errors.append(f"{SOURCE_AQICN_CITY}: {city_error}")
            logger.warning(
                "stage=%s event=stage_failed next_stage=%s reason=%s",
                SOURCE_AQICN_CITY,
                SOURCE_AQICN_GEO,
                city_error,
            )

        logger.info("stage=%s event=attempt_started", SOURCE_AQICN_GEO)
        try:
            return self._fetch_from_aqicn_by_geo()
        except APIError as geo_error:
            errors.append(f"{SOURCE_AQICN_GEO}: {geo_error}")
            logger.warning(
                "stage=%s event=stage_failed next_stage=%s reason=%s",
                SOURCE_AQICN_GEO,
                SOURCE_OPENWEATHER_FALLBACK,
                geo_error,
            )

        logger.info("stage=%s event=attempt_started", SOURCE_OPENWEATHER_FALLBACK)
        try:
            return self._fetch_from_openweather_fallback()
        except APIError as fallback_error:
            errors.append(f"{SOURCE_OPENWEATHER_FALLBACK}: {fallback_error}")
            logger.error(
                "stage=%s event=all_stages_exhausted errors=%s",
                "AQI-all-sources",
                " | ".join(errors),
            )
            raise APIError(
                f"All AQI sources failed. {' | '.join(errors)}",
                source="AQI-all-sources",
            ) from fallback_error


def fetch_air_quality_data(settings: Settings) -> dict[str, Any]:
    """Convenience function: fetch AQI data using a fresh `AirQualityFetcher`.

    Args:
        settings: Loaded pipeline settings.

    Returns:
        Standardized air-quality dictionary.
    """
    fetcher = AirQualityFetcher(settings)
    return fetcher.fetch_air_quality()


def fetch_historical_aqi_data(
    settings: Settings,
    latitude: float | None = None,
    longitude: float | None = None,
    timestamp: int | datetime | None = None,
) -> dict[str, Any]:
    """Convenience function: fetch historical AQI data via OpenAQ v3.

    Args:
        settings:  Loaded pipeline settings.
        latitude:  Latitude (defaults to ``settings.latitude``).
        longitude: Longitude (defaults to ``settings.longitude``).
        timestamp: Unix epoch int, ``datetime``, or ``None`` (defaults to now).

    Returns:
        Standardized air-quality dictionary.
    """
    fetcher = AirQualityFetcher(settings)
    return fetcher.fetch_historical_aqi(latitude, longitude, timestamp)