"""
Custom exception hierarchy for the AQI Prediction data ingestion pipeline.

Using a dedicated exception hierarchy (instead of raising bare `Exception` or
generic built-ins) allows callers to catch and handle failure modes precisely,
and makes the pipeline's failure surface explicit and self-documenting.
"""

from __future__ import annotations


class AQIPipelineError(Exception):
    """Base class for all custom exceptions raised by this project.

    Catching this exception type will catch any error that originates from
    within the pipeline's own logic (as opposed to third-party library
    errors that were not wrapped).
    """


class ConfigurationError(AQIPipelineError):
    """Raised when required configuration/environment variables are missing,
    empty, or otherwise invalid (e.g. a malformed latitude/longitude).
    """


class APIError(AQIPipelineError):
    """Raised when an external API call fails.

    This includes network failures, timeouts, non-2xx HTTP responses, and
    malformed/unexpected response payloads that cannot be parsed.
    """

    def __init__(self, message: str, *, source: str | None = None, status_code: int | None = None):
        self.source = source
        self.status_code = status_code
        full_message = message
        if source:
            full_message = f"[{source}] {full_message}"
        if status_code is not None:
            full_message = f"{full_message} (status_code={status_code})"
        super().__init__(full_message)


class ValidationError(AQIPipelineError):
    """Raised when a dataset or record fails validation rules
    (missing values, wrong datatypes, out-of-range values, etc.).
    """


class StorageError(AQIPipelineError):
    """Raised when reading from or writing to disk fails
    (e.g. permission errors, malformed JSON/CSV, disk I/O issues).
    """


class PipelineError(AQIPipelineError):
    """Raised when an orchestration-level pipeline operation fails
    (e.g. the feature pipeline itself, historical backfill, or live
    pipeline encounters an unrecoverable error).
    """

class FeatureStoreError(PipelineError):
    """Raised when a Hopsworks Feature Store operation fails: login,
    feature group creation/retrieval, feature insertion, or reading
    features back.
    """