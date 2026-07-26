"""
Centralized logging configuration for the AQI Prediction data ingestion pipeline.

All modules in the project should obtain their logger via `get_logger(__name__)`
rather than instantiating `logging.getLogger` directly, so that log formatting,
log level, and log destinations (console + file) stay consistent everywhere.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# Project root is two levels up from this file: src/common/logger.py -> project root
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LOGS_DIR = os.path.join(_PROJECT_ROOT, "logs")
_LOG_FILE = os.path.join(_LOGS_DIR, "pipeline.log")

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root_logger() -> None:
    """Configure the root logger once with a console handler and a rotating
    file handler. Subsequent calls are no-ops.
    """
    global _configured
    if _configured:
        return

    os.makedirs(_LOGS_DIR, exist_ok=True)

    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    # Rotating file handler: keeps log files bounded in size in a long-running
    # pipeline (5 MB per file, 5 backups retained).
    file_handler = RotatingFileHandler(
        _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured to write to both console and
    `logs/pipeline.log`.

    Args:
        name: Typically `__name__` of the calling module.

    Returns:
        A configured `logging.Logger` instance.
    """
    _configure_root_logger()
    return logging.getLogger(name)
