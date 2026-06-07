"""
harness/logger.py — Structured Logging
=======================================
Uses loguru for structured JSON + human-readable logging.
All agents share this logger. Log level is controlled via config.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from typing import Optional

from loguru import logger as _loguru_logger


_configured = False


def configure_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure loguru globally. Call once at startup.
    Adds:
      - stderr sink with colored human-readable format
      - optional file sink with JSON serialization (for ingestion by ELK / Datadog)
    """
    global _configured
    _loguru_logger.remove()

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    )
    _loguru_logger.add(sys.stderr, level=log_level, format=fmt, colorize=True)

    if log_file:
        _loguru_logger.add(
            log_file,
            level=log_level,
            serialize=True,          # JSON lines
            rotation="100 MB",
            retention="14 days",
            compression="gz",
        )

    _configured = True


@lru_cache(maxsize=None)
def get_logger(name: str):
    """Return a bound logger tagged with the caller's module name."""
    return _loguru_logger.bind(name=name)


# Expose convenience aliases
def bind(**kwargs):
    return _loguru_logger.bind(**kwargs)
