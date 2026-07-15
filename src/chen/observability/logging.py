"""Structured logging via structlog.

Configures structlog to emit JSON logs in production and pretty console
logs in development. The logger is process-wide — call
:func:`configure_logging` once at startup.

Usage::

    from chen.observability.logging import configure_logging, get_logger

    configure_logging(level="INFO")
    log = get_logger(__name__)
    log.info("event", prompt="hello", phase=1)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

try:
    import structlog

    _STRUCTLOG_AVAILABLE = True
except ImportError:
    _STRUCTLOG_AVAILABLE = False

_CONFIGURED = False


def configure_logging(
    level: str = "INFO",
    json_logs: bool | None = None,
) -> None:
    """Configure structured logging globally.

    Args:
        level: Log level ("DEBUG", "INFO", "WARNING", "ERROR").
        json_logs: If True, emit JSON logs; if False, pretty console logs.
            If None, auto-detect (JSON if CHEN_LOG_JSON=1 or not a TTY).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    if json_logs is None:
        json_logs = os.environ.get("CHEN_LOG_JSON", "") == "1" or not sys.stderr.isatty()

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    if _STRUCTLOG_AVAILABLE:
        processors: list[Any] = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
        ]
        if json_logs:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, level.upper(), logging.INFO)
            ),
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )

    _CONFIGURED = True


def get_logger(name: str | None = None) -> Any:
    """Return a structured logger. Falls back to stdlib logging if structlog is missing."""
    if _STRUCTLOG_AVAILABLE:
        if not _CONFIGURED:
            configure_logging()
        return structlog.get_logger(name)
    return logging.getLogger(name)
