"""Observability: structured logging and metrics.

CHEN uses ``structlog`` for structured JSON logging and exposes Prometheus
metrics via the ``chen.observability.metrics`` module. Both are optional
in the sense that the codebase works without them, but production
deployments should configure them.
"""

from __future__ import annotations

from chen.observability.logging import configure_logging, get_logger

__all__ = ["configure_logging", "get_logger"]
