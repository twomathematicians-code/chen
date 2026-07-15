"""CHEN HTTP API server.

Provides a FastAPI application exposing CHEN pipelines over HTTP:

* ``POST /v1/infer``   — run a prompt through a pipeline
* ``GET  /v1/health``  — health check
* ``GET  /v1/metrics`` — Prometheus metrics
* ``GET  /v1/runs``    — list recent runs (from SQLite store)
* ``GET  /v1/runs/{id}`` — fetch a specific run

Requires the ``server`` extra: ``pip install -e '.[server]'``
"""

from __future__ import annotations

from chen.server.app import create_app

__all__ = ["create_app"]
