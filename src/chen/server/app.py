"""FastAPI application factory.

Usage::

    from chen.server.app import create_app
    app = create_app()

Or via uvicorn::

    uvicorn chen.server.app:create_app --factory --port 8000

Or via the CLI::

    chen serve --port 8000
"""

from __future__ import annotations

import os
import time
from typing import Any

from chen import __version__
from chen.observability.logging import configure_logging, get_logger
from chen.observability.metrics import (
    init_build_info,
    metrics_text,
    observe_request_latency,
    record_request,
)
from chen.persistence.run_store import RunStore
from chen.server.routes import router

configure_logging(level=os.environ.get("CHEN_LOG_LEVEL", "INFO"))
init_build_info(__version__)

_log = get_logger("chen.server")


def create_app() -> Any:
    """Create and configure the FastAPI application."""
    try:
        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import PlainTextResponse
    except ImportError as e:
        raise ImportError(
            "FastAPI backend requires the 'server' extra. Install with: pip install -e '.[server]'"
        ) from e

    backend_default = os.environ.get("CHEN_DEFAULT_BACKEND", "mock")

    app = FastAPI(
        title="CHEN — Collaborative Heterogeneous Expert Network",
        description=(
            "Distributed inference architecture that replaces a single monolithic "
            "model with a coordinated garage of specialized, low-parameter models. "
            "Routes tokens through a dynamic pipeline and shares latent memory "
            "states (KV-caches) between models."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Wire up state
    app.state.backend_default = backend_default
    app.state.run_store = RunStore.default()

    # Routes
    app.include_router(router, prefix="/v1")

    @app.middleware("http")
    async def _metrics_middleware(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        route = request.url.path
        record_request(request.method, route, response.status_code)
        observe_request_latency(route, elapsed)
        return response

    @app.get("/v1/health", tags=["meta"])
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @app.get("/v1/metrics", tags=["meta"], response_class=PlainTextResponse)
    async def prometheus_metrics() -> str:
        return metrics_text()

    @app.on_event("startup")
    async def _on_startup() -> None:
        _log.info("server.start", version=__version__, backend=backend_default)

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:
        _log.info("server.stop")

    return app
