"""OpenTelemetry distributed tracing for CHEN pipelines.

Provides span instrumentation for:
- Each pipeline run (root span)
- Each expert invocation (child span)
- Each KV-cache transfer (child span)
- Each router decision (child span)

If OpenTelemetry is not installed, all functions are no-ops — the
codebase works without tracing, but production deployments should
install OTel and configure an exporter (Jaeger, Zipkin, Tempo).

Usage::

    from chen.observability.tracing import tracer, span

    with tracer().start_as_current_span("pipeline.run") as s:
        s.set_attribute("phase", 1)
        s.set_attribute("prompt_length", len(prompt))
        result = pipeline.run(prompt)

Install::

    pip install opentelemetry-api opentelemetry-sdk
    pip install opentelemetry-exporter-jaeger  # or otlp, zipkin
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Optional

from chen.observability.logging import get_logger

_log = get_logger("chen.observability.tracing")

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    trace = None  # type: ignore

_OTEL_INITIALIZED = False


def init_tracing(
    service_name: str = "chen",
    exporter: str = "console",
    endpoint: Optional[str] = None,  # noqa: UP045
) -> None:
    """Initialize OpenTelemetry tracing.

    Args:
        service_name: Service name for traces.
        exporter: "console", "jaeger", "otlp", or "none".
        endpoint: Exporter endpoint (e.g. "http://localhost:4317").
    """
    global _OTEL_INITIALIZED
    if _OTEL_INITIALIZED or not _OTEL_AVAILABLE:
        return

    provider = TracerProvider(
        resource={"service.name": service_name},
    )

    if exporter == "console":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif exporter == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint or "http://localhost:4317"))
            )
        except ImportError:
            _log.warning("tracing.otlp_not_installed")
    elif exporter == "jaeger":
        try:
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter

            provider.add_span_processor(
                BatchSpanProcessor(JaegerExporter(endpoint=endpoint or "http://localhost:14268"))
            )
        except ImportError:
            _log.warning("tracing.jaeger_not_installed")

    trace.set_tracer_provider(provider)
    _OTEL_INITIALIZED = True
    _log.info("tracing.initialized", service=service_name, exporter=exporter)


def tracer() -> Any:
    """Return the OTel tracer (or a no-op if OTel is not installed)."""
    if _OTEL_AVAILABLE and _OTEL_INITIALIZED:
        return trace.get_tracer("chen")
    return _NoOpTracer()


class _NoOpSpan:
    """No-op span for when OTel is not installed."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[dict] = None) -> None:  # noqa: UP045
        pass

    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpTracer:
    """No-op tracer for when OTel is not installed."""

    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Context manager for a traced span.

    Works with or without OTel installed. Attributes are set on the span.

    Usage::

        with span("expert.invoke", expert_name="reasoner", role="reasoner"):
            result = expert.invoke(prompt)
    """
    t = tracer()
    with t.start_as_current_span(name) as s:
        for k, v in attributes.items():
            s.set_attribute(k, v)
        yield s


def init_from_env() -> None:
    """Initialize tracing from environment variables.

    Reads:
        CHEN_TRACING_ENABLED: "1" to enable.
        CHEN_TRACING_EXPORTER: "console" | "otlp" | "jaeger"
        CHEN_TRACING_ENDPOINT: exporter endpoint URL
        CHEN_TRACING_SERVICE_NAME: service name (default "chen")
    """
    if os.environ.get("CHEN_TRACING_ENABLED", "") != "1":
        return
    if not _OTEL_AVAILABLE:
        _log.warning("tracing.requested_but_not_installed")
        return
    init_tracing(
        service_name=os.environ.get("CHEN_TRACING_SERVICE_NAME", "chen"),
        exporter=os.environ.get("CHEN_TRACING_EXPORTER", "console"),
        endpoint=os.environ.get("CHEN_TRACING_ENDPOINT"),
    )
