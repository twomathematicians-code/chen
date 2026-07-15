"""Tests for OpenTelemetry tracing (no-op when OTel is not installed)."""

from __future__ import annotations

from chen.observability.tracing import (
    init_from_env,
    init_tracing,
    span,
    tracer,
)


class TestTracing:
    def test_tracer_returns_object(self):
        t = tracer()
        assert t is not None

    def test_span_context_manager_works(self):
        with span("test.span", key1="value1", key2=42) as s:
            assert s is not None

    def test_span_does_not_raise_without_otel(self):
        # Even if OTel is not installed, span() should work as a no-op
        with span("test.span"):
            pass

    def test_init_tracing_does_not_raise(self):
        # Should be idempotent and not raise even if OTel is missing
        init_tracing(service_name="test", exporter="console")
        init_tracing(service_name="test", exporter="console")

    def test_init_from_env_does_not_raise(self, monkeypatch):
        monkeypatch.setenv("CHEN_TRACING_ENABLED", "0")
        init_from_env()  # should be a no-op

    def test_init_from_env_with_enabled(self, monkeypatch):
        monkeypatch.setenv("CHEN_TRACING_ENABLED", "1")
        monkeypatch.setenv("CHEN_TRACING_EXPORTER", "console")
        init_from_env()  # should init or warn gracefully
