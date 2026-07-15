"""Tests for the observability layer."""

from __future__ import annotations

from chen.observability.logging import configure_logging, get_logger
from chen.observability.metrics import (
    init_build_info,
    metrics_text,
    observe_request_latency,
    record_expert_invocation,
    record_kv_transfer,
    record_pipeline_run,
    record_request,
    record_tokens,
)


class TestLogging:
    def test_configure_logging_idempotent(self):
        # Multiple calls should not raise.
        configure_logging(level="INFO")
        configure_logging(level="DEBUG")
        configure_logging(level="INFO")

    def test_get_logger_returns_logger(self):
        log = get_logger("test")
        # Should be either a structlog logger or stdlib logger.
        assert hasattr(log, "info")
        assert hasattr(log, "debug")
        assert hasattr(log, "error")

    def test_logger_emits_without_error(self, capsys):
        configure_logging(level="DEBUG")
        log = get_logger("test")
        log.info("test_event", key="value")
        # No assertion on output — just verify it doesn't raise.


class TestMetrics:
    def test_init_build_info_does_not_raise(self):
        init_build_info("0.1.0")

    def test_record_expert_invocation(self):
        # Should not raise; if prometheus_client is missing, it's a no-op.
        record_expert_invocation("analyst", "analyst")

    def test_record_kv_transfer(self):
        record_kv_transfer("analyst", "reasoner", success=True)
        record_kv_transfer("reasoner", "synthesizer", success=False)

    def test_record_tokens(self):
        record_tokens("input", 100)
        record_tokens("output", 50)

    def test_record_pipeline_run(self):
        record_pipeline_run(1)
        record_pipeline_run(2)
        record_pipeline_run(3)

    def test_record_request(self):
        record_request("POST", "/v1/infer", 200)
        record_request("GET", "/v1/health", 200)

    def test_observe_request_latency(self):
        observe_request_latency("/v1/infer", 0.123)

    def test_metrics_text_returns_string(self):
        text = metrics_text()
        assert isinstance(text, str)
