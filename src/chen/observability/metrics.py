"""Prometheus metrics for the CHEN HTTP server.

Exposes counters and histograms for:
- requests (count, by route + status)
- request latency (histogram)
- expert invocations (counter, by expert name)
- KV-cache transfers (counter, success + failure)
- tokens processed (counter, in + out)
- pipeline runs (counter, by phase)
"""

from __future__ import annotations

try:
    from prometheus_client import Counter, Gauge, Histogram, Info

    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False


if _PROM_AVAILABLE:
    REQUESTS = Counter(
        "chen_requests_total",
        "Total HTTP requests",
        ["method", "route", "status"],
    )
    REQUEST_LATENCY = Histogram(
        "chen_request_latency_seconds",
        "HTTP request latency in seconds",
        ["route"],
    )
    EXPERT_INVOCATIONS = Counter(
        "chen_expert_invocations_total",
        "Total expert invocations",
        ["expert_name", "role"],
    )
    KV_TRANSFERS = Counter(
        "chen_kv_cache_transfers_total",
        "KV-cache transfers between experts",
        ["source", "target", "result"],  # result: "success" | "failure"
    )
    TOKENS_PROCESSED = Counter(
        "chen_tokens_processed_total",
        "Total tokens processed",
        ["direction"],  # direction: "input" | "output"
    )
    PIPELINE_RUNS = Counter(
        "chen_pipeline_runs_total",
        "Pipeline runs by phase",
        ["phase"],
    )
    ACTIVE_PIPELINES = Gauge(
        "chen_active_pipelines",
        "Currently running pipelines",
    )
    BUILD_INFO = Info(
        "chen_build",
        "CHEN build information",
    )

    def init_build_info(version: str) -> None:
        BUILD_INFO.info({"version": version})

    def record_expert_invocation(expert_name: str, role: str) -> None:
        EXPERT_INVOCATIONS.labels(expert_name=expert_name, role=role).inc()

    def record_kv_transfer(source: str, target: str, success: bool) -> None:
        KV_TRANSFERS.labels(
            source=source, target=target, result="success" if success else "failure"
        ).inc()

    def record_tokens(direction: str, count: int) -> None:
        TOKENS_PROCESSED.labels(direction=direction).inc(count)

    def record_pipeline_run(phase: int) -> None:
        PIPELINE_RUNS.labels(phase=str(phase)).inc()

    def record_request(method: str, route: str, status: int) -> None:
        REQUESTS.labels(method=method, route=route, status=str(status)).inc()

    def observe_request_latency(route: str, seconds: float) -> None:
        REQUEST_LATENCY.labels(route=route).observe(seconds)

else:  # _PROM_AVAILABLE is False — no-op stubs

    def init_build_info(version: str) -> None:
        pass

    def record_expert_invocation(expert_name: str, role: str) -> None:
        pass

    def record_kv_transfer(source: str, target: str, success: bool) -> None:
        pass

    def record_tokens(direction: str, count: int) -> None:
        pass

    def record_pipeline_run(phase: int) -> None:
        pass

    def record_request(method: str, route: str, status: int) -> None:
        pass

    def observe_request_latency(route: str, seconds: float) -> None:
        pass


def metrics_text() -> str:
    """Return Prometheus exposition text. Empty string if prometheus_client is missing."""
    if not _PROM_AVAILABLE:
        return ""
    from prometheus_client import generate_latest

    return generate_latest().decode("utf-8")
