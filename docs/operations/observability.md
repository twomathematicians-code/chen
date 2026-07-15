# Observability

CHEN emits structured logs and Prometheus metrics. This document
describes what's available and how to consume it.

## 1. Logging

CHEN uses [structlog](https://www.structlog.org/) for structured
logging. Logs are emitted as JSON in production and as colored console
output in development (auto-detected from TTY).

### Configuration

| Env var | Default | Effect |
|---------|---------|--------|
| `CHEN_LOG_LEVEL` | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `CHEN_LOG_JSON` | (auto) | `1` = JSON logs; `0` = pretty console; auto = JSON if not a TTY |

### Log events

| Event | When | Key fields |
|-------|------|------------|
| `chen.run.start` | CLI `run` command begins | `prompt`, `phase`, `backend` |
| `chen.bench.start` | CLI `bench` command begins | `phase`, `router` |
| `chen.serve.start` | HTTP server starts | `version`, `backend` |
| `infer.complete` | `/v1/infer` request completes | `run_id`, `elapsed_ms`, `tokens` |
| `run.start` | Pipeline run begins (via `track_run`) | `config_hash`, `seed`, `config` |
| `run.end` | Pipeline run ends | `config_hash` |
| `run.failed` | Pipeline run raises | `config_hash`, `error` |

### Example JSON log line

```json
{"event": "infer.complete", "run_id": "a1b2c3d4e5f67890",
 "elapsed_ms": 12.3, "tokens": 145, "level": "info",
 "timestamp": "2025-01-15T10:30:00Z"}
```

### Consuming logs

- **Local dev**: pretty console output via structlog's ConsoleRenderer.
- **Docker**: JSON to stdout, captured by `docker logs chen`.
- **Kubernetes**: JSON to stdout, captured by `kubectl logs`.
- **Production**: ship to Loki, Elasticsearch, or CloudWatch via Fluent Bit / Filebeat.

## 2. Metrics

CHEN exposes Prometheus metrics at `GET /v1/metrics`.

### Counters

| Metric | Labels | Description |
|--------|--------|-------------|
| `chen_requests_total` | `method`, `route`, `status` | HTTP requests |
| `chen_expert_invocations_total` | `expert_name`, `role` | Expert invocations |
| `chen_kv_cache_transfers_total` | `source`, `target`, `result` | KV-cache transfers |
| `chen_tokens_processed_total` | `direction` (`input`/`output`) | Tokens processed |
| `chen_pipeline_runs_total` | `phase` | Pipeline runs by phase |

### Histograms

| Metric | Labels | Description |
|--------|--------|-------------|
| `chen_request_latency_seconds` | `route` | HTTP request latency |

### Gauges

| Metric | Description |
|--------|-------------|
| `chen_active_pipelines` | Currently running pipelines |

### Info

| Metric | Description |
|--------|-------------|
| `chen_build_info` | Build version (label: `version`) |

### Example Prometheus queries

```promql
# Requests per second by route
rate(chen_requests_total[1m])

# 99th percentile latency
histogram_quantile(0.99, rate(chen_request_latency_seconds_bucket[5m]))

# KV cache transfer success rate
rate(chen_kv_cache_transfers_total{result="success"}[5m])
  / rate(chen_kv_cache_transfers_total[5m])

# Tokens per second
rate(chen_tokens_processed_total[1m])

# Expert invocation distribution
sum by (expert_name) (rate(chen_expert_invocations_total[5m]))
```

### Alerting rules

```yaml
# Alert: KV transfer failure spike
- alert: ChenKVTransferFailures
  expr: rate(chen_kv_cache_transfers_total{result="failure"}[5m]) > 0.1
  for: 5m
  annotations:
    summary: "CHEN KV-cache transfer failures"
    description: "{{ $value }} failures/sec over the last 5 minutes"

# Alert: High error rate
- alert: ChenHighErrorRate
  expr: |
    sum(rate(chen_requests_total{status=~"5.."}[5m]))
      / sum(rate(chen_requests_total[5m])) > 0.05
  for: 2m
  annotations:
    summary: "CHEN API error rate above 5%"

# Alert: Server down
- alert: ChenDown
  expr: up{job="chen"} == 0
  for: 1m
  annotations:
    summary: "CHEN server is down"
```

## 3. Tracing

Distributed tracing (OpenTelemetry) is on the roadmap for v0.2. For
now, the `run_id` (config hash) in logs acts as a correlation ID —
search for a specific `run_id` to see all log lines related to one
pipeline run.

## 4. Run history

Every persisted run is queryable via:

- HTTP API: `GET /v1/runs` and `GET /v1/runs/{run_id}`
- SQLite: `sqlite3 chen_data/runs.sqlite3 "SELECT * FROM runs WHERE epu > 3;"`

This is the audit trail — useful for reproducibility, regression
detection, and cost analysis.
