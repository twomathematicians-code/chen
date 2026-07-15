# Operations

This directory contains operational documentation for running CHEN in
production.

## Topics

- [Deployment](deployment.md) — how to deploy CHEN as a service
- [Observability](observability.md) — logging, metrics, tracing
- [Runbooks](runbooks/) — step-by-step guides for common operational tasks

## Quick reference

### Start the API server (Docker)

```bash
cd docker
docker compose up -d
# API is now at http://localhost:8000
# Docs at http://localhost:8000/docs
# Health at http://localhost:8000/v1/health
# Metrics at http://localhost:8000/v1/metrics
```

### Start the API server (local)

```bash
pip install -e ".[server]"
chen serve --host 0.0.0.0 --port 8000
```

### Run the CLI

```bash
chen info
chen run --prompt "Explain recursion." --phase 1 --backend mock
chen bench --phase 1
```

### Check run history

```bash
# Via CLI (when added)
chen runs list

# Direct SQLite query
sqlite3 chen_data/runs.sqlite3 "SELECT run_id, timestamp, phase, total_tokens, epu FROM runs ORDER BY timestamp DESC LIMIT 10;"
```
