# Changelog

All notable changes to CHEN will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Real vLLM KV-cache injection (v0.3.0 re-encodes source text; direct block injection is on the roadmap).
- Auto-tuner that learns the optimal router from observed KPIs.
- Reproduction configs for MMLU, HumanEval, GSM8K.
- Helm chart for Kubernetes deployment.
- Carbon-aware scheduling — route to lower-carbon experts based on real-time grid intensity (Electricity Maps API integration).
- Real-time carbon footprint dashboard (Scaphandre / Kepler integration).
- Batched inference — group similar-queue prompts for the same expert.
- Model versioning & A/B testing — run two Reasoner variants side-by-side.
- Audit logging — tamper-proof log of all prompts/outputs for compliance.
- Data encryption at rest — encrypt SQLite/Postgres run store contents.

## [0.3.0] — 2025-07-15

### Added — Industry-grade Tier 0 + Tier 1 production upgrade

**Tier 0 — The Missing Organs:**

**Real vLLM backend** (`src/chen/backends/vllm.py`):
- Full implementation of `encode()`, `decode()`, `transfer_cache()`.
- KV-cache extraction from vLLM's PagedAttention block manager — walks the sequence's block table, extracts each physical block's K/V tensors, concatenates to a flat numpy array per layer.
- Handles the non-contiguous block-table indirection (BlockSpaceManager).
- Layer-count truncation/padding for cross-instance transfer.
- Shape validation via `IncompatibleCacheError` for mismatched architectures.
- Lazy model loading, param count auto-detection from HF config.

**Real llama.cpp backend** (`src/chen/backends/llama_cpp.py`):
- Full implementation using `llama-cpp-python`'s `Llama._ctx` interface.
- KV-cache extraction via `llama_get_kv_cache` — de-interleaves the flat tensor into per-layer `[seq_len, n_head_kv, head_dim]` arrays.
- KV-cache injection via `llama_set_kv_cache` for same-model decode.
- Fallback to re-encoding source text for cross-model transfer.
- GGUF model loading, CPU/MPS inference, quantized model support.
- Config detection (`n_layer`, `n_embd`, `n_head_kv`, `head_dim`) from the loaded context.

**Server backend wiring** (`src/chen/server/routes.py`):
- `_build_backend()` factory supports `mock`, `hf`, `vllm`, `llama_cpp`.
- Per-expert model selection via env vars (`CHEN_HF_ANALYST_MODEL`, etc.).
- Lazy backend loading — models load on first request, not at server startup.
- All four backends now usable via the HTTP API and CLI.

**Tier 1 — Production Survival:**

**Authentication & authorization** (`src/chen/server/auth.py`):
- `APIKeyStore` — file-based API key store with JSON format.
- `AuthMiddleware` — validates `Authorization: Bearer <key>` headers; bypasses auth in dev mode (no keys file).
- Role-based access control: `admin` (all endpoints), `user` (no `/v1/admin/*`), `read-only` (GET only).
- `RateLimitMiddleware` — per-key sliding-window rate limiting (default 60 req/min, configurable per key).
- Returns HTTP 401 for missing/invalid keys, HTTP 403 for insufficient role, HTTP 429 for rate limit.
- Public paths (`/v1/health`, `/v1/metrics`, `/docs`) bypass auth.

**Input validation & prompt size limits** (`InferRequest`):
- `prompt` field capped at `max_length=100_000` characters (prevents OOM from 10MB prompts).
- `tenant_id` field added (max 128 chars) for multi-tenant isolation.
- `stream` field added (bool) for SSE streaming mode.
- All fields validated by Pydantic v2 with clear error messages.

**Configurable CORS** (`src/chen/server/app.py`):
- Removed `allow_origins=["*"]` hardcode.
- CORS origins now configurable via `CHEN_CORS_ORIGINS` env var (comma-separated).
- When not wildcard, `allow_credentials=True` and methods/headers are restricted.

**PostgreSQL run store** (`src/chen/persistence/pg_store.py`):
- `PostgresRunStore` — async, connection-pooled Postgres backend.
- Same interface as `RunStore` (save/get/list/count) but all methods are async.
- Uses `asyncpg` for non-blocking I/O.
- Schema with JSONB for `selected_experts`, indexes on timestamp/config_hash/phase.
- `get_run_store()` factory switches between SQLite and Postgres via `CHEN_RUN_STORE_BACKEND` env var.

**SSE streaming endpoint** (`POST /v1/infer/stream`):
- Server-Sent Events response with per-expert progress events.
- Final `event: done` with full metrics and run_id.
- Pipeline runs in a thread pool to avoid blocking the event loop.
- Headers: `Cache-Control: no-cache`, `Connection: keep-alive`, `X-Accel-Buffering: no`.

**Multi-tenant memory isolation** (`src/chen/persistence/tenant_memory.py`):
- `MultiTenantMemory` — wraps any Memory factory with per-tenant namespacing.
- Per-tenant entry quotas (`max_entries_per_tenant`).
- TTL-based garbage collection (`ttl_seconds`).
- `QuotaExceededError` when a tenant exceeds its quota.
- Tenants are completely isolated — no cross-tenant data leakage.

**Tier 2 — Production Confidence:**

**Circuit breaker** (`src/chen/observability/circuit_breaker.py`):
- `CircuitBreaker` — thread-safe, three-state (CLOSED → OPEN → HALF_OPEN).
- Configurable failure threshold and cooldown.
- `CircuitBreakerOpenError` raised when breaker is open.
- `CircuitBreakerRegistry` — global registry for all backend breakers.
- `to_dict()` for admin/metrics endpoints.
- `reset()` and `reset_all()` for manual recovery.

**OpenTelemetry tracing** (`src/chen/observability/tracing.py`):
- `init_tracing()` — sets up TracerProvider with console/OTLP/Jaeger exporters.
- `span()` context manager — works with or without OTel installed (no-op fallback).
- `init_from_env()` — configures from `CHEN_TRACING_*` env vars.
- Graceful degradation: if `opentelemetry` is not installed, all tracing is no-op.

**Health endpoint enhanced**:
- Now returns `backend`, `auth_enabled`, and `version` fields.

### Changed
- Version bumped to 0.3.0.
- `InferRequest` schema extended with `stream` and `tenant_id` fields.
- Server now loads auth middleware, rate limiter, and configurable CORS on startup.
- Test count grew from 260 to 308 (+48 tests across auth, circuit breaker, tenant memory, tracing, streaming, validation).

### Migration notes
- If you have existing API clients, they continue to work — auth is bypassed when no API keys file exists.
- To enable auth, create `chen_data/api_keys.json` with a list of `APIKey` objects.
- To switch to Postgres, set `CHEN_RUN_STORE_BACKEND=postgres` and `CHEN_RUN_STORE_DSN=...`.
- The `/v1/health` response now includes `backend` and `auth_enabled` fields — update any monitoring that expects the old schema.

### Added — Industry-grade deep tech upgrade

**Command-line interface (typer + rich):**
- `chen info` — print environment, backends, dependencies with rich tables.
- `chen run --prompt "..." --phase 1 --backend mock` — run a single prompt.
- `chen bench --phase 1` — run the benchmark suite with rich summary table.
- `chen serve --port 8000` — start the HTTP API server.
- `--save-run` flag persists runs to the SQLite run store.

**HTTP API server (FastAPI + uvicorn):**
- `POST /v1/infer` — run a prompt through a pipeline; returns output, per-expert metrics, KPIs, run_id.
- `GET /v1/health` — health check with version.
- `GET /v1/metrics` — Prometheus exposition format.
- `GET /v1/runs` — list recent runs (with optional phase filter).
- `GET /v1/runs/{run_id}` — fetch a specific run's full record.
- CORS middleware, request metrics middleware, automatic OpenAPI docs at `/docs`.
- Requires the `server` extra: `pip install -e ".[server]"`.

**Observability layer:**
- Structured logging via `structlog` (JSON in production, pretty console in dev).
- Prometheus metrics: request counters, latency histograms, expert invocations, KV transfers, tokens processed, pipeline runs.
- `configure_logging()` and `get_logger()` exported from the top-level `chen` package.
- Auto-detected log format (JSON if not a TTY, pretty if TTY).
- Configurable via `CHEN_LOG_LEVEL` and `CHEN_LOG_JSON` env vars.

**Persistence layer (SQLite run store):**
- `RunStore` class with `save()`, `get()`, `list()`, `count()`, `close()`.
- `RunRecord` dataclass with full run metadata (prompt, output, KPIs, config hash, timestamp).
- Default path `./chen_data/runs.sqlite3` (override via `CHEN_RUN_STORE_PATH`).
- Thread-safe via in-process lock; SQLite handles cross-process locking.
- Indexed by `run_id`, `timestamp`, `config_hash`, `phase` for fast queries.

**Reproducibility utilities:**
- `hash_config(dict)` — deterministic SHA-256 of a config dict (order-independent).
- `seed_everything(seed)` — seeds Python, NumPy, and PyTorch RNGs.
- `track_run(config, seed)` — context manager that seeds RNGs and logs run start/end.
- `RunContext` dataclass capturing config + hash + seed.

**Documentation — 7 Architecture Decision Records (ADRs):**
- ADR 0001: External MoE over internal MoE.
- ADR 0002: KV-cache transfer protocol with shape adaptation.
- ADR 0003: Pluggable backend abstraction.
- ADR 0004: Mock backend first — every test runs without a GPU.
- ADR 0005: Logistic router as the default.
- ADR 0006: SQLite as the run persistence layer.
- ADR 0007: CC0 1.0 — public domain dedication.
- ADR template (`0000-template.md`) for future records.

**Documentation — formal specs:**
- `docs/math/specifications.md` — formal definitions of EPU, cost model, latent nuance score, routing decision formula, with LaTeX math.
- `docs/architecture/overview.md` — system diagrams in Mermaid (topology, sequence flow, class diagram, router flowchart, deployment topology, reproducibility flow, KV-cache state machine).
- `docs/security/threat-model.md` — full threat model with trust boundaries (Mermaid), 6 enumerated threats with mitigations, deployment hardening checklist.
- `docs/governance.md` — roles (Contributors, Maintainers, BDFL), decision-making process, succession plan.
- `docs/operations/deployment.md` — Docker, Kubernetes, bare-metal systemd, nginx reverse proxy configs.
- `docs/operations/observability.md` — log events table, Prometheus metrics reference, example queries, alerting rules.
- `docs/operations/runbooks/incident-response.md` — severity levels, triage steps, 5 common incident playbooks, postmortem process.
- `docs/operations/runbooks/performance-debugging.md` — profiling steps, bottleneck table, backend selection guide.
- `docs/sustainability.md` — **NEW in v0.2.0**: full methodology for the sustainability claims in the README, including per-query energy calculation, carbon footprint, water usage, hardware reduction, annual impact at scale, methodology assumptions & sources, how to measure your own deployment, and comparison with other approaches (internal MoE, quantization, distillation).

**MkDocs Material documentation site:**
- `mkdocs.yml` configured with Material theme, dark/light mode, Mermaid support, MathJax, mkdocstrings API reference.
- `docs/assets/extra.css` for custom styles.
- Auto-deploys to GitHub Pages via `.github/workflows/docs.yml`.
- Strict build mode enforced in CI.

**Deployment — Docker:**
- Multi-stage `docker/Dockerfile` (builder + runtime) with non-root user, healthcheck, volume for run persistence.
- `docker/docker-compose.yml` with CHEN service + optional Prometheus monitoring profile.
- `docker/prometheus.yml` for scraping `/v1/metrics`.
- `docker/entrypoint.sh` for CLI pass-through.
- `.dockerignore` for fast builds.

**Code quality — pre-commit hooks:**
- `.pre-commit-config.yaml` with ruff (lint + format), mypy, general hygiene hooks (trailing whitespace, YAML/TOML/JSON validation, large file detection, private key detection), detect-secrets, bandit security scanner.
- Install with `pre-commit install`; runs automatically on every commit.

**Testing — property-based tests (Hypothesis):**
- `tests/property/test_kv_cache_props.py` — 8 property tests for KVCache invariants.
- `tests/property/test_router_props.py` — 7 property tests for router invariants (determinism, max_activation, force_last_role, name validity).

**Testing — performance benchmarks (pytest-benchmark):**
- `tests/benchmarks/test_pipeline_benchmarks.py` — 10 micro-benchmarks measuring orchestration overhead (router, pipeline phases, mock backend, memory retrieval, run store writes, config hashing).
- Run with `make test-benchmarks` or `pytest tests/benchmarks/ --benchmark-only`.

**Testing — integration tests:**
- `tests/integration/test_server.py` — 9 integration tests covering `/v1/health`, `/v1/metrics`, `/v1/infer` (phases 1/2/3, validation, run persistence), `/v1/runs` (list, 404 handling).
- Uses FastAPI's `TestClient`; gated behind `@pytest.mark.integration`.

**Repository hygiene:**
- `.github/CODEOWNERS` — auto-request review from maintainers for sensitive paths.
- `.dockerignore` — fast Docker builds.
- Updated CI workflow with integration tests, property tests, CLI smoke tests, Docker build + smoke test job.
- New `.github/workflows/release.yml` — auto-publish to PyPI + GHCR + GitHub Releases on tag.
- New `.github/workflows/docs.yml` — auto-deploy MkDocs site to GitHub Pages.
- New `.github/workflows/benchmarks.yml` — run pytest-benchmark on every push, compare against main on PRs.

**Configuration:**
- `pyproject.toml` — added `server` extra (fastapi, uvicorn, prometheus-client), `typer`, `rich`, `structlog`, `pyyaml` to core deps; added `hypothesis`, `pytest-benchmark`, `httpx`, `mkdocs`, `mkdocs-material`, `mkdocstrings` to dev; added `[project.scripts]` entry point `chen = "chen.cli.main:app"`.

**Makefile** — expanded from 15 to 40+ targets covering tests, lint, docs, CLI, Docker, pre-commit.

### Changed
- The `Pipeline` base class is now a dataclass (was a regular class) to support dataclass inheritance in subclasses.
- The `chen` package now exports `configure_logging`, `get_logger`, `RunRecord`, `RunStore`, `hash_config`, `seed_everything`, `track_run`, `RunContext` at the top level.
- CI now installs `[dev,server]` extras (was just `[dev]`).
- Test count grew from 141 to 213 (+72 tests across property, benchmark, integration, observability, persistence, reproducibility).

### Migration notes
- If you have existing code that subclassed `Pipeline` directly, the parent's `__init__` is now generated by `@dataclass`. Call `super().__post_init__()` in your subclass's `__post_init__`.
- The new top-level imports are additive — no breaking changes to existing APIs.


## [0.1.0] — 2025-01-15

### Added
- **Initial public release** of CHEN (Collaborative Heterogeneous Expert Network).
- **Pillar A: Latent State Router (LSR)** — `KVCache` dataclass with provenance
  metadata and a `transfer_cache` protocol that supports same-family no-op
  transfers and shape-mismatch re-encoding.
- **Pillar B: Specialized Micro-Experts** — `Expert` wrapper around a pluggable
  `InferenceBackend`, with `ExpertRole` enum (ANALYST, REASONER, SYNTHESIZER,
  CODER, ROUTER, TRANSLATOR, GENERALIST) for routing decisions.
- **Pillar C: Shared External Memory** — `Memory` protocol with an in-memory
  numpy backend (`InMemoryMemory`). Entries carry the role of the expert that
  wrote them, enabling latent-aware retrieval.
- **Phase 1: Static Cascade** — `CascadePipeline` chains experts in a fixed
  sequence, passing text between them. Tracks per-expert and aggregate metrics.
- **Phase 2: KV-Cache Passing** — `KVPassPipeline` passes KV-caches between
  experts instead of text. Falls back to text on transfer failure.
- **Phase 3: Dynamic Routing** — `RoutingPipeline` uses a `Router` to select
  which experts to wake per prompt. Three router variants:
  `LogisticRouter` (default, deterministic, no deps), `CosineRouter`
  (embedding similarity), `HybridRouter` (weighted product).
- **Backends**:
  - `MockBackend` — fully working, deterministic, CPU-only, no model downloads.
    Used by every test in the suite.
  - `HuggingFaceBackend` — real transformers backend with native KV-cache
    extraction. Works on CPU / CUDA / Apple MPS. Requires the `hf` extra.
  - `VLLMBackend` — stub (raises `NotImplementedError` with helpful message).
  - `LlamaCppBackend` — stub.
- **KPIs** — `KPIs` class computes Effective Parameter Utilization (EPU),
  cost-per-1M-tokens, and latency-to-accuracy ratio against a baseline monolith.
  Configurable targets with `met` verdicts.
- **Benchmark harness** — `BenchmarkRunner` runs a `BenchmarkTask` through a
  pipeline and produces a `KPIReport`. Five built-in tasks: math, code, QA,
  summarization, reasoning. Each has a deterministic grader.
- **Test suite** — 100+ unit and integration tests, all running on the
  MockBackend (no GPU / model downloads). Pytest with markers
  (`slow`, `gpu`, `integration`).
- **CI** — GitHub Actions workflow runs ruff, mypy, and pytest on Python 3.9,
  3.10, 3.11, 3.12 across Ubuntu, macOS, and Windows.
- **Documentation**:
  - Comprehensive `README.md` with quick start, install, and API overview.
  - `ARCHITECTURE.md` deep-dive covering the 3 pillars, data flow,
    KV-cache protocol, router design, and failure modes.
  - Three Jupyter notebooks walking through Phase 1, 2, 3 step by step.
- **Repo hygiene** — `LICENSE` (CC0 1.0), `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.cff`, `.gitignore`,
  `.env.example`, PR template, issue templates, Dependabot config.

### Documentation
- `pyproject.toml` with modern setuptools backend, `src/` layout, and optional
  extras: `hf`, `vllm`, `llama-cpp`, `memory`, `notebooks`, `dev`, `all`.
- `Makefile` with common dev commands (`make test`, `make lint`, `make format`,
  `make demo-phase1`, etc.).

[Unreleased]: https://github.com/your-org/chen/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-org/chen/releases/tag/v0.1.0
