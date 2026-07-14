# Changelog

All notable changes to CHEN will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Real vLLM backend with PagedAttention KV extraction.
- Real llama.cpp backend with GGUF KV export.
- Async / streaming pipeline (token-by-token handoff).
- Auto-tuner that learns the optimal router from observed KPIs.
- Reproduction configs for MMLU, HumanEval, GSM8K.

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
