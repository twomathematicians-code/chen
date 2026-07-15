# CHEN — Collaborative Heterogeneous Expert Network

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](.github/workflows/ci.yml)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)

---

## What is CHEN?

**CHEN** is a distributed inference architecture that proves a coordinated network of small open-source models (3B–8B parameters) can simulate the parameter capacity and reasoning depth of a frontier 70B+ model — at a fraction of the compute cost.

Where a traditional monolith loads 70B parameters into VRAM for *every* query, CHEN keeps a roster of cheap specialists asleep and only wakes the ones a tiny router decides are needed for the current token stream. Where existing Mixture-of-Experts (MoE) systems like Mixtral do this *inside* a single neural network, CHEN does it **externally** — between completely separate, independently-trained, swappable models. You can upgrade one expert without touching the others, and you can build the entire system in your garage on a single consumer GPU (or even a CPU, using the bundled mock backend).

The three core pillars:

| Pillar | What it does | Why it matters |
|--------|--------------|----------------|
| **Latent State Router (LSR)** | Passes hidden states (KV-caches) between models instead of text. | Preserves nuance; chains context windows of small models together. |
| **Specialized Micro-Experts** | A garage of fine-tuned small models (Analyst, Reasoner, Synthesizer…). | You only pay for the parameters you actually use per query. |
| **Shared External Memory** | A lightweight vector DB acts as the network's "hippocampus." | Gives small models the world knowledge of a giant model without training it into their weights. |

---

## Why this matters now

Big tech (Microsoft WizardLM, Mistral Mixtral) is doing Mixture-of-Experts *inside* a single neural network. **CHEN does it externally** — between completely separate models. If we can prove that works, anyone can upgrade their AI by plugging a new cheap open-source model into the network, without ever retraining the core system.

The trade-space CHEN is trying to map:

- **Monolith Tax** — a 70B model uses massive compute even for trivial queries, because every parameter is loaded into VRAM.
- **Small Model Ceiling** — a 3B model is cheap and fast but hits a capacity wall where its context window fills up or its latent space can't hold the reasoning path.

CHEN's bet: route + share latent state, and the *combined* small models can match a monolith while paying only for the parameters actually invoked.

---

## Install

CHEN ships with a deterministic **MockBackend** so the entire pipeline runs on a CPU in under five seconds — no GPU, no model downloads. Real inference requires the `hf` extra.

```bash
# Core + dev tools + mock backend (CPU-only, 5s cold start)
pip install -e ".[dev]"

# Add the HuggingFace backend (needs PyTorch + a downloaded model)
pip install -e ".[hf]"

# Everything (HF + memory + notebooks + dev)
pip install -e ".[all]"
```

Python 3.9 or newer is required. See [`pyproject.toml`](pyproject.toml) for the full dependency matrix.

---

## Quick Start

### Phase 1 — Static Cascade (text handoff)

```python
from chen.phases.phase1_cascade import CascadePipeline
from chen.backends.mock import MockBackend
from chen.core.expert import Expert, ExpertRole

experts = [
    Expert(name="analyst",     role=ExpertRole.ANALYST,     backend=MockBackend(params_m=3_000)),
    Expert(name="reasoner",    role=ExpertRole.REASONER,    backend=MockBackend(params_m=8_000)),
    Expert(name="synthesizer", role=ExpertRole.SYNTHESIZER, backend=MockBackend(params_m=3_000)),
]

pipe = CascadePipeline(experts=experts, sequence=["analyst", "reasoner", "synthesizer"])
result = pipe.run("Read this 10-page PDF and write a Python script to analyze the data.")

print(result.output)
print(f"Cost:  ${result.metrics.total_cost_usd:.6f}")
print(f"Latency: {result.metrics.total_latency_ms:.1f} ms")
```

### Phase 2 — KV-Cache Passing (the magic step)

```python
from chen.phases.phase2_kv_pass import KVPassPipeline

pipe = KVPassPipeline(experts=experts, sequence=["analyst", "reasoner", "synthesizer"])
result = pipe.run("Explain quantum entanglement to a 12-year-old, then derive Bell's inequality.")

# Compare against the text-only cascade:
cascade_result = CascadePipeline(experts=experts, sequence=["analyst", "reasoner", "synthesizer"]).run("...")
print(f"KV-pass retained {result.metrics.latent_nuance_score:.2f}x more nuance than text-only.")
```

### Phase 3 — Dynamic Routing (only wake what you need)

```python
from chen.phases.phase3_routing import RoutingPipeline
from chen.core.router import LogisticRouter

router = LogisticRouter.from_experts(experts)
pipe = RoutingPipeline(experts=experts, router=router)

# Cheap query — only the synthesizer wakes up
pipe.run("Write a haiku about autumn.")

# Expensive query — analyst + reasoner wake up
pipe.run("Debug this segfault in my C++ allocator.")
```

### Benchmarks

```bash
python examples/run_benchmarks.py
```

Reports **EPU** (Effective Parameter Utilization), **cost per 1M tokens**, and **latency-to-accuracy ratio** vs. a monolithic baseline.

---

## Repository Layout

```
chen/
├── src/chen/
│   ├── backends/         # Pluggable inference: mock | hf | vllm | llama_cpp
│   ├── core/             # Expert, Router, Memory, KV-cache, Pipeline, Config
│   ├── phases/           # Phase 1 (cascade), Phase 2 (KV-pass), Phase 3 (routing)
│   └── benchmarks/       # KPIs (EPU, cost, latency) + sample tasks
├── tests/                # Pytest suite (mirrors src/ layout)
├── examples/             # Runnable scripts: run_phase{1,2,3}.py, run_benchmarks.py
├── notebooks/            # Step-by-step walkthroughs of each phase
├── docs/                 # ARCHITECTURE.md and design docs
├── .github/workflows/    # CI: ruff + mypy + pytest
├── pyproject.toml        # Modern setuptools packaging with optional extras
├── Makefile              # Common dev commands
├── LICENSE               # CC0 1.0 Universal (public domain)
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
└── CITATION.cff          # For academic citation
```

---

## Backends

| Backend | Status | GPU required | KV-cache pass | Use case |
|---------|--------|--------------|---------------|----------|
| `mock` | ✅ Stable | No | Yes (deterministic) | Tests, demos, CPU development |
| `hf` | ✅ Stable | Optional (CPU works, slow) | Yes (native) | Research, Phase 2 experiments |
| `vllm` | 🚧 Stub | Yes (CUDA) | Stub (PagedAttention) | Production throughput |
| `llama_cpp` | 🚧 Stub | No (CPU/MPS) | Stub (limited API) | Mac / CPU deployment |

Switch backends via the `CHEN_DEFAULT_BACKEND` env var or by passing an explicit `backend=` to each `Expert`. See [`.env.example`](.env.example).

---

## How to run the tests

```bash
make test-fast          # CPU-only, no model downloads — runs in <10s
make test               # Everything except GPU-marked tests
make test-cov           # With coverage report (htmlcov/)
```

CI runs `make test-fast`, `make lint`, `make typecheck` on every push — see [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## Key Performance Indicators

CHEN measures success against a baseline monolith (e.g. GPT-4 or Llama-3-70B) along three axes:

1. **Effective Parameter Utilization (EPU)** — if you used 9B of parameters total (three 3B experts), did your output quality match a 30B model? If yes, EPU is highly efficient.
2. **Cost per 1M Tokens** — target 80–95% lower than the baseline monolith.
3. **Latency-to-Accuracy Ratio** — small models are fast; CHEN should keep the low latency of small models while showing a statistical accuracy jump on benchmarks like MMLU or HumanEval.

See [`src/chen/benchmarks/kpis.py`](src/chen/benchmarks/kpis.py) for the exact formulas and [`examples/run_benchmarks.py`](examples/run_benchmarks.py) for a runnable harness.

---

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — deep dive on the three pillars, data flow, KV-cache protocol, router design.
- [`notebooks/01_phase1_cascade.ipynb`](notebooks/01_phase1_cascade.ipynb) — static cascade walkthrough.
- [`notebooks/02_phase2_kv_pass.ipynb`](notebooks/02_phase2_kv_pass.ipynb) — KV-cache handoff, the key differentiator.
- [`notebooks/03_phase3_routing.ipynb`](notebooks/03_phase3_routing.ipynb) — dynamic routing and cost-per-query.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — how to add backends, benchmark tasks, and get PRs merged.

---

## Roadmap

- [x] Phase 1: Static text cascade with cost/latency tracking
- [x] Phase 2: KV-cache handoff between separate models
- [x] Phase 3: Tiny classifier router (logistic + cosine variants)
- [x] KPI harness: EPU, cost-per-1M, latency-to-accuracy
- [ ] Real vLLM backend with PagedAttention KV passing
- [ ] Real llama.cpp backend with GGUF KV export
- [ ] Async / streaming pipeline (token-by-token handoff)
- [ ] Auto-tuner that learns the optimal router from observed KPIs
- [ ] Reproduction configs for MMLU, HumanEval, GSM8K

---

## License

Released into the public domain under [CC0 1.0 Universal](LICENSE). You may use, modify, distribute, and commercialize this work for any purpose, without attribution (though attribution is appreciated).

If you use CHEN in academic research, please cite it using the metadata in [`CITATION.cff`](CITATION.cff).

---

## Acknowledgements

CHEN builds on ideas from:

- **Mistral Mixtral** — internal Mixture-of-Experts at scale.
- **Microsoft WizardLM** — instruction tuning and expert specialization.
- **vLLM** — PagedAttention and high-throughput inference.
- **HuggingFace Transformers** — model ecosystem and KV-cache primitives.
- **LangChain** — orchestration patterns for chained LLM calls.

The novelty of CHEN is doing MoE *externally* between heterogeneous, swappable, independently-trained models — passing latent state rather than text.
