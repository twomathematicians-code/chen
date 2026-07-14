# CHEN Architecture

This document is the canonical reference for how CHEN is structured, why each piece exists, and how data flows from a user prompt to a final answer. It is intended for contributors, researchers reproducing the experiments, and engineers considering CHEN for production deployment.

For a quick start, see the [README](README.md). For usage examples, see the [notebooks](notebooks/).

---

## Table of Contents

1. [Mental Model](#1-mental-model)
2. [The Three Pillars](#2-the-three-pillars)
   - [Pillar A: Latent State Router (LSR)](#pillar-a-latent-state-router-lsr)
   - [Pillar B: Specialized Micro-Experts](#pillar-b-specialized-micro-experts)
   - [Pillar C: Shared External Memory](#pillar-c-shared-external-memory)
3. [System Topology](#3-system-topology)
4. [Data Flow](#4-data-flow)
5. [KV-Cache Protocol](#5-kv-cache-protocol)
6. [Router Design](#6-router-design)
7. [Backend Abstraction](#7-backend-abstraction)
8. [Cost & Telemetry Model](#8-cost--telemetry-model)
9. [Experimental Phases](#9-experimental-phases)
10. [Failure Modes & Mitigations](#10-failure-modes--mitigations)
11. [Open Questions](#11-open-questions)

---

## 1. Mental Model

A traditional LLM call is a function:

```
output = monolith(prompt)
```

The monolith loads *all* of its parameters into VRAM for *every* call, even if the prompt is "what is 2+2." CHEN replaces this with:

```
output = pipeline(prompt, experts, router, memory)
```

where `pipeline` decides which subset of `experts` to wake up, what `memory` to inject, and whether to pass hidden states (KV-caches) or plain text between experts. The bet is that the *combined* cost of the awakened experts is much lower than the monolith, while the *combined* capacity (latent dimensions + context window) approaches or matches the monolith.

The architecture is explicitly **external MoE**: experts are independent processes (or even independent model binaries) that communicate over a defined protocol. This is in contrast to Mixtral or GShard, where experts are sub-networks inside a single forward pass.

---

## 2. The Three Pillars

### Pillar A: Latent State Router (LSR)

The LSR is the heart of CHEN. In Phase 1 it passes **text** between experts (lossy). In Phase 2 it passes **KV-caches** (lossless within a model family). In Phase 3 it adds **dynamic expert selection** — only wake the experts the current prompt actually needs.

A KV-cache is the intermediate tensor a transformer computes during its forward pass — specifically the cached keys and values at every attention layer. It is the model's *mathematical memory* of the prompt. Two facts make it powerful:

1. It is **dense**: a 4KB KV-cache can carry more nuance than 4KB of English text, because it is the model's own internal representation.
2. It is **transferable**: if model B has the same architecture and tokenizer as model A, B can often resume generation from A's KV-cache as if B had processed the prompt itself.

The Phase 2 experiment is the key differentiator of CHEN. We measure whether passing KV-caches between heterogeneous small models preserves more nuance than re-encoding text — and at what cost in compatibility.

**Protocol.** The LSR exposes three operations:

| Operation | Signature | Purpose |
|-----------|-----------|---------|
| `encode` | `(text) → KVCache` | Run the source model's prefill, return its KV-cache. |
| `transfer` | `(KVCache, target_model) → KVCache` | Adapt a KV-cache for a different model architecture. May be a no-op for same-family models, or a learned projection for cross-family. |
| `decode` | `(KVCache, max_tokens) → text` | Run the target model's decode loop starting from the transferred cache. |

The `transfer` step is where most research happens. For same-family transfers (e.g. Llama-3-3B → Llama-3-8B) it is often a no-op. For cross-family transfers (Llama → Mistral) we project via a learned linear map; the projection matrix is small (a few MB) and trained once per pair.

### Pillar B: Specialized Micro-Experts

An expert in CHEN is a `(name, role, backend)` triple. The `role` is a semantic tag (Analyst, Reasoner, Synthesizer, Coder, etc.) used by the router. The `backend` is a pluggable inference engine (see [Backend Abstraction](#7-backend-abstraction)).

| Role | Typical size | Trained for | Example model |
|------|--------------|-------------|---------------|
| `ANALYST` | 3B | Entity extraction, logic parsing | SmallCerebra-3B-Instruct |
| `REASONER` | 8B | Math, code, multi-step reasoning | Llama-3-8B-Instruct |
| `SYNTHESIZER` | 3B | Latent → natural language | SmallCerebra-3B-Instruct |
| `CODER` | 7B | Code generation, debugging | CodeLlama-7B-Instruct |
| `ROUTER` | 0.5B | Prompt classification | SmolLM-360M |

Crucially, experts are **independent**. You can replace the Reasoner with a different 8B model without retraining the Analyst or Synthesizer. You can add a new `TRANSLATOR` role without touching the router (the router falls back to a default expert when it sees a role it doesn't know).

### Pillar C: Shared External Memory

Small models lack the world knowledge baked into a 70B model's weights. CHEN compensates with a shared vector store — the network's "hippocampus." Before an expert generates, it issues a `memory.retrieve(query, k)` call; the memory returns the top-k relevant chunks, which are injected into the expert's prompt.

Two design choices distinguish CHEN's memory from vanilla RAG:

1. **Shared across experts.** All experts in the pipeline read and write the same memory. When the Analyst extracts entities, it writes them; when the Synthesizer generates, it reads them. This means later experts in the pipeline see structured outputs from earlier ones — not as text, but as memory entries.
2. **Latent-aware retrieval.** The memory stores both text and (optionally) the embedding of the expert that wrote the entry. Retrieval can filter by "written by the Reasoner" or "high-confidence," not just by topic.

The default backend is an in-memory `numpy`-based store (no dependencies, deterministic). For larger deployments, swap in ChromaDB via the `memory` extra.

---

## 3. System Topology

```
                              ┌────────────────────────┐
                              │   Shared External      │
                              │   Memory (RAG+)        │
                              │   • in-memory          │
                              │   • ChromaDB           │
                              └──────────┬─────────────┘
                                         │ retrieve / write
                                         ▼
┌──────────────┐    prompt    ┌────────────────────────┐    KV-cache     ┌──────────────┐
│              │ ───────────► │   Latent State Router  │ ──────────────► │   Expert A   │
│   User       │              │   (LSR)                │                 │   (Analyst)  │
│              │ ◄─────────── │   • encode             │ ◄────────────── │              │
└──────────────┘    output    │   • transfer           │    KV-cache     └──────────────┘
                              │   • decode             │
                              │   • route              │    KV-cache     ┌──────────────┐
                              │                        │ ──────────────► │   Expert B   │
                              │   LogisticRouter       │                 │   (Reasoner) │
                              │   CosineRouter         │ ◄────────────── │              │
                              │   HybridRouter         │    KV-cache     └──────────────┘
                              └──────────┬─────────────┘
                                         │    KV-cache     ┌──────────────┐
                                         │ ──────────────► │   Expert C   │
                                         │                 │ (Synthesizer)│
                                         │ ◄────────────── │              │
                                         │    text output  └──────────────┘
                                         ▼
                              ┌────────────────────────┐
                              │   Telemetry / KPIs     │
                              │   • EPU                │
                              │   • cost / 1M tok      │
                              │   • latency            │
                              └────────────────────────┘
```

---

## 4. Data Flow

A single `pipeline.run(prompt)` call proceeds as follows:

1. **Routing** (Phase 3 only). The router scores the prompt against each expert's prototype embedding and returns an ordered list of experts to activate. For Phase 1 and Phase 2, the sequence is statically configured.

2. **Memory Retrieval.** Before expert 1 runs, the pipeline calls `memory.retrieve(prompt, k=4)` and prepends the retrieved chunks to the prompt. This is the network's "hippocampus lookup."

3. **Expert 1 — Encode.** Expert 1 runs its prefill on the augmented prompt. It produces:
   - A text output (Phase 1 only — used as the input to the next expert).
   - A KV-cache (Phase 2+ — the latent memory of the prompt).
   - Optionally, structured writes to memory (e.g. extracted entities).

4. **Handoff.** Phase 1 passes text. Phase 2 passes the KV-cache via the LSR's `transfer` operation (no-op for same-family, learned projection for cross-family).

5. **Expert 2 — Decode.** Expert 2 begins generation from the transferred KV-cache. It may also retrieve from memory mid-generation (advanced; not in v0.1).

6. **Repeat** for each expert in the sequence.

7. **Telemetry.** The pipeline records per-expert latency, token counts, parameter count invoked, and (Phase 2+) a latent nuance score comparing KV-pass vs. text-pass on a held-out probe.

8. **Return.** A `PipelineResult` containing the final output, per-expert metrics, and aggregate KPIs.

---

## 5. KV-Cache Protocol

The `KVCache` dataclass is the lingua franca between experts:

```python
@dataclass
class KVCache:
    # Per-layer tensors. keys[i] and values[i] are the K and V for layer i.
    keys: list[np.ndarray]       # each shape: [n_heads, seq_len, head_dim]
    values: list[np.ndarray]     # same shape as keys

    # Provenance — needed for transfer validation
    source_model: str            # e.g. "HuggingFaceTB/SmallCerebra-3B"
    source_layer_count: int
    source_hidden_size: int
    source_n_heads: int

    # The text that produced this cache (for fallback / debugging)
    source_text: str

    # Tokenizer state needed to continue generation
    last_token_id: int
    position: int                # absolute position of the last token
```

**Same-family transfer** (e.g. Llama-3-3B → Llama-3-8B): if `n_heads` and `head_dim` match, the cache is passed directly. If only `n_layers` differs, we truncate or pad. If `hidden_size` differs, we raise `IncompatibleCacheError` and fall back to text.

**Cross-family transfer** (e.g. Llama → Mistral): we apply a learned projection `W ∈ R^{d_target × d_source}` per layer. The projection is trained once per (source, target) pair on a small alignment corpus (10k prompts suffices in our experiments). The MockBackend ships with identity projections for testing.

**Validation.** Before decoding from a transferred cache, the target expert runs a quick sanity probe: it generates one token from the cache and from a fresh encode of the source text, and compares logits via KL divergence. If divergence exceeds a threshold (default 5.0 nats), the pipeline logs a warning and optionally falls back to text.

---

## 6. Router Design

CHEN ships with three router variants, all implementing the `Router` protocol:

```python
class Router(Protocol):
    def route(self, prompt: str, available_experts: list[Expert]) -> list[str]:
        """Return the ordered list of expert names to activate."""
```

### LogisticRouter

A per-expert logistic classifier on prompt features (bag-of-words + length + presence of code/math keywords). Weights are either learned from labeled (prompt, expert) pairs or set heuristically. This is the default in v0.1 — it is deterministic, fast (<1ms), and requires no model download.

### CosineRouter

Computes the cosine similarity between the prompt's embedding (from a small sentence-transformer) and each expert's prototype embedding (precomputed mean of the expert's training prompts). Top-k experts are activated. Requires the `memory` extra for the embedding model.

### HybridRouter

Combines both: the logistic router produces a prior, the cosine router produces a likelihood, and the final score is a weighted product. The weight is a tunable hyperparameter.

All routers support:

- **Min activation** — guarantee at least one expert is always activated.
- **Max activation** — cap the number of experts per query (default 3) to bound cost.
- **Role constraints** — e.g. "always end with a Synthesizer."

---

## 7. Backend Abstraction

Every expert talks to a backend that implements the `InferenceBackend` protocol:

```python
class InferenceBackend(Protocol):
    @property
    def params_m(self) -> int: ...

    def encode(self, prompt: str) -> KVCache: ...
    def decode(self, cache: KVCache, max_tokens: int) -> str: ...
    def generate(self, prompt: str, max_tokens: int) -> str: ...
    def transfer_cache(self, cache: KVCache) -> KVCache: ...
```

The MockBackend implements all four with deterministic numpy operations (a tiny FFN that hashes the prompt to a fixed-dim embedding). This lets the entire pipeline — including Phase 2 KV-passing — run in under five seconds on a CPU, with no model downloads. Every test in the suite uses the MockBackend.

The HuggingFaceBackend wraps `transformers` and exposes real KV-caches via `model.forward(use_cache=True)`. It works on CPU (slow), CUDA (fast), and Apple MPS. This is the backend to use for actual Phase 2 experiments.

The vLLM and llama_cpp backends are stubs in v0.1 — they raise `NotImplementedError` with a helpful message. Implementing them is on the roadmap and is a great first contribution (see [CONTRIBUTING.md](CONTRIBUTING.md)).

---

## 8. Cost & Telemetry Model

Every `Expert.invoke()` call records:

- `params_m_invoked` — parameters loaded for this call (the expert's `params_m`).
- `input_tokens` — tokens in the prompt or transferred cache.
- `output_tokens` — tokens generated.
- `latency_ms` — wall-clock time.
- `cache_transfer_ms` — time spent in `transfer_cache` (Phase 2+).

The pipeline aggregates these into:

| KPI | Formula | Target |
|-----|---------|--------|
| **Cost per 1M tokens** | `(sum(params_m_invoked) × cost_per_1M_params) / total_tokens × 1e6` | 80–95% lower than monolith baseline |
| **EPU** | `effective_quality_b / sum(params_m_invoked)` where `effective_quality_b` is the parameter count of the smallest monolith that matches our quality | ≥ 3.0 (we use 3x our params and match a 3x-larger monolith) |
| **Latency-to-Accuracy** | `accuracy / latency_ms × 1000` | Higher than monolith on simple queries, competitive on hard ones |

Costs default to public list prices (see `config.CostModel`). Override via env vars if you have negotiated rates.

---

## 9. Experimental Phases

The three phases are intentionally incremental — each builds on the previous and isolates one variable.

### Phase 1: Static Cascading (text handoff)

- **Variable isolated:** Does chaining small models at all work?
- **Setup:** Hard-code the sequence `[Analyst, Reasoner, Synthesizer]`. Each expert gets the previous expert's *text* output.
- **Baseline:** Same prompt, same total token budget, one monolithic model.
- **Pass criterion:** Output quality ≥ baseline on at least 3 of 5 sample tasks.

### Phase 2: Latent State Passing (KV-cache handoff)

- **Variable isolated:** Does passing KV-caches preserve more nuance than text?
- **Setup:** Same sequence, but each expert receives the previous expert's KV-cache (via `transfer_cache`). No text is passed between experts.
- **Baseline:** Phase 1 (text handoff).
- **Pass criterion:** Latent nuance score (KL divergence between expert 2's logits under KV-pass vs. text-pass) is below threshold; benchmark accuracy improves by ≥5%.

### Phase 3: Dynamic Routing

- **Variable isolated:** Does waking only the needed experts cut cost without hurting quality?
- **Setup:** Router chooses which experts to wake per prompt. Pipeline runs the chosen subset.
- **Baseline:** Phase 2 (always-all-experts).
- **Pass criterion:** Average cost per query drops by ≥50% with no statistically significant accuracy regression on the benchmark suite.

---

## 10. Failure Modes & Mitigations

| Failure | Symptom | Mitigation |
|---------|---------|------------|
| Incompatible KV-cache (cross-family without projection) | `IncompatibleCacheError` | Fall back to text handoff; log warning. |
| Router picks zero experts | Empty activation list | Min-activation guarantee: pick the Synthesizer by default. |
| Router picks too many experts | Cost blowup | Max-activation cap (default 3). |
| Expert OOMs on long prompt | `torch.cuda.OutOfMemoryError` | Truncate prompt to expert's context window; log; continue. |
| Memory returns irrelevant chunks | Quality regression | Lower `k` for that expert; or fall back to no-memory mode. |
| MockBackend gives misleading numbers | Over-optimistic KPIs | Always also report numbers against a real HF baseline before publishing. |

---

## 11. Open Questions

These are intentionally left open in v0.1 and are good research directions:

1. **Optimal projection for cross-family KV transfer.** Is a per-layer linear map sufficient, or do we need attention-based projection? How small can the alignment corpus be?
2. **Mid-generation memory retrieval.** Should an expert be able to query memory *during* decode, not just before encode? What's the right granularity (per-token? per-sentence?)?
3. **Router training signal.** Should the router be trained on (prompt, expert, quality) triples from observed runs, or on a fixed supervised corpus? Online vs. offline?
4. **Streaming / async.** Can expert 2 start decoding from expert 1's partial KV-cache before expert 1 finishes? This would pipeline the experts and cut latency.
5. **EPU definition.** Is "smallest monolith that matches quality" the right denominator, or should we use a smoother scaling-law-based estimate?

If you have thoughts on any of these, please open a [Discussion](https://github.com/your-org/chen/discussions).
