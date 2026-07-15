# ADR 0003: Pluggable backend abstraction

- Status: Accepted
- Date: 2025-01-15
- Deciders: CHEN core team

## Context

CHEN needs to run on:

- A CI runner with no GPU and no model downloads (for tests).
- A developer laptop with CPU and PyTorch (for prototyping).
- A workstation with a CUDA GPU (for real experiments).
- A Mac with Apple Metal (for edge deployment).
- Eventually, a production cluster with vLLM or TGI.

Each of these has different dependencies, different APIs, and different
KV-cache formats. Hardcoding any one of them would make the codebase
untestable in the others.

## Decision

Define a minimal `InferenceBackend` protocol with four methods plus a
`params_m` property and a `capabilities` declaration:

```
generate(prompt, max_tokens) -> str
encode(prompt) -> KVCache
decode(cache, max_tokens) -> str
transfer_cache(cache) -> KVCache
```

Backends register themselves in a global `BACKEND_REGISTRY` keyed by
short name (`mock`, `hf`, `vllm`, `llama_cpp`).

The `capabilities` field declares what a backend can do
(`supports_kv_cache`, `supports_streaming`, `deterministic`). The
pipeline reads capabilities and falls back gracefully.

## Consequences

### Positive

- **Tests are GPU-free** — `MockBackend` implements the full protocol deterministically.
- **Adding a backend is local** — one new file, one registry call.
- **Capability-based fallback** — backends that can't do KV-cache still work in Phase 1.

### Negative

- The protocol is a lowest-common-denominator — features that exist in only one backend (e.g. vLLM's PagedAttention) can't be exposed without extending the protocol.
- The `transfer_cache` semantics are subtle (no-op for same-family, projection for cross-family). Bugs here are silent.

### Neutral

- The MockBackend is the de-facto protocol reference. Changes to MockBackend should be treated as protocol changes.

## Alternatives considered

### Alternative A: Inherit from a base class instead of a Protocol

Use a `BaseBackend` class with abstract methods.

**Why not:** Python's `Protocol` is structural — a class doesn't need to
inherit from anything to satisfy it. This makes it easier to wrap
third-party objects (e.g. a vLLM `LLM` instance) as a CHEN backend.

### Alternative B: Hardcode HuggingFace

Just use `transformers` everywhere.

**Why not:** untestable on CI without downloading 4 GB of weights, and
locks out vLLM/llama.cpp users.

### Alternative C: Multiple separate codebases per backend

Different packages: `chen-hf`, `chen-vllm`, `chen-llamacpp`.

**Why not:** the *point* of CHEN is that the same pipeline runs across
backends. Splitting the codebase would make cross-backend comparisons
impossible.
