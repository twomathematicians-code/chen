# ADR 0002: KV-cache transfer protocol with shape adaptation

- Status: Accepted
- Date: 2025-01-15
- Deciders: CHEN core team

## Context

Phase 2 of CHEN passes KV-caches (the cached keys and values at every
attention layer) between experts instead of text. This is the key
differentiator vs. text-cascading pipelines.

Two challenges:
1. Different model families have different layer counts, head counts,
   and head dimensions. A Llama-3 KV-cache cannot be directly fed into
   a Qwen2.5 model.
2. We need to fail gracefully — if a transfer is impossible, the pipeline
   must fall back to text rather than crashing.

## Decision

The `KVCache` dataclass carries full **provenance metadata**:
`source_model`, `source_layer_count`, `source_n_heads`,
`source_hidden_size`, plus the original `source_text` for fallback.

The `transfer_cache` operation on each backend implements:

1. **Same-family** (matching layer count + head dim): no-op, return as-is.
2. **Different layer count**: truncate or zero-pad.
3. **Different head dim**: raise `IncompatibleCacheError`. A learned
   projection would be needed (not bundled in v0.1).
4. **Cross-family** with mismatched shapes: raise
   `IncompatibleCacheError`. The pipeline catches this and falls back
   to text handoff.

The pipeline records `cache_transfer_succeeded` per expert and a
pipeline-level `latent_nuance_score` = (successful transfers) /
(attempted transfers).

## Consequences

### Positive

- Graceful degradation — no prompt is ever lost; worst case we fall back to text.
- Observable — every transfer attempt is logged and metricized.
- Extensible — adding a learned projection for a specific (source, target) pair is a localized change.

### Negative

- The MockBackend's "transfer" is fake (re-encodes the source text) — this means tests pass but don't validate real cross-family transfer.
- The HuggingFaceBackend's `decode` re-encodes the source text rather than feeding the cache directly into `model.generate()` — a known limitation in v0.1 (see code comment).

### Neutral

- The `source_text` field on `KVCache` is essential for fallback but doubles memory if the prompt is long. Acceptable for v0.1.

## Alternatives considered

### Alternative A: Always re-encode text

Skip the transfer protocol entirely; always pass text between experts.

**Why not:** this is Phase 1. Phase 2's whole point is to test whether
KV-passing preserves more nuance.

### Alternative B: Require same-family experts only

Constrain the garage to models with identical architecture (e.g. all
Llama-3 variants).

**Why not:** too restrictive. The "heterogeneous" in CHEN's name is the
point. Same-family is a special case; the protocol should handle both.

### Alternative C: Bundle a universal projection

Ship a pre-trained projection matrix for every common (source, target) pair.

**Why not:** projection training is a research project on its own. We
ship the *protocol* and let users opt in to projections.
