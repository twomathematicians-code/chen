# ADR 0005: Logistic router as the default, not embedding-based

- Status: Accepted
- Date: 2025-01-15
- Deciders: CHEN core team

## Context

The router decides which experts to wake up per prompt (Phase 3).
Three reasonable designs:

1. **Logistic** — per-expert logistic classifier on hand-crafted prompt features (keyword density, length, code/math indicators).
2. **Cosine** — embedding similarity between the prompt and each expert's prototype.
3. **Hybrid** — weighted product of (1) and (2).

The default router is what users get when they don't specify one. It
shapes first impressions and the out-of-the-box experience.

## Decision

The default router is `LogisticRouter` with heuristic weights. It is:

- **Deterministic** — no model download, no sampling.
- **Fast** — <1 ms per routing decision.
- **Dependency-free** — no `sentence-transformers` needed.
- **Interpretable** — the weights are human-readable; you can see *why* a prompt routed to the Coder.

`CosineRouter` and `HybridRouter` are available for users who want
embedding-based routing (install the `memory` extra).

## Consequences

### Positive

- Zero-dependency install works out of the box.
- Routing decisions are auditable.
- Tests are deterministic — no mocked embedding model needed.

### Negative

- Heuristic weights are brittle — they encode assumptions about prompt wording that may not generalize.
- No semantic understanding — "write a function" and "implement a subroutine" route identically only if both keywords are weighted.

### Neutral

- Users who want better routing can pass `CosineRouter` or `HybridRouter` explicitly. The protocol is the same.

## Alternatives considered

### Alternative A: Cosine router as default

Use `sentence-transformers/all-MiniLM-L6-v2` for embeddings.

**Why not:** adds a 80 MB model download to the default install. Too
heavy for a "just try CHEN" experience.

### Alternative B: Trained classifier

Train a small classifier on (prompt, expert) pairs.

**Why not:** no labeled dataset exists for CHEN yet. Heuristic weights
are a reasonable starting point; learned weights are a future
contribution (see Roadmap).

### Alternative C: LLM-based routing

Use a tiny LLM (e.g. SmolLM-360M) to classify the prompt.

**Why not:** adds the "router tax" of loading another model. The
logistic router's <1ms latency is hard to beat.
