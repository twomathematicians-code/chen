# ADR 0001: External MoE over independent models, not internal MoE

- Status: Accepted
- Date: 2025-01-15
- Deciders: CHEN core team

## Context

Mixture-of-Experts (MoE) is a well-established technique where multiple
specialized subnetworks ("experts") process different parts of the input,
and a gating network decides which expert to use per token. Existing
production MoE systems — Mistral Mixtral, GShard, Switch Transformer —
implement MoE *inside* a single neural network: the experts share a
tokenizer, share an embedding space, and live in the same model binary.

The CHEN project asks: can we do MoE *externally* — between completely
separate, independently-trained models, each potentially from a
different family (Llama, Qwen, Mistral)?

This decision shapes every other architectural choice: backend
abstraction, KV-cache transfer protocol, router design, deployment
story.

## Decision

CHEN will implement **external MoE**: experts are independent processes
(or at least independent model binaries), communicating over a defined
protocol. The protocol passes either text (Phase 1) or KV-caches
(Phase 2) between experts.

## Consequences

### Positive

- **Independent upgrade** — replace one expert without retraining the others.
- **Heterogeneous models** — mix Llama + Qwen + Mistral in one pipeline.
- **Cost isolation** — only wake the experts the prompt actually needs (Phase 3).
- **Swappable backends** — MockBackend for tests, HuggingFaceBackend for research, vLLM for production, all behind the same protocol.

### Negative

- **No shared embedding space** — cross-family KV-cache transfer requires a learned projection (Phase 2).
- **Higher orchestration overhead** — the router and pipeline add latency vs. a single forward pass.
- **Memory duplication** — each expert loads its own tokenizer and model weights.
- **No gradient flow** — experts cannot be jointly fine-tuned end-to-end (by design, but limits optimization opportunities).

### Neutral

- The protocol is the API surface. Changing it is a breaking change.
- The MockBackend is the reference implementation of the protocol.

## Alternatives considered

### Alternative A: Internal MoE (Mixtral-style)

Implement one model with multiple expert subnetworks inside it.

**Why not:** this is what everyone else already does. The CHEN thesis is
specifically that *external* MoE works. Implementing internal MoE would
duplicate existing work without testing the hypothesis.

### Alternative B: Single model with dynamic routing at the API level

Use one model (e.g. Llama-3-8B) but route different prompt types to
different *prompt templates* or *LoRA adapters*.

**Why not:** this still requires loading all 8B parameters for every
query. The "monolith tax" CHEN is trying to eliminate is not addressed.

### Alternative C: Cloud-API orchestration (LangChain-style)

Chain calls to hosted APIs (OpenAI, Anthropic) rather than running local models.

**Why not:** defeats the purpose. Cloud APIs are opaque, can't expose
KV-caches, and don't allow the cost-isolation experiments CHEN needs.
Also locks users into vendor pricing.
