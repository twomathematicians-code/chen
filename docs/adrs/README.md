# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for CHEN.

## What is an ADR?

An ADR is a short text document that captures *why* a decision was made.
They are not specs (what) or runbooks (how) — they explain the *why*
behind a single architectural choice.

Each ADR follows this template:

```
# ADR NNN: Title

- Status: proposed | accepted | rejected | deprecated | superseded by ADR NNN
- Date: YYYY-MM-DD
- Deciders: who was involved

## Context

What is the issue we're facing? What constraints apply?

## Decision

What did we decide to do?

## Consequences

What are the trade-offs? What becomes easier / harder?
```

## Index

| # | Title | Status | Date |
|---|-------|--------|------|
| [0001](0001-external-moe-vs-internal-moe.md) | External MoE over independent models vs. internal MoE | Accepted | 2025-01-15 |
| [0002](0002-kv-cache-transfer-protocol.md) | KV-cache transfer protocol with shape adaptation | Accepted | 2025-01-15 |
| [0003](0003-pluggable-backend-abstraction.md) | Pluggable backend abstraction (mock + hf + stubs) | Accepted | 2025-01-15 |
| [0004](0004-mock-backend-first.md) | Mock backend first — every test runs without a GPU | Accepted | 2025-01-15 |
| [0005](0005-logistic-router-default.md) | Logistic router as the default (no embedding dependency) | Accepted | 2025-01-15 |
| [0006](0006-sqlite-run-store.md) | SQLite as the run persistence layer | Accepted | 2025-01-15 |
| [0007](0007-cc0-license.md) | CC0 1.0 — public domain dedication | Accepted | 2025-01-15 |

## When to write an ADR

Write an ADR when:

* You are about to make a decision that will be hard to reverse (e.g.
  choosing a database, choosing a serialization format, locking in a
  public API surface).
* You are about to *not* do something that "everyone else does" (e.g.
  not using LangChain, not using asyncio).
* You are introducing a new module, dependency, or abstraction boundary.

Do **not** write an ADR for:

* Bug fixes.
* Implementation details that don't affect the architecture.
* Things that are obvious from the code.

## Process

1. Copy `0000-template.md` to `NNNN-short-title.md`.
2. Fill in the template.
3. Open a PR. The ADR is "Proposed" until the PR is merged.
4. On merge, the ADR becomes "Accepted".
5. If an ADR is later reversed, mark it "Superseded by ADR NNNN" and write
   a new ADR explaining the reversal. Do not delete the old one.
