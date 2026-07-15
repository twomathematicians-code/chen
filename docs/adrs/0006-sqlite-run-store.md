# ADR 0006: SQLite as the run persistence layer

- Status: Accepted
- Date: 2025-01-15
- Deciders: CHEN core team

## Context

CHEN is research software. Reproducibility — "I ran this config 3 weeks
ago, what were the KPIs?" — is a first-class requirement. We need to
persist:

- The prompt and output
- The configuration (phase, backend, router, max_tokens, expert sequence)
- The KPIs (tokens, cost, latency, EPU, KV transfers)
- A timestamp
- A deterministic hash of the configuration (so identical configs are dedup-able)

The persistence layer must work for:

- A single user on a laptop (no external services).
- A small team sharing a workstation.
- Eventually, a production deployment with many concurrent writers.

## Decision

Use **SQLite** via Python's stdlib `sqlite3` module. One table, one
row per run, indexed by `run_id` (short hash) and `timestamp`.

The store is created lazily at first access. Default path is
`./chen_data/runs.sqlite3` (override via `CHEN_RUN_STORE_PATH`).

## Consequences

### Positive

- **Zero dependencies** — `sqlite3` is in Python's stdlib.
- **Single file** — easy to back up, copy, or scp to a colleague.
- **ACID** — concurrent writers are safe (SQLite handles locking).
- **Good enough scale** — SQLite handles ~1000 writes/sec, far beyond CHEN's needs.
- **Queryable** — users can `sqlite3 chen_data/runs.sqlite3 "SELECT * FROM runs WHERE epu > 3"` directly.

### Negative

- **Not multi-process safe for high concurrency** — if you spin up 10 CHEN server processes, they will contend on the SQLite file lock. For production multi-tenant deployment, swap in Postgres (the schema is trivially portable).
- **No built-in replication** — backups are a manual `cp`.
- **Schema migrations are manual** — for v0.1 we just `CREATE TABLE IF NOT EXISTS`. When the schema evolves, we'll need a migration tool (e.g. alembic).

### Neutral

- The `RunStore` class is the only write surface. Swapping SQLite for Postgres later means implementing the same `save/get/list/count` methods on a new backend.

## Alternatives considered

### Alternative A: JSON Lines file

Append one JSON object per line to `runs.jsonl`.

**Why not:** no indexing, no concurrent-write safety, slow to query.
Fine for logging, not for retrieval.

### Alternative B: Postgres from day one

Use `asyncpg` + Postgres.

**Why not:** requires the user to run Postgres. Too heavy for "just try
CHEN on my laptop." SQLite is the right default; Postgres is the right
upgrade path.

### Alternative C: MLflow

Use MLflow's experiment tracking.

**Why not:** MLflow is great for ML experiments but adds a heavy
dependency and a running server. CHEN's run store is much simpler than
what MLflow offers. Integration with MLflow is a future
*optional* layer on top of `RunStore`.
