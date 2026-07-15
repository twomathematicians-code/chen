# ADR 0010: PostgreSQL as optional production run store

- Status: Accepted
- Date: 2025-07-15
- Deciders: CHEN core team

## Context

v0.2.0 used SQLite as the run store. SQLite is excellent for
single-node development but fails under concurrent writes when CHEN
runs as multiple replicas behind a load balancer:

- SQLite uses file-level write locking — only one writer at a time.
- Under high concurrency, writes fail with `database is locked`.
- No replication — each replica has its own file, so runs are not shared.

For production multi-replica deployments, we need a store that supports
concurrent writes and is shared across replicas.

## Decision

Add `PostgresRunStore` as an optional backend for the run store. The
default remains SQLite (for dev/single-node); Postgres is opt-in via
environment variable.

### Interface

`PostgresRunStore` implements the same interface as `RunStore`
(`save`, `get`, `list`, `count`, `close`) but all methods are `async`.
This is necessary because Postgres I/O is async (via `asyncpg`).

### Configuration

```bash
export CHEN_RUN_STORE_BACKEND=postgres
export CHEN_RUN_STORE_DSN=postgresql://user:pass@localhost:5432/chen
```

### Schema

The Postgres schema mirrors the SQLite schema, with two improvements:
- `selected_experts` uses `JSONB` (queryable JSON).
- `timestamp` uses `TIMESTAMPTZ` (timezone-aware).
- Additional index on `tenant_id` for multi-tenant queries.

### Connection pooling

Uses `asyncpg.create_pool()` with configurable min/max pool size
(default 2/10). Connections are reused across requests.

## Consequences

### Positive

- Multi-replica deployments work — all replicas share the same Postgres.
- Concurrent writes are safe — Postgres handles row-level locking.
- JSONB enables rich queries on `selected_experts` (e.g., "find all runs that used the 'reasoner' expert").
- Timezone-aware timestamps prevent confusion across regions.

### Negative

- Async interface means the server endpoints must be async (they already are in FastAPI).
- Adds `asyncpg` as an optional dependency (`pip install asyncpg`).
- Postgres must be provisioned and managed — more operational burden than SQLite.
- Schema migrations are manual in v0.3.0 — Alembic integration is roadmap.

### Neutral

- The `get_run_store()` factory function abstracts the choice — application code doesn't change.
- SQLite remains the default — no breaking change for existing users.

## Alternatives considered

### Alternative A: SQLite with WAL mode

Enable Write-Ahead Logging in SQLite for better concurrent-read performance.

**Why not:** WAL helps with concurrent reads but writes still serialize. Doesn't solve the multi-replica sharing problem.

### Alternative B: Redis

Use Redis as the run store.

**Why not:** Redis is in-memory (or persistent with RDB/AOF), but its data model is key-value, not relational. Run records have structure that benefits from SQL queries. Redis is better suited for caching and rate limiting than for persistent run history.

### Alternative C: MongoDB

Use MongoDB as the run store.

**Why not:** MongoDB is document-oriented, which fits the run record structure well, but adds a non-SQL dependency. Postgres with JSONB gives us the best of both worlds — relational queries when needed, document queries when convenient. Postgres is also more widely deployed in enterprise environments.
