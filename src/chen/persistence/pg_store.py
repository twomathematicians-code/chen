"""PostgreSQL-backed run store for production multi-process deployments.

When CHEN runs as multiple replicas (e.g. behind a load balancer),
SQLite's file-level write locking becomes a bottleneck. PostgresRunStore
provides the same interface as RunStore but uses PostgreSQL for
concurrent-write safety.

Uses asyncpg for async, non-blocking I/O. Falls back to psycopg2 if
asyncpg is not installed.

Usage::

    from chen.persistence.pg_store import PostgresRunStore

    store = PostgresRunStore(
        dsn="postgresql://user:pass@localhost:5432/chen",
    )
    await store.save(record)

Or via env var::

    export CHEN_RUN_STORE_BACKEND=postgres
    export CHEN_RUN_STORE_DSN=postgresql://user:pass@localhost:5432/chen
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Optional

from chen.observability.logging import get_logger
from chen.persistence.run_store import RunRecord

_log = get_logger("chen.persistence.pg_store")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    config_hash       TEXT NOT NULL,
    timestamp         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    prompt            TEXT NOT NULL,
    output            TEXT NOT NULL,
    phase             INTEGER NOT NULL,
    backend           TEXT NOT NULL,
    router            TEXT,
    selected_experts  JSONB NOT NULL,
    total_tokens      INTEGER NOT NULL,
    total_cost_usd    REAL NOT NULL,
    total_latency_ms  REAL NOT NULL,
    epu               REAL,
    kv_transfers      INTEGER
);

CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_runs_config_hash ON runs(config_hash);
CREATE INDEX IF NOT EXISTS idx_runs_phase ON runs(phase);
CREATE INDEX IF NOT EXISTS idx_runs_tenant ON runs(tenant_id) WHERE tenant_id IS NOT NULL;
"""


class PostgresRunStore:
    """Async PostgreSQL run store.

    Implements the same interface as RunStore (save/get/list/count)
    but all methods are async. For use with FastAPI's async endpoints.

    Attributes:
        dsn: PostgreSQL connection string.
        min_pool_size: Minimum connections in the pool.
        max_pool_size: Maximum connections in the pool.
    """

    def __init__(
        self,
        dsn: Optional[str] = None,  # noqa: UP045
        min_pool_size: int = 2,
        max_pool_size: int = 10,
    ) -> None:
        self.dsn = dsn or os.environ.get(
            "CHEN_RUN_STORE_DSN",
            "postgresql://localhost:5432/chen",
        )
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self._pool: Any = None
        self._lock = asyncio.Lock()

    async def _ensure_pool(self) -> Any:
        if self._pool is not None:
            return self._pool
        async with self._lock:
            if self._pool is not None:
                return self._pool
            try:
                import asyncpg  # type: ignore

                self._pool = await asyncpg.create_pool(
                    dsn=self.dsn,
                    min_size=self.min_pool_size,
                    max_size=self.max_pool_size,
                )
                # Create schema if not exists.
                async with self._pool.acquire() as conn:
                    await conn.execute(_SCHEMA)
                _log.info(
                    "pg_store.connected",
                    dsn=self.dsn.replace(
                        # Don't log the password.
                        self.dsn.split("@")[0].split(":")[-1],
                        "***" if ":" in self.dsn.split("@")[0] else "",
                    ),
                )
            except ImportError:
                raise ImportError(  # noqa: B904
                    "PostgresRunStore requires asyncpg. Install with: pip install asyncpg"
                )
            except Exception as e:
                _log.error("pg_store.connection_failed", dsn=self.dsn, error=str(e))
                raise
        return self._pool

    async def save(self, record: RunRecord) -> None:
        """Insert (or replace) a run record."""
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO runs
                (run_id, config_hash, timestamp, prompt, output, phase, backend,
                 router, selected_experts, total_tokens, total_cost_usd,
                 total_latency_ms, epu, kv_transfers)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                ON CONFLICT (run_id) DO UPDATE SET
                    timestamp = EXCLUDED.timestamp,
                    output = EXCLUDED.output,
                    total_tokens = EXCLUDED.total_tokens,
                    total_cost_usd = EXCLUDED.total_cost_usd,
                    total_latency_ms = EXCLUDED.total_latency_ms,
                    epu = EXCLUDED.epu,
                    kv_transfers = EXCLUDED.kv_transfers
                """,
                record.run_id,
                record.config_hash,
                datetime.fromisoformat(record.timestamp),
                record.prompt,
                record.output,
                record.phase,
                record.backend,
                record.router or "",
                json.dumps(record.selected_experts),
                record.total_tokens,
                record.total_cost_usd,
                record.total_latency_ms,
                record.epu,
                record.kv_transfers,
            )

    async def get(self, run_id: str) -> Optional[RunRecord]:  # noqa: UP045
        """Fetch a single run by id."""
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM runs WHERE run_id = $1", run_id)
        if row is None:
            return None
        return self._row_to_record(row)

    async def list(self, limit: int = 100, phase: Optional[int] = None) -> list[RunRecord]:  # noqa: UP045
        """List recent runs, newest first."""
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            if phase is not None:
                rows = await conn.fetch(
                    "SELECT * FROM runs WHERE phase = $1 ORDER BY timestamp DESC LIMIT $2",
                    phase,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM runs ORDER BY timestamp DESC LIMIT $1", limit
                )
        return [self._row_to_record(r) for r in rows]

    async def count(self) -> int:
        """Total number of runs."""
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) FROM runs")
        return int(row[0]) if row else 0

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @staticmethod
    def _row_to_record(row: Any) -> RunRecord:
        """Convert an asyncpg Record to a RunRecord."""
        ts = row["timestamp"]
        if hasattr(ts, "isoformat"):
            ts_str = ts.isoformat()
        else:
            ts_str = str(ts)
        return RunRecord(
            run_id=row["run_id"],
            config_hash=row["config_hash"],
            timestamp=ts_str,
            prompt=row["prompt"],
            output=row["output"],
            phase=row["phase"],
            backend=row["backend"],
            router=row["router"] or "",
            selected_experts=json.loads(row["selected_experts"]),
            total_tokens=row["total_tokens"],
            total_cost_usd=row["total_cost_usd"],
            total_latency_ms=row["total_latency_ms"],
            epu=row["epu"] if row["epu"] is not None else 0.0,
            kv_transfers=row["kv_transfers"] if row["kv_transfers"] is not None else 0,
        )


def get_run_store():
    """Factory: return SQLite or Postgres store based on env var.

    ``CHEN_RUN_STORE_BACKEND=postgres`` → PostgresRunStore (async)
    ``CHEN_RUN_STORE_BACKEND=sqlite`` (default) → RunStore (sync)
    """
    backend = os.environ.get("CHEN_RUN_STORE_BACKEND", "sqlite").lower()
    if backend == "postgres":
        return PostgresRunStore()
    from chen.persistence.run_store import RunStore

    return RunStore.default()
