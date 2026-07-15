"""SQLite run store — persists pipeline runs for reproducibility.

Schema (single table ``runs``):

    run_id            TEXT PRIMARY KEY   -- short hash of the config
    config_hash       TEXT NOT NULL      -- full SHA-256 of the config
    timestamp         TEXT NOT NULL      -- ISO 8601 UTC
    prompt            TEXT NOT NULL
    output            TEXT NOT NULL
    phase             INTEGER NOT NULL
    backend           TEXT NOT NULL
    router            TEXT
    selected_experts  TEXT NOT NULL      -- JSON array
    total_tokens      INTEGER NOT NULL
    total_cost_usd    REAL NOT NULL
    total_latency_ms  REAL NOT NULL
    epu               REAL
    kv_transfers      INTEGER

The store is created lazily on first access. Default location is
``./chen_data/runs.sqlite3`` (override with ``CHEN_RUN_STORE_PATH``).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    config_hash       TEXT NOT NULL,
    timestamp         TEXT NOT NULL,
    prompt            TEXT NOT NULL,
    output            TEXT NOT NULL,
    phase             INTEGER NOT NULL,
    backend           TEXT NOT NULL,
    router            TEXT,
    selected_experts  TEXT NOT NULL,
    total_tokens      INTEGER NOT NULL,
    total_cost_usd    REAL NOT NULL,
    total_latency_ms  REAL NOT NULL,
    epu               REAL,
    kv_transfers      INTEGER,
    trace_id          TEXT,
    tenant_id         TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(timestamp);
CREATE INDEX IF NOT EXISTS idx_runs_config_hash ON runs(config_hash);
CREATE INDEX IF NOT EXISTS idx_runs_phase ON runs(phase);
CREATE INDEX IF NOT EXISTS idx_runs_trace_id ON runs(trace_id) WHERE trace_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_runs_tenant ON runs(tenant_id) WHERE tenant_id IS NOT NULL;
"""


@dataclass
class RunRecord:
    """One persisted pipeline run."""

    run_id: str
    config_hash: str
    prompt: str
    output: str
    phase: int
    backend: str
    router: str = ""
    selected_experts: list[str] = field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    epu: float = 0.0
    kv_transfers: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    trace_id: str = ""
    tenant_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["selected_experts"] = json.dumps(d["selected_experts"])
        return d

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> RunRecord:
        return cls(
            run_id=row["run_id"],
            config_hash=row["config_hash"],
            timestamp=row["timestamp"],
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
            trace_id=row["trace_id"] if "trace_id" in row.keys() and row["trace_id"] else "",
            tenant_id=row["tenant_id"] if "tenant_id" in row.keys() and row["tenant_id"] else "",
        )


class RunStore:
    """SQLite-backed run store. Thread-safe via a single connection lock."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        # Run lightweight migrations for existing databases (add columns
        # that were introduced in later versions).
        self._migrate()

    def _migrate(self) -> None:
        """Add columns that were introduced after the initial schema.

        Uses ``PRAGMA table_info`` to check existing columns and only
        adds missing ones. This is a simple migration strategy that
        works for additive schema changes.
        """
        with self._lock:
            cols = {row[1] for row in self._conn.execute("PRAGMA table_info(runs)").fetchall()}
            if "trace_id" not in cols:
                self._conn.execute("ALTER TABLE runs ADD COLUMN trace_id TEXT")
            if "tenant_id" not in cols:
                self._conn.execute("ALTER TABLE runs ADD COLUMN tenant_id TEXT")

    @classmethod
    def default(cls) -> RunStore:
        """Return a RunStore at the default path ($CHEN_RUN_STORE_PATH or ./chen_data/runs.sqlite3)."""
        path = os.environ.get("CHEN_RUN_STORE_PATH", "./chen_data/runs.sqlite3")
        return cls(path)

    def save(self, record: RunRecord) -> None:
        """Insert (or replace) a run record."""
        d = record.to_dict()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO runs
                (run_id, config_hash, timestamp, prompt, output, phase, backend,
                 router, selected_experts, total_tokens, total_cost_usd,
                 total_latency_ms, epu, kv_transfers, trace_id, tenant_id)
                VALUES
                (:run_id, :config_hash, :timestamp, :prompt, :output, :phase,
                 :backend, :router, :selected_experts, :total_tokens,
                 :total_cost_usd, :total_latency_ms, :epu, :kv_transfers,
                 :trace_id, :tenant_id)
                """,
                d,
            )

    def get(self, run_id: str) -> RunRecord | None:
        """Fetch a single run by id, or None if not found."""
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return RunRecord.from_row(row) if row else None

    def list(self, limit: int = 100, phase: int | None = None) -> list[RunRecord]:
        """List recent runs, newest first."""
        if phase is not None:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT * FROM runs WHERE phase = ? ORDER BY timestamp DESC LIMIT ?",
                    (phase, limit),
                ).fetchall()
        else:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,)
                ).fetchall()
        return [RunRecord.from_row(r) for r in rows]

    def count(self) -> int:
        """Total number of runs."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM runs").fetchone()
        return int(row[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()
