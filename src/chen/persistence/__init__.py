"""Persistence layer — SQLite-backed run store for reproducibility.

Every pipeline run can be persisted to a SQLite database via
:class:`RunStore`. This enables:

* **Reproducibility** — re-run any past configuration by ``run_id``.
* **Auditing** — track what was run, when, with what config, and what
  the KPIs were.
* **Diffing** — compare KPIs across runs to detect regressions.

The schema is intentionally minimal — one row per run with the prompt,
output, KPIs, and a hash of the configuration. For richer tracking
(per-expert metrics, KV transfer details), see the in-memory
:class:`~chen.core.pipeline.PipelineResult` object.
"""

from __future__ import annotations

from chen.persistence.run_store import RunRecord, RunStore

__all__ = ["RunRecord", "RunStore"]
