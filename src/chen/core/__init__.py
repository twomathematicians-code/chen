"""Core orchestration primitives for CHEN.

This package contains the building blocks that the phases compose:

* :mod:`chen.core.kv_cache`  — the ``KVCache`` dataclass & transfer protocol.
* :mod:`chen.core.expert`    — the ``Expert`` wrapper around a backend.
* :mod:`chen.core.router`    — prompt → expert routing (logistic, cosine, hybrid).
* :mod:`chen.core.memory`    — shared external RAG+ store.
* :mod:`chen.core.pipeline`  — base ``Pipeline`` class and result types.
* :mod:`chen.core.config`    — settings & cost model.
"""

from __future__ import annotations

from chen.core.config import CostModel, Settings
from chen.core.expert import Expert, ExpertRole
from chen.core.kv_cache import IncompatibleCacheError, KVCache
from chen.core.memory import InMemoryMemory, Memory, MemoryEntry
from chen.core.pipeline import (
    AggregateMetrics,
    ExpertMetrics,
    Pipeline,
    PipelineResult,
)
from chen.core.router import (
    CosineRouter,
    HybridRouter,
    LogisticRouter,
    Router,
    RouterDecision,
)

__all__ = [
    "Settings",
    "CostModel",
    "Expert",
    "ExpertRole",
    "KVCache",
    "IncompatibleCacheError",
    "Memory",
    "MemoryEntry",
    "InMemoryMemory",
    "Router",
    "LogisticRouter",
    "CosineRouter",
    "HybridRouter",
    "RouterDecision",
    "Pipeline",
    "PipelineResult",
    "ExpertMetrics",
    "AggregateMetrics",
]
