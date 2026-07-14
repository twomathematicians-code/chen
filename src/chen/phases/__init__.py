"""Experimental phases of CHEN.

Each phase isolates one variable of the architecture and is implemented
as a standalone pipeline:

* :mod:`chen.phases.phase1_cascade`   — Static text-based cascade (Phase 1).
* :mod:`chen.phases.phase2_kv_pass`   — Latent state (KV-cache) handoff (Phase 2).
* :mod:`chen.phases.phase3_routing`   — Dynamic routing (Phase 3).

See ``ARCHITECTURE.md`` §9 for the experimental design and pass criteria.
"""

from __future__ import annotations

from chen.phases.phase1_cascade import CascadePipeline
from chen.phases.phase2_kv_pass import KVPassPipeline
from chen.phases.phase3_routing import RoutingPipeline

__all__ = [
    "CascadePipeline",
    "KVPassPipeline",
    "RoutingPipeline",
]
