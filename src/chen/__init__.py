"""CHEN — Collaborative Heterogeneous Expert Network.

A distributed inference architecture that replaces a single monolithic
hyper-scale model with a coordinated garage of specialized, low-parameter
models (3B-8B). Routes tokens through a dynamic pipeline and shares latent
memory states (KV-caches) between models.

Public API:
    Expert, ExpertRole       — micro-expert wrapper
    Router, LogisticRouter   — prompt → expert routing
    Memory, MemoryEntry      — shared external RAG+ store
    KVCache                  — latent state container
    InferenceBackend         — pluggable backend protocol
    Pipeline                 — orchestration base class

    CascadePipeline          — Phase 1 (text handoff)
    KVPassPipeline           — Phase 2 (KV-cache handoff)
    RoutingPipeline          — Phase 3 (dynamic routing)

    PipelineResult, Metrics  — return types with KPIs

    KPIs                     — EPU, cost-per-1M, latency-to-accuracy
    BenchmarkRunner          — benchmark harness
"""

from __future__ import annotations

from chen.backends.base import BackendCapabilities, InferenceBackend
from chen.backends.mock import MockBackend
from chen.backends.registry import (
    BACKEND_REGISTRY,
    get_backend,
    list_backends,
    register_backend,
)
from chen.benchmarks.kpis import KPIReport, KPIs
from chen.benchmarks.runner import BenchmarkRunner
from chen.benchmarks.tasks import TASK_REGISTRY, BenchmarkTask
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
from chen.observability.logging import configure_logging, get_logger
from chen.persistence.run_store import RunRecord, RunStore
from chen.phases.phase1_cascade import CascadePipeline
from chen.phases.phase2_kv_pass import KVPassPipeline
from chen.phases.phase3_routing import RoutingPipeline
from chen.reproducibility import RunContext, hash_config, seed_everything, track_run

# Security & encryption (graceful import — cryptography is a core dep).
try:
    from chen.security import (
        CrypticStream,
        CryptoError,
        Decryptor,
        EncryptedBackend,
        EncryptionConfig,
        Encryptor,
        KeyStore,
        SecuritySettings,
    )

    _SECURITY_AVAILABLE = True
except ImportError:
    _SECURITY_AVAILABLE = False

__version__ = "0.3.0"

__all__ = [
    # Core
    "Expert",
    "ExpertRole",
    "Router",
    "LogisticRouter",
    "CosineRouter",
    "HybridRouter",
    "RouterDecision",
    "Memory",
    "MemoryEntry",
    "InMemoryMemory",
    "KVCache",
    "IncompatibleCacheError",
    "Pipeline",
    "PipelineResult",
    "ExpertMetrics",
    "AggregateMetrics",
    "Settings",
    "CostModel",
    # Phases
    "CascadePipeline",
    "KVPassPipeline",
    "RoutingPipeline",
    # Backends
    "InferenceBackend",
    "BackendCapabilities",
    "MockBackend",
    "get_backend",
    "register_backend",
    "list_backends",
    "BACKEND_REGISTRY",
    # Benchmarks
    "KPIs",
    "KPIReport",
    "BenchmarkRunner",
    "BenchmarkTask",
    "TASK_REGISTRY",
    # Observability
    "configure_logging",
    "get_logger",
    # Persistence
    "RunRecord",
    "RunStore",
    # Reproducibility
    "hash_config",
    "seed_everything",
    "track_run",
    "RunContext",
    # Security & Encryption
    "CrypticStream",
    "CryptoError",
    "EncryptedBackend",
    "EncryptionConfig",
    "Encryptor",
    "Decryptor",
    "KeyStore",
    "SecuritySettings",
    # Meta
    "__version__",
]
