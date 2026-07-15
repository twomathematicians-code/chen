"""Pipeline orchestration overhead benchmarks.

These measure the *orchestration* overhead (router + pipeline + metrics),
not the inference time. All benchmarks use MockBackend, which has <1ms
per-call latency.
"""

from __future__ import annotations

import pytest

from chen.backends.mock import MockBackend
from chen.core.expert import Expert, ExpertRole
from chen.core.router import LogisticRouter
from chen.phases.phase1_cascade import CascadePipeline
from chen.phases.phase2_kv_pass import KVPassPipeline
from chen.phases.phase3_routing import RoutingPipeline


@pytest.fixture
def experts():
    return [
        Expert(
            name="analyst",
            role=ExpertRole.ANALYST,
            backend=MockBackend(params_m=3_000, role_hint="analyst"),
        ),
        Expert(
            name="reasoner",
            role=ExpertRole.REASONER,
            backend=MockBackend(params_m=8_000, role_hint="reasoner"),
        ),
        Expert(
            name="synthesizer",
            role=ExpertRole.SYNTHESIZER,
            backend=MockBackend(params_m=3_000, role_hint="synthesizer"),
        ),
    ]


@pytest.fixture
def long_prompt():
    return "word " * 500  # ~500 tokens


def test_phase1_cascade_benchmark(benchmark, experts, long_prompt):
    """Benchmark Phase 1 cascade (text handoff)."""
    pipe = CascadePipeline(
        experts=experts,
        sequence=["analyst", "reasoner", "synthesizer"],
        memory_retrieve_k=0,
        write_intermediate_to_memory=False,
    )
    result = benchmark(pipe.run, long_prompt)
    assert result.metrics.total_tokens > 0


def test_phase2_kv_pass_benchmark(benchmark, experts, long_prompt):
    """Benchmark Phase 2 KV-cache passing."""
    pipe = KVPassPipeline(
        experts=experts,
        sequence=["analyst", "reasoner", "synthesizer"],
        memory_retrieve_k=0,
        write_intermediate_to_memory=False,
    )
    result = benchmark(pipe.run, long_prompt)
    assert result.metrics.kv_cache_transfers >= 2


def test_phase3_routing_benchmark(benchmark, experts, long_prompt):
    """Benchmark Phase 3 dynamic routing."""
    router = LogisticRouter.from_experts(experts)
    pipe = RoutingPipeline(
        experts=experts,
        router=router,
        handoff="text",
        memory_retrieve_k=0,
        write_intermediate_to_memory=False,
    )
    result = benchmark(pipe.run, long_prompt)
    assert len(result.selected_experts) >= 1


def test_router_only_benchmark(benchmark, experts):
    """Benchmark just the router (no pipeline)."""
    router = LogisticRouter.from_experts(experts)
    result = benchmark(router.route, "debug this python code", experts)
    assert len(result) >= 1


def test_mock_backend_generate_benchmark(benchmark):
    """Benchmark MockBackend.generate() in isolation."""
    backend = MockBackend(params_m=3_000)
    result = benchmark(backend.generate, "hello world" * 100)
    assert isinstance(result, str)


def test_mock_backend_encode_benchmark(benchmark):
    """Benchmark MockBackend.encode() (KV-cache production) in isolation."""
    backend = MockBackend(params_m=3_000)
    result = benchmark(backend.encode, "hello world" * 100)
    assert result.seq_len > 0


def test_mock_backend_transfer_cache_benchmark(benchmark):
    """Benchmark MockBackend.transfer_cache() in isolation."""
    src = MockBackend(params_m=3_000)
    dst = MockBackend(params_m=8_000, n_layers=8, n_heads=16, head_dim=32)
    cache = src.encode("test prompt")
    result = benchmark(dst.transfer_cache, cache)
    assert result.source_layer_count == dst.n_layers


def test_memory_retrieve_benchmark(benchmark):
    """Benchmark memory store retrieval with 1000 entries."""
    from chen.core.memory import InMemoryMemory, MemoryEntry

    mem = InMemoryMemory()
    for i in range(1000):
        mem.write(MemoryEntry(text=f"entry number {i}", role="analyst"))
    result = benchmark(mem.retrieve, "entry 500", k=4)
    assert len(result) <= 4


def test_run_store_save_benchmark(benchmark, tmp_path):
    """Benchmark SQLite run store writes."""
    from chen.persistence.run_store import RunRecord, RunStore

    store = RunStore(tmp_path / "bench.sqlite3")
    record = RunRecord(
        run_id="bench",
        config_hash="abc" * 21,
        prompt="test prompt",
        output="test output",
        phase=1,
        backend="mock",
    )
    benchmark(store.save, record)
    assert store.count() >= 1


def test_config_hash_benchmark(benchmark):
    """Benchmark config hashing (used for run_id)."""
    from chen.reproducibility import hash_config

    config = {
        "phase": 1,
        "backend": "mock",
        "router": "logistic",
        "max_tokens": 128,
        "sequence": ["analyst", "reasoner", "synthesizer"],
    }
    result = benchmark(hash_config, config)
    assert len(result) == 64
