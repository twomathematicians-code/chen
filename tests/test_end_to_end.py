"""End-to-end smoke test: run all three phases and the benchmark suite."""

from __future__ import annotations

import pytest

from chen.backends.mock import MockBackend
from chen.benchmarks import TASK_REGISTRY, BenchmarkRunner
from chen.benchmarks.kpis import BaselineMetrics, KPIs
from chen.core.expert import Expert, ExpertRole
from chen.core.router import HybridRouter, LogisticRouter
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
            name="coder",
            role=ExpertRole.CODER,
            backend=MockBackend(params_m=7_000, role_hint="coder"),
        ),
        Expert(
            name="synthesizer",
            role=ExpertRole.SYNTHESIZER,
            backend=MockBackend(params_m=3_000, role_hint="synthesizer"),
        ),
    ]


class TestEndToEnd:
    def test_phase1_runs_and_produces_metrics(self, experts):
        pipe = CascadePipeline(
            experts=experts,
            sequence=["analyst", "reasoner", "synthesizer"],
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
        result = pipe.run("Explain recursion to a 5-year-old.")
        assert isinstance(result.output, str)
        assert result.metrics.total_tokens > 0
        assert result.metrics.total_cost_usd > 0
        assert result.metrics.epu > 0
        assert result.metrics.cost_per_1m_tokens > 0

    def test_phase2_runs_and_uses_kv_cache(self, experts):
        pipe = KVPassPipeline(
            experts=experts,
            sequence=["analyst", "reasoner", "synthesizer"],
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
        result = pipe.run("Explain quantum entanglement.")
        assert result.metrics.kv_cache_transfers >= 2
        assert result.metrics.latent_nuance_score > 0

    def test_phase3_runs_with_router(self, experts):
        router = LogisticRouter.from_experts(experts)
        pipe = RoutingPipeline(
            experts=experts,
            router=router,
            handoff="kv_cache",
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
        result = pipe.run("Debug this Python code: def foo(): pass")
        assert len(result.selected_experts) >= 1
        assert result.selected_experts[-1] == "synthesizer"

    def test_benchmark_suite_runs_all_tasks(self, experts):
        pipe = CascadePipeline(
            experts=experts,
            sequence=["analyst", "synthesizer"],
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
        runner = BenchmarkRunner(
            pipeline=pipe,
            baseline=BaselineMetrics(params_m=70_000, accuracy=0.5, latency_ms=1500.0),
            kpis=KPIs(epu_target=1.0, cost_reduction_target_pct=10.0),
        )
        results = runner.run_all(max_tokens=32)
        assert len(results) == len(TASK_REGISTRY)
        for r in results:
            assert r.kpi_report is not None
            assert 0.0 <= r.avg_accuracy <= 1.0

    def test_deterministic_repeated_runs(self, experts):
        pipe = CascadePipeline(
            experts=experts,
            sequence=["analyst", "synthesizer"],
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
        r1 = pipe.run("hello world")
        r2 = pipe.run("hello world")
        # MockBackend is deterministic, so outputs should match.
        assert r1.output == r2.output
        assert r1.metrics.total_tokens == r2.metrics.total_tokens

    def test_three_routers_all_work(self, experts):
        from chen.core.router import CosineRouter

        for make in [
            lambda: LogisticRouter.from_experts(experts),
            lambda: CosineRouter.from_experts(experts),
            lambda: HybridRouter.from_experts(experts),
        ]:
            router = make()
            pipe = RoutingPipeline(
                experts=experts,
                router=router,
                handoff="text",
                memory_retrieve_k=0,
                write_intermediate_to_memory=False,
            )
            result = pipe.run("test prompt")
            assert len(result.selected_experts) >= 1


class TestPublicAPI:
    """Verify the public API exports work as documented."""

    def test_top_level_imports(self):
        import chen

        assert hasattr(chen, "Expert")
        assert hasattr(chen, "ExpertRole")
        assert hasattr(chen, "Router")
        assert hasattr(chen, "LogisticRouter")
        assert hasattr(chen, "Memory")
        assert hasattr(chen, "InMemoryMemory")
        assert hasattr(chen, "KVCache")
        assert hasattr(chen, "Pipeline")
        assert hasattr(chen, "PipelineResult")
        assert hasattr(chen, "CascadePipeline")
        assert hasattr(chen, "KVPassPipeline")
        assert hasattr(chen, "RoutingPipeline")
        assert hasattr(chen, "KPIs")
        assert hasattr(chen, "BenchmarkRunner")
        assert hasattr(chen, "MockBackend")
        assert hasattr(chen, "get_backend")
        assert hasattr(chen, "register_backend")
        assert chen.__version__ == "0.2.0"

    def test_backends_subpackage_imports(self):
        from chen.backends import (
            BACKEND_REGISTRY,
            get_backend,
            list_backends,
        )

        assert "mock" in BACKEND_REGISTRY
        assert "mock" in list_backends()
        b = get_backend("mock")
        assert b is not None

    def test_phases_subpackage_imports(self):
        from chen.phases import (
            CascadePipeline,
            KVPassPipeline,
            RoutingPipeline,
        )

        assert CascadePipeline is not None
        assert KVPassPipeline is not None
        assert RoutingPipeline is not None

    def test_benchmarks_subpackage_imports(self):
        from chen.benchmarks import (
            TASK_REGISTRY,
            BenchmarkRunner,
            KPIs,
        )

        assert KPIs is not None
        assert BenchmarkRunner is not None
        assert len(TASK_REGISTRY) > 0
