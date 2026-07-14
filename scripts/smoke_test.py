"""CHEN smoke test — verifies the package is installed and runnable.

This script is intended for CI and for users who want to verify their
installation quickly. It runs each of the three phases on the MockBackend
and prints OK/FAIL for each.

Usage:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import traceback


def check(label: str, fn) -> bool:
    """Run ``fn()`` and print OK/FAIL. Returns True on success."""
    try:
        fn()
        print(f"  [OK]   {label}")
        return True
    except Exception:
        print(f"  [FAIL] {label}")
        traceback.print_exc()
        return False


def test_imports() -> None:
    import chen

    assert chen.__version__ == "0.1.0"
    from chen.backends import list_backends

    backends = list_backends()
    assert "mock" in backends, f"mock backend missing from {backends}"


def test_phase1() -> None:
    from chen.backends.mock import MockBackend
    from chen.core.expert import Expert, ExpertRole
    from chen.phases.phase1_cascade import CascadePipeline

    experts = [
        Expert(name="a", role=ExpertRole.ANALYST, backend=MockBackend(params_m=3_000)),
        Expert(name="s", role=ExpertRole.SYNTHESIZER, backend=MockBackend(params_m=3_000)),
    ]
    pipe = CascadePipeline(
        experts=experts,
        memory_retrieve_k=0,
        write_intermediate_to_memory=False,
    )
    result = pipe.run("hello world")
    assert isinstance(result.output, str) and len(result.output) > 0


def test_phase2() -> None:
    from chen.backends.mock import MockBackend
    from chen.core.expert import Expert, ExpertRole
    from chen.phases.phase2_kv_pass import KVPassPipeline

    experts = [
        Expert(name="a", role=ExpertRole.ANALYST, backend=MockBackend(params_m=3_000)),
        Expert(name="s", role=ExpertRole.SYNTHESIZER, backend=MockBackend(params_m=3_000)),
    ]
    pipe = KVPassPipeline(
        experts=experts,
        memory_retrieve_k=0,
        write_intermediate_to_memory=False,
    )
    result = pipe.run("hello world")
    assert isinstance(result.output, str) and len(result.output) > 0
    assert result.metrics.kv_cache_transfers >= 1


def test_phase3() -> None:
    from chen.backends.mock import MockBackend
    from chen.core.expert import Expert, ExpertRole
    from chen.core.router import LogisticRouter
    from chen.phases.phase3_routing import RoutingPipeline

    experts = [
        Expert(name="a", role=ExpertRole.ANALYST, backend=MockBackend(params_m=3_000)),
        Expert(name="s", role=ExpertRole.SYNTHESIZER, backend=MockBackend(params_m=3_000)),
    ]
    router = LogisticRouter.from_experts(experts)
    pipe = RoutingPipeline(
        experts=experts,
        router=router,
        handoff="text",
        memory_retrieve_k=0,
        write_intermediate_to_memory=False,
    )
    result = pipe.run("hello world")
    assert len(result.selected_experts) >= 1


def test_benchmarks() -> None:
    from chen.backends.mock import MockBackend
    from chen.benchmarks import BenchmarkRunner, TASK_REGISTRY
    from chen.benchmarks.kpis import BaselineMetrics
    from chen.core.expert import Expert, ExpertRole
    from chen.phases.phase1_cascade import CascadePipeline

    experts = [Expert(name="g", role=ExpertRole.SYNTHESIZER, backend=MockBackend(params_m=3_000))]
    pipe = CascadePipeline(
        experts=experts,
        memory_retrieve_k=0,
        write_intermediate_to_memory=False,
    )
    runner = BenchmarkRunner(
        pipeline=pipe,
        baseline=BaselineMetrics(params_m=70_000, accuracy=0.5),
    )
    result = runner.run_task("qa_factual", max_tokens=32)
    assert result.kpi_report is not None
    assert result.task_name == "qa_factual"
    assert len(TASK_REGISTRY) >= 5


def main() -> int:
    print("CHEN smoke test")
    print("-" * 50)
    all_ok = True
    all_ok &= check("imports", test_imports)
    all_ok &= check("Phase 1 (cascade)", test_phase1)
    all_ok &= check("Phase 2 (KV-pass)", test_phase2)
    all_ok &= check("Phase 3 (routing)", test_phase3)
    all_ok &= check("benchmarks", test_benchmarks)
    print("-" * 50)
    print("Result:", "ALL OK" if all_ok else "SOME FAILURES")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
