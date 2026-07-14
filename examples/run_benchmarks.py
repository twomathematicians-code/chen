"""Run the full benchmark suite and print KPI reports.

Runs every registered benchmark task through a CHEN pipeline (Phase 1
by default; pass ``--phase 2`` or ``--phase 3`` to swap) and prints the
EPU, cost-per-1M-tokens, and latency-to-accuracy KPIs against a
monolithic 70B baseline.

Usage::

    python examples/run_benchmarks.py
    python examples/run_benchmarks.py --phase 3 --router hybrid
    python examples/run_benchmarks.py --baseline-params 70000  # 70B baseline
"""

from __future__ import annotations

import argparse
import sys

from chen.backends.mock import MockBackend
from chen.benchmarks import TASK_REGISTRY, BenchmarkRunner
from chen.benchmarks.kpis import BaselineMetrics, KPIs
from chen.core.expert import Expert, ExpertRole
from chen.phases.phase1_cascade import CascadePipeline
from chen.phases.phase2_kv_pass import KVPassPipeline
from chen.phases.phase3_routing import RoutingPipeline
from chen.core.router import (
    CosineRouter,
    HybridRouter,
    LogisticRouter,
)


def build_experts() -> list[Expert]:
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


def make_pipeline(phase: int, router_kind: str, experts: list[Expert]):
    if phase == 1:
        return CascadePipeline(
            experts=experts,
            sequence=["analyst", "reasoner", "synthesizer"],
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
    elif phase == 2:
        return KVPassPipeline(
            experts=experts,
            sequence=["analyst", "reasoner", "synthesizer"],
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
    elif phase == 3:
        if router_kind == "logistic":
            router = LogisticRouter.from_experts(experts)
        elif router_kind == "cosine":
            router = CosineRouter.from_experts(experts)
        elif router_kind == "hybrid":
            router = HybridRouter.from_experts(experts)
        else:
            raise ValueError(f"Unknown router: {router_kind}")
        return RoutingPipeline(
            experts=experts,
            router=router,
            handoff="kv_cache",
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
    raise ValueError(f"Unknown phase: {phase}")


def main() -> int:
    parser = argparse.ArgumentParser(description="CHEN benchmark suite.")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--router", choices=["logistic", "cosine", "hybrid"], default="logistic")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--baseline-params",
        type=int,
        default=70_000,
        help="Baseline monolith params in millions (default 70_000 = 70B).",
    )
    parser.add_argument(
        "--baseline-accuracy",
        type=float,
        default=0.85,
        help="Baseline monolith avg accuracy on the benchmark (0..1).",
    )
    parser.add_argument(
        "--baseline-latency-ms",
        type=float,
        default=1500.0,
        help="Baseline monolith avg per-query latency in ms.",
    )
    parser.add_argument(
        "--epu-target",
        type=float,
        default=3.0,
        help="Minimum EPU for 'met' verdict.",
    )
    parser.add_argument(
        "--cost-target-pct",
        type=float,
        default=80.0,
        help="Minimum cost reduction %% for 'met' verdict.",
    )
    args = parser.parse_args()

    experts = build_experts()
    pipeline = make_pipeline(args.phase, args.router, experts)
    baseline = BaselineMetrics(
        params_m=args.baseline_params,
        accuracy=args.baseline_accuracy,
        latency_ms=args.baseline_latency_ms,
    )
    kpis = KPIs(epu_target=args.epu_target, cost_reduction_target_pct=args.cost_target_pct)
    runner = BenchmarkRunner(pipeline=pipeline, baseline=baseline, kpis=kpis)

    print("=" * 70)
    print(f"CHEN Benchmark Suite  (Phase {args.phase}, router={args.router})")
    print(f"Baseline: {baseline.params_m}M params, accuracy={baseline.accuracy:.3f}, "
          f"latency={baseline.latency_ms:.0f}ms")
    print("=" * 70)

    all_met = True
    for name, task in TASK_REGISTRY.items():
        result = runner.run_task(task, max_tokens=args.max_tokens)
        print()
        print(result.summary())
        if result.kpi_report is not None:
            verdict = (
                "PASS" if (
                    result.kpi_report.cost_target_met
                    and result.kpi_report.epu_target_met
                    and result.kpi_report.latency_target_met
                ) else "FAIL"
            )
            if verdict == "FAIL":
                all_met = False
            print(f"\n  >>> VERDICT: {verdict}")

    print()
    print("=" * 70)
    print(f"Overall: {'ALL TASKS MET TARGETS' if all_met else 'SOME TARGETS NOT MET'}")
    print("=" * 70)
    print()
    print("Note: Numbers above are from the MockBackend and are useful for")
    print("verifying the harness end-to-end. For real numbers, swap MockBackend")
    print("for HuggingFaceBackend and rerun: see examples/run_phase1.py for the")
    print("pattern.")
    return 0 if all_met else 1


if __name__ == "__main__":
    sys.exit(main())
