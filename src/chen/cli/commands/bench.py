"""``chen bench`` — run the benchmark suite and print KPI reports."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from chen.backends.mock import MockBackend
from chen.benchmarks import TASK_REGISTRY, BenchmarkRunner
from chen.benchmarks.kpis import BaselineMetrics, KPIs
from chen.core.expert import Expert, ExpertRole
from chen.core.router import CosineRouter, HybridRouter, LogisticRouter
from chen.observability.logging import configure_logging, get_logger

console = Console()


def _build_experts() -> list[Expert]:
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


def _make_pipeline(phase: int, router_kind: str, experts: list[Expert], max_tokens: int):
    if phase == 1:
        from chen.phases.phase1_cascade import CascadePipeline

        return CascadePipeline(
            experts=experts,
            sequence=["analyst", "synthesizer"],
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
    elif phase == 2:
        from chen.phases.phase2_kv_pass import KVPassPipeline

        return KVPassPipeline(
            experts=experts,
            sequence=["analyst", "synthesizer"],
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
    elif phase == 3:
        from chen.phases.phase3_routing import RoutingPipeline

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
            handoff="text",
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
    raise ValueError(f"Unknown phase: {phase}")


def bench_command(
    phase: int,
    router: str,
    max_tokens: int,
    task: str,
    baseline_params: int,
) -> None:
    """Run the benchmark suite and print KPI reports."""
    configure_logging(level="WARNING")
    log = get_logger(__name__)
    log.info("chen.bench.start", phase=phase, router=router)

    console.print(
        f"\n[bold blue]CHEN Benchmark Suite[/bold blue]  "
        f"phase={phase}  router={router}  baseline={baseline_params}M params\n"
    )

    experts = _build_experts()
    pipeline = _make_pipeline(phase, router, experts, max_tokens)
    baseline = BaselineMetrics(params_m=baseline_params, accuracy=0.5, latency_ms=1500.0)
    kpis = KPIs(epu_target=1.0, cost_reduction_target_pct=10.0)
    runner = BenchmarkRunner(pipeline=pipeline, baseline=baseline, kpis=kpis)

    tasks = (
        [(name, TASK_REGISTRY[name]) for name in [task] if name in TASK_REGISTRY]
        if task != "all"
        else list(TASK_REGISTRY.items())
    )

    summary_table = Table(title="Summary", border_style="dim")
    summary_table.add_column("task", style="bold")
    summary_table.add_column("samples", justify="right")
    summary_table.add_column("accuracy", justify="right")
    summary_table.add_column("tokens", justify="right")
    summary_table.add_column("cost ($)", justify="right")
    summary_table.add_column("EPU", justify="right")
    summary_table.add_column("cost/1M ($)", justify="right")
    summary_table.add_column("verdict")

    all_met = True
    for name, t in tasks:
        result = runner.run_task(t, max_tokens=max_tokens)
        verdict = (
            "[green]PASS[/green]"
            if result.kpi_report
            and result.kpi_report.cost_target_met
            and result.kpi_report.epu_target_met
            and result.kpi_report.latency_target_met
            else "[red]FAIL[/red]"
        )
        if "FAIL" in verdict:
            all_met = False
        summary_table.add_row(
            name,
            str(len(result.per_sample)),
            f"{result.avg_accuracy:.3f}",
            str(result.total_tokens),
            f"{result.total_cost_usd:.6f}",
            f"{result.kpi_report.epu:.2f}" if result.kpi_report else "—",
            f"{result.kpi_report.chen_cost_per_1m:.4f}" if result.kpi_report else "—",
            verdict,
        )
    console.print(summary_table)

    console.print(
        f"\n[bold]Overall:[/bold] "
        f"{'[green]ALL TARGETS MET[/green]' if all_met else '[red]SOME TARGETS NOT MET[/red]'}"
    )
    console.print(
        "[dim]Note: Numbers above are from the MockBackend. For real numbers, "
        "swap MockBackend for HuggingFaceBackend in the source.[/dim]\n"
    )
