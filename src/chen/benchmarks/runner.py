"""Benchmark runner.

Runs a :class:`~chen.benchmarks.tasks.BenchmarkTask` against a CHEN
:class:`~chen.core.pipeline.Pipeline` and computes the
:class:`~chen.benchmarks.kpis.KPIReport` against a baseline monolith.

Usage::

    from chen.benchmarks import BenchmarkRunner, TASK_REGISTRY
    from chen.phases.phase1_cascade import CascadePipeline
    from chen.backends.mock import MockBackend
    from chen.core.expert import Expert, ExpertRole

    experts = [Expert(name="e", role=ExpertRole.GENERALIST, backend=MockBackend())]
    pipe = CascadePipeline(experts=experts)
    runner = BenchmarkRunner(pipeline=pipe)
    report = runner.run_task(TASK_REGISTRY["math_arithmetic"])
    print(report.kpi_report.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from chen.benchmarks.kpis import BaselineMetrics, KPIReport, KPIs
from chen.benchmarks.tasks import TASK_REGISTRY, BenchmarkTask
from chen.core.pipeline import Pipeline, PipelineResult

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRunResult:
    """Result of running one task through the benchmark harness.

    Attributes:
        task_name: Name of the task.
        per_sample: List of (prompt, expected, output, score, latency_ms,
            pipeline_result) tuples, one per sample.
        avg_accuracy: Average score across samples (0..1).
        total_tokens: Total tokens (in + out) across all samples.
        total_cost_usd: Total USD cost across all samples.
        total_latency_ms: Total wall-clock latency across all samples.
        params_invoked_m: Sum of params_m across all expert invocations.
        distinct_params_m: Sum of params_m of distinct experts used
            (taken from the last run; assumes the pipeline doesn't
            change its expert pool across samples).
        kpi_report: KPI report against the baseline.
    """

    task_name: str
    per_sample: list[tuple[str, str, str, float, float, PipelineResult]] = field(
        default_factory=list
    )
    avg_accuracy: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    params_invoked_m: int = 0
    distinct_params_m: int = 0
    kpi_report: KPIReport | None = None

    def summary(self) -> str:
        """Multi-line human-readable summary."""
        lines = [
            f"Benchmark: {self.task_name}",
            f"  samples: {len(self.per_sample)}",
            f"  avg accuracy: {self.avg_accuracy:.3f}",
            f"  total tokens: {self.total_tokens}",
            f"  total cost: ${self.total_cost_usd:.6f}",
            f"  total latency: {self.total_latency_ms:.1f} ms",
            f"  params invoked: {self.params_invoked_m}M (distinct: {self.distinct_params_m}M)",
        ]
        if self.kpi_report is not None:
            lines.append("")
            lines.append(self.kpi_report.summary())
        return "\n".join(lines)


@dataclass
class BenchmarkRunner:
    """Run benchmark tasks against a CHEN pipeline and compute KPIs.

    Attributes:
        pipeline: The CHEN pipeline to benchmark.
        baseline: Baseline monolith metrics to compare against.
        kpis: KPI computer (with targets).
    """

    pipeline: Pipeline
    baseline: BaselineMetrics = field(default_factory=BaselineMetrics)
    kpis: KPIs = field(default_factory=KPIs)

    def run_task(
        self,
        task: BenchmarkTask | str,
        max_tokens: int = 256,
        **pipeline_kwargs: Any,
    ) -> BenchmarkRunResult:
        """Run one task through the pipeline and compute KPIs.

        Args:
            task: A :class:`BenchmarkTask` instance or its name in the registry.
            max_tokens: Per-expert token budget.
            **pipeline_kwargs: Forwarded to ``pipeline.run()``.
        """
        if isinstance(task, str):
            if task not in TASK_REGISTRY:
                raise KeyError(f"Unknown task '{task}'. Registered: {list(TASK_REGISTRY)}")
            task = TASK_REGISTRY[task]

        result = BenchmarkRunResult(task_name=task.name)
        last_pipeline_result: PipelineResult | None = None

        for prompt, expected in task.samples:
            pr = self.pipeline.run(prompt, max_tokens=max_tokens, **pipeline_kwargs)
            score = task.grader(pr.output, expected)
            result.per_sample.append(
                (prompt, expected, pr.output, score, pr.metrics.total_latency_ms, pr)
            )
            result.avg_accuracy += score
            result.total_tokens += pr.metrics.total_tokens
            result.total_cost_usd += pr.metrics.total_cost_usd
            result.total_latency_ms += pr.metrics.total_latency_ms
            result.params_invoked_m += pr.metrics.total_params_invoked_m
            last_pipeline_result = pr

        n = len(task.samples)
        if n > 0:
            result.avg_accuracy /= n

        if last_pipeline_result is not None:
            result.distinct_params_m = last_pipeline_result.metrics.distinct_params_invoked_m

        result.kpi_report = self.kpis.compute(
            chen_total_tokens=result.total_tokens,
            chen_total_cost_usd=result.total_cost_usd,
            chen_latency_ms=result.total_latency_ms,
            chen_accuracy=result.avg_accuracy,
            chen_params_invoked_m=result.params_invoked_m,
            chen_distinct_params_m=result.distinct_params_m,
            baseline=self.baseline,
            n_queries=n,
            task_name=task.name,
        )
        return result

    def run_all(self, max_tokens: int = 256, **pipeline_kwargs: Any) -> list[BenchmarkRunResult]:
        """Run every registered task and return their results."""
        return [
            self.run_task(name, max_tokens=max_tokens, **pipeline_kwargs) for name in TASK_REGISTRY
        ]
