"""Benchmark harness: KPIs, sample tasks, and a runner.

The benchmark layer measures CHEN against the three KPIs described in
the README:

1. **Effective Parameter Utilization (EPU)** — did our small-model swarm
   match the quality of a much larger monolith?
2. **Cost per 1M Tokens** — should be 80–95% lower than the baseline.
3. **Latency-to-Accuracy Ratio** — small models are fast; CHEN should
   keep the low latency while showing an accuracy jump.

The harness ships with sample tasks (math, code, QA, summarization) and
deterministic graders so the tests run without external dependencies.
"""

from __future__ import annotations

from chen.benchmarks.kpis import KPIReport, KPIs
from chen.benchmarks.runner import BenchmarkRunner, BenchmarkRunResult
from chen.benchmarks.tasks import (
    TASK_REGISTRY,
    BenchmarkTask,
    register_task,
)

__all__ = [
    "KPIs",
    "KPIReport",
    "BenchmarkRunner",
    "BenchmarkRunResult",
    "BenchmarkTask",
    "TASK_REGISTRY",
    "register_task",
]
