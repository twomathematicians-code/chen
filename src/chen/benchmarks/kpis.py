"""Key Performance Indicators for CHEN.

Three KPIs are computed against a baseline monolith:

1. **Effective Parameter Utilization (EPU)** — ratio of the baseline
   monolith's parameter count to the sum of distinct expert parameter
   counts invoked. If we used 9B of parameters total (three 3B experts)
   and matched the quality of a 30B monolith, EPU is 30/9 = 3.33.
2. **Cost per 1M Tokens** — USD per 1M total tokens. Should be 80–95%
   lower than baseline for CHEN to be worth adopting.
3. **Latency-to-Accuracy Ratio** — accuracy / latency_ms × 1000. Higher
   is better. Small-model swarms should beat the monolith on simple
   queries and be competitive on hard ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chen.core.config import CostModel


@dataclass
class BaselineMetrics:
    """Metrics for the baseline monolith we compare against.

    Attributes:
        params_m: Monolith parameter count in millions (e.g. 70_000 for 70B).
        input_tokens: Per-query average input tokens.
        output_tokens: Per-query average output tokens.
        latency_ms: Per-query average latency.
        accuracy: Per-query average accuracy on the benchmark task (0..1).
    """

    params_m: int = 70_000
    input_tokens: int = 200
    output_tokens: int = 300
    latency_ms: float = 1500.0
    accuracy: float = 0.85


@dataclass
class KPIReport:
    """Bundle of KPIs for a single benchmark run.

    All fields are computed by :meth:`KPIs.compute`.
    """

    # CHEN side
    chen_cost_per_1m: float
    chen_latency_ms: float
    chen_accuracy: float
    chen_params_invoked_m: int
    chen_distinct_params_m: int

    # Baseline side
    baseline_cost_per_1m: float
    baseline_latency_ms: float
    baseline_accuracy: float
    baseline_params_m: int

    # Derived KPIs
    epu: float
    cost_reduction_pct: float
    latency_to_accuracy: float
    baseline_latency_to_accuracy: float

    # Verdict
    cost_target_met: bool
    epu_target_met: bool
    latency_target_met: bool

    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Multi-line human-readable summary."""
        lines = [
            "CHEN KPI Report",
            "─────────────────────────────────────────────────────────────",
            f"  EPU (effective param utilization): {self.epu:.2f}x",
            f"    target: ≥ 3.0x       met: {self.epu_target_met}",
            f"  Cost per 1M tokens:    CHEN ${self.chen_cost_per_1m:.4f}  "
            f"vs  baseline ${self.baseline_cost_per_1m:.4f}",
            f"    reduction: {self.cost_reduction_pct:.1f}%   "
            f"target: ≥ 80%   met: {self.cost_target_met}",
            f"  Latency-to-accuracy:   CHEN {self.latency_to_accuracy:.4f}  "
            f"vs  baseline {self.baseline_latency_to_accuracy:.4f}",
            f"    target: CHEN ≥ baseline   met: {self.latency_target_met}",
            "─────────────────────────────────────────────────────────────",
            f"  CHEN: {self.chen_params_invoked_m}M params invoked "
            f"({self.chen_distinct_params_m}M distinct), "
            f"{self.chen_latency_ms:.0f} ms, accuracy={self.chen_accuracy:.3f}",
            f"  Base: {self.baseline_params_m}M params, "
            f"{self.baseline_latency_ms:.0f} ms, accuracy={self.baseline_accuracy:.3f}",
        ]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        """Flat dict for downstream serialization."""
        return {
            "epu": self.epu,
            "chen_cost_per_1m": self.chen_cost_per_1m,
            "baseline_cost_per_1m": self.baseline_cost_per_1m,
            "cost_reduction_pct": self.cost_reduction_pct,
            "chen_latency_ms": self.chen_latency_ms,
            "baseline_latency_ms": self.baseline_latency_ms,
            "chen_accuracy": self.chen_accuracy,
            "baseline_accuracy": self.baseline_accuracy,
            "latency_to_accuracy": self.latency_to_accuracy,
            "baseline_latency_to_accuracy": self.baseline_latency_to_accuracy,
            "chen_params_invoked_m": self.chen_params_invoked_m,
            "chen_distinct_params_m": self.chen_distinct_params_m,
            "baseline_params_m": self.baseline_params_m,
            "cost_target_met": self.cost_target_met,
            "epu_target_met": self.epu_target_met,
            "latency_target_met": self.latency_target_met,
            **self.extra,
        }


@dataclass
class KPIs:
    """Compute CHEN KPIs against a baseline monolith.

    Attributes:
        cost: Cost model (USD rates).
        epu_target: Minimum EPU for "met" verdict.
        cost_reduction_target_pct: Minimum cost reduction for "met" verdict.
        latency_target_met_if: If "chen_ge_baseline", CHEN's
            latency-to-accuracy must be ≥ baseline's.
    """

    cost: CostModel = field(default_factory=CostModel)
    epu_target: float = 3.0
    cost_reduction_target_pct: float = 80.0
    latency_target_met_if: str = "chen_ge_baseline"

    def compute(
        self,
        chen_total_tokens: int,
        chen_total_cost_usd: float,
        chen_latency_ms: float,
        chen_accuracy: float,
        chen_params_invoked_m: int,
        chen_distinct_params_m: int,
        baseline: BaselineMetrics,
        n_queries: int = 1,
        **extra: Any,
    ) -> KPIReport:
        """Compute the KPI report for one benchmark run.

        Args:
            chen_total_tokens: Total tokens (in + out) across all CHEN
                invocations in this run.
            chen_total_cost_usd: Total USD cost across all CHEN invocations.
            chen_latency_ms: Total wall-clock latency of the CHEN pipeline
                (sum across queries, or average — caller decides; must
                match ``baseline.latency_ms`` which is per-query).
            chen_accuracy: Average accuracy across queries (0..1).
            chen_params_invoked_m: Sum of params_m across all expert
                invocations (counts re-invocations).
            chen_distinct_params_m: Sum of params_m of distinct experts used.
            baseline: Baseline monolith metrics.
            n_queries: Number of queries in this run. Used to normalize
                CHEN's totals to per-query averages.
            **extra: Extra fields to attach to the report.
        """
        # CHEN per-query averages
        q = max(1, n_queries)
        chen_tokens_per_q = chen_total_tokens / q
        chen_cost_per_q = chen_total_cost_usd / q
        chen_cost_per_1m = (
            chen_cost_per_q / chen_tokens_per_q * 1_000_000 if chen_tokens_per_q > 0 else 0.0
        )
        chen_latency_per_q = chen_latency_ms / q

        # Baseline per-1M cost
        baseline_tokens = baseline.input_tokens + baseline.output_tokens
        baseline_cost_per_q = self.cost.cost_for(
            baseline.input_tokens, baseline.output_tokens, baseline.params_m
        )
        baseline_cost_per_1m = (
            baseline_cost_per_q / baseline_tokens * 1_000_000 if baseline_tokens > 0 else 0.0
        )

        # EPU: baseline params / CHEN distinct params
        epu = baseline.params_m / chen_distinct_params_m if chen_distinct_params_m > 0 else 0.0

        # Cost reduction percentage
        cost_reduction_pct = (
            (baseline_cost_per_1m - chen_cost_per_1m) / baseline_cost_per_1m * 100.0
            if baseline_cost_per_1m > 0
            else 0.0
        )

        # Latency-to-accuracy ratio (higher is better)
        lta = chen_accuracy / chen_latency_per_q * 1000.0 if chen_latency_per_q > 0 else 0.0
        baseline_lta = (
            baseline.accuracy / baseline.latency_ms * 1000.0 if baseline.latency_ms > 0 else 0.0
        )

        # Verdicts
        cost_target_met = cost_reduction_pct >= self.cost_reduction_target_pct
        epu_target_met = epu >= self.epu_target
        latency_target_met = lta >= baseline_lta

        return KPIReport(
            chen_cost_per_1m=chen_cost_per_1m,
            chen_latency_ms=chen_latency_per_q,
            chen_accuracy=chen_accuracy,
            chen_params_invoked_m=chen_params_invoked_m,
            chen_distinct_params_m=chen_distinct_params_m,
            baseline_cost_per_1m=baseline_cost_per_1m,
            baseline_latency_ms=baseline.latency_ms,
            baseline_accuracy=baseline.accuracy,
            baseline_params_m=baseline.params_m,
            epu=epu,
            cost_reduction_pct=cost_reduction_pct,
            latency_to_accuracy=lta,
            baseline_latency_to_accuracy=baseline_lta,
            cost_target_met=cost_target_met,
            epu_target_met=epu_target_met,
            latency_target_met=latency_target_met,
            extra=extra,
        )
