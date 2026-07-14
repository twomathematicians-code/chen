"""Pipeline base class and result types.

A :class:`Pipeline` orchestrates a sequence of experts. The three
concrete pipelines — :class:`~chen.phases.phase1_cascade.CascadePipeline`,
:class:`~chen.phases.phase2_kv_pass.KVPassPipeline`, and
:class:`~chen.phases.phase3_routing.RoutingPipeline` — implement the
three experimental phases of CHEN, but they all share the result types
defined here.

Key types:

* :class:`ExpertMetrics`     — per-invocation metrics (in expert.py).
* :class:`AggregateMetrics`  — pipeline-level aggregates.
* :class:`PipelineResult`    — final return type with output + metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from chen.core.config import CostModel, Settings
from chen.core.expert import Expert, ExpertMetrics
from chen.core.memory import InMemoryMemory, Memory

logger = logging.getLogger(__name__)


@dataclass
class AggregateMetrics:
    """Pipeline-level aggregate metrics across all experts invoked."""

    expert_metrics: list[ExpertMetrics] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0
    total_params_invoked_m: int = 0  # sum of params_m across invocations
    distinct_params_invoked_m: int = 0  # sum of params_m of distinct experts used
    total_cost_usd: float = 0.0
    kv_cache_transfers: int = 0
    kv_cache_failures: int = 0
    latent_nuance_score: float = 0.0  # Phase 2+ only; ratio of nuance preserved
    memory_entries_read: int = 0
    memory_entries_written: int = 0
    cost: CostModel = field(default_factory=CostModel)

    def add(self, m: ExpertMetrics) -> None:
        self.expert_metrics.append(m)
        self.total_input_tokens += m.input_tokens
        self.total_output_tokens += m.output_tokens
        self.total_latency_ms += m.latency_ms + m.cache_transfer_ms
        self.total_params_invoked_m += m.params_m
        if m.used_kv_cache:
            self.kv_cache_transfers += 1
            if not m.cache_transfer_succeeded:
                self.kv_cache_failures += 1
        self.total_cost_usd += self.cost.cost_for(m.input_tokens, m.output_tokens, m.params_m)

    def finalize(self, distinct_experts: list[Expert]) -> None:
        """Recompute derived fields after all experts have been added."""
        self.distinct_params_invoked_m = sum(e.params_m for e in distinct_experts)
        # Latent nuance score: ratio of successful KV transfers to total
        # KV-capable experts in the chain. Heuristic; real Phase 2 should
        # override this with a KL-divergence-based probe.
        n_kv_used = sum(1 for m in self.expert_metrics if m.used_kv_cache)
        n_kv_ok = sum(
            1 for m in self.expert_metrics if m.used_kv_cache and m.cache_transfer_succeeded
        )
        if n_kv_used > 0:
            self.latent_nuance_score = n_kv_ok / n_kv_used

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def cost_per_1m_tokens(self) -> float:
        """USD cost per 1M total tokens."""
        if self.total_tokens == 0:
            return 0.0
        return self.total_cost_usd / self.total_tokens * 1_000_000

    @property
    def epu(self) -> float:
        """Effective Parameter Utilization (heuristic).

        Defined as ``distinct_params_invoked_m / total_params_invoked_m``.
        A value close to 1.0 means every distinct expert was invoked
        exactly once (no re-invocation); a value < 1.0 means experts were
        re-invoked, suggesting routing inefficiency.

        Note: this is a *capacity utilization* EPU, not the quality-matching
        EPU described in the README. The latter requires a baseline
        monolith quality comparison and is computed by the KPI harness.
        """
        if self.total_params_invoked_m == 0:
            return 0.0
        return self.distinct_params_invoked_m / self.total_params_invoked_m

    def summary(self) -> str:
        """Human-readable one-liner."""
        return (
            f"experts={len(self.expert_metrics)}, "
            f"tokens={self.total_tokens}, "
            f"latency={self.total_latency_ms:.1f}ms, "
            f"cost=${self.total_cost_usd:.6f}, "
            f"cost/1M=${self.cost_per_1m_tokens:.4f}, "
            f"EPU={self.epu:.3f}, "
            f"kv_xfers={self.kv_cache_transfers}/{self.kv_cache_transfers + self.kv_cache_failures}, "
            f"nuance={self.latent_nuance_score:.2f}"
        )


@dataclass
class PipelineResult:
    """The return type of every ``pipeline.run()`` call.

    Attributes:
        output: The final text output (from the last expert).
        metrics: Aggregate metrics across all experts.
        per_expert: List of per-expert metrics, in invocation order.
        selected_experts: Names of experts actually invoked (Phase 3 may
            differ from the configured sequence).
        metadata: Free-form pipeline-specific metadata.
    """

    output: str
    metrics: AggregateMetrics
    per_expert: list[ExpertMetrics] = field(default_factory=list)
    selected_experts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Pipeline:
    """Base class for all CHEN pipelines.

    Subclasses implement :meth:`run` to define the orchestration logic.
    The base class provides shared utilities: memory access, telemetry,
    and a ``_invoke_expert`` helper that records metrics.

    This base class is a dataclass so that subclasses (which are also
    dataclasses) can extend its fields via dataclass inheritance. The
    ``__post_init__`` validates the experts list and builds the
    ``experts_by_name`` lookup.
    """

    experts: list[Expert] = field(default_factory=list)
    memory: Memory | None = None
    settings: Settings | None = None
    cost: CostModel | None = None

    # Derived (set in __post_init__, not constructor args)
    experts_by_name: dict[str, Expert] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.experts:
            raise ValueError("Pipeline requires at least one expert.")
        self.experts_by_name = {e.name: e for e in self.experts}
        if self.memory is None:
            self.memory = InMemoryMemory()
        if self.settings is None:
            self.settings = Settings()
        if self.cost is None:
            self.cost = self.settings.cost

    def get_expert(self, name: str) -> Expert:
        if name not in self.experts_by_name:
            raise KeyError(f"Expert '{name}' not found. Available: {list(self.experts_by_name)}")
        return self.experts_by_name[name]

    def _invoke_expert(
        self,
        expert: Expert,
        prompt: str | None = None,
        cache: Any | None = None,
        max_tokens: int = 256,
        memory_retrieve_k: int = 0,
        **kwargs: Any,
    ) -> tuple[str, Any, ExpertMetrics]:
        """Invoke one expert with memory augmentation and return metrics.

        If ``memory_retrieve_k > 0``, retrieve up to that many memory
        entries relevant to the prompt and prepend them to the prompt
        before invocation.
        """
        # Memory augmentation (only for text path).
        augmented_prompt = prompt
        if memory_retrieve_k > 0 and prompt is not None:
            entries = self.memory.retrieve(prompt, k=memory_retrieve_k)
            if entries:
                ctx = "\n".join(f"- [{e.role}] {e.text}" for e in entries)
                augmented_prompt = f"Context from shared memory:\n{ctx}\n\n---\n\n{prompt}"

        output, out_cache, metrics = expert.invoke(
            prompt=augmented_prompt,
            cache=cache,
            max_tokens=max_tokens,
            **kwargs,
        )
        return output, out_cache, metrics

    def run(self, prompt: str, **kwargs: Any) -> PipelineResult:  # noqa: D401
        """Run the pipeline. Subclasses must implement."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.run() must be implemented by subclasses."
        )
