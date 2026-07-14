"""Phase 1: Static cascading (text handoff).

The simplest pipeline: hard-code a sequence of experts, pass each
expert's *text* output as the next expert's prompt. No KV-cache, no
routing. The point is to verify that chaining small models at all works
and to establish a cost/quality baseline that Phase 2 and Phase 3 build on.

Variable isolated: Does chaining small models at all work?

See ``ARCHITECTURE.md`` §9 for the experimental design.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from chen.core.memory import MemoryEntry
from chen.core.pipeline import (
    AggregateMetrics,
    Pipeline,
    PipelineResult,
)

logger = logging.getLogger(__name__)


@dataclass
class CascadePipeline(Pipeline):
    """Phase 1: static cascade.

    Attributes:
        sequence: Ordered list of expert names to invoke. Each expert's
            text output is fed as the next expert's prompt.
        max_tokens_per_expert: Token budget per expert.
        write_intermediate_to_memory: If True, each expert's output is
            written to shared memory under its role. This lets later
            experts retrieve earlier outputs as structured entries
            (Phase 1.5 — half-step toward shared memory).
        memory_retrieve_k: Number of memory entries to retrieve and
            prepend to each expert's prompt. 0 disables memory retrieval.
    """

    sequence: list[str] = field(default_factory=list)
    max_tokens_per_expert: int = 256
    write_intermediate_to_memory: bool = True
    memory_retrieve_k: int = 2

    def __post_init__(self) -> None:
        # Call parent's __post_init__ to set up experts_by_name, memory, etc.
        super().__post_init__()
        if not self.sequence:
            # Default: invoke experts in the order they were passed in.
            self.sequence = [e.name for e in self.experts]
        missing = [n for n in self.sequence if n not in self.experts_by_name]
        if missing:
            raise ValueError(
                f"CascadePipeline.sequence references unknown experts: {missing}. "
                f"Available: {list(self.experts_by_name)}"
            )

    def run(self, prompt: str, **kwargs: Any) -> PipelineResult:
        """Run the static cascade.

        Each expert in ``self.sequence`` is invoked in order. The first
        expert receives the user's prompt; each subsequent expert receives
        the previous expert's text output as its prompt.
        """
        metrics = AggregateMetrics(cost=self.cost)
        per_expert: list = []
        current_text = prompt

        # Optionally write the user's prompt to memory as a "user" entry.
        if self.write_intermediate_to_memory:
            self.memory.write(MemoryEntry(text=prompt, role="user", expert_name="user"))

        for i, name in enumerate(self.sequence):
            expert = self.get_expert(name)
            is_last = i == len(self.sequence) - 1
            max_tokens = kwargs.pop("max_tokens", self.max_tokens_per_expert)

            output, _cache, m = self._invoke_expert(
                expert=expert,
                prompt=current_text,
                max_tokens=max_tokens,
                memory_retrieve_k=self.memory_retrieve_k,
            )
            metrics.add(m)
            per_expert.append(m)

            if self.write_intermediate_to_memory:
                self.memory.write(
                    MemoryEntry(
                        text=output,
                        role=expert.role.value,
                        expert_name=expert.name,
                        confidence=0.9 if not is_last else 1.0,
                    )
                )
                metrics.memory_entries_written += 1
            if self.memory_retrieve_k > 0:
                metrics.memory_entries_read += self.memory_retrieve_k

            current_text = output

        metrics.finalize(distinct_experts=[self.get_expert(n) for n in self.sequence])
        return PipelineResult(
            output=current_text,
            metrics=metrics,
            per_expert=per_expert,
            selected_experts=list(self.sequence),
            metadata={
                "phase": 1,
                "handoff": "text",
                "sequence": list(self.sequence),
            },
        )
