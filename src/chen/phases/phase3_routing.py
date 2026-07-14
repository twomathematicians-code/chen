"""Phase 3: Dynamic routing.

The router decides which subset of experts to wake up per prompt.
Pipeline runs only the chosen subset — so a "write a haiku" prompt
wakes only the 3B Synthesizer (cost: ~$0.0001), while a "debug this
segfault" prompt wakes the Analyst + Reasoner + Synthesizer
(cost: ~$0.0005).

Variable isolated: Does waking only the needed experts cut cost without
hurting quality?

This pipeline supports both text handoff (Phase 1-style) and KV-cache
handoff (Phase 2-style) for the selected experts — set ``handoff`` to
``"text"`` or ``"kv_cache"``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from chen.core.kv_cache import KVCache
from chen.core.memory import MemoryEntry
from chen.core.pipeline import (
    AggregateMetrics,
    Pipeline,
    PipelineResult,
)
from chen.core.router import Router

logger = logging.getLogger(__name__)


@dataclass
class RoutingPipeline(Pipeline):
    """Phase 3: dynamic routing.

    Attributes:
        router: The router instance to use. Must implement the
            :class:`~chen.core.router.Router` protocol.
        handoff: "text" (Phase 1-style) or "kv_cache" (Phase 2-style).
        max_tokens_per_expert: Token budget per expert.
        write_intermediate_to_memory: If True, write each expert's
            output to shared memory.
        memory_retrieve_k: Number of memory entries to retrieve for the
            first expert only.
        fallback_to_text: Used only when ``handoff="kv_cache"`` — fall
            back to text on KV transfer failure.
    """

    router: Router = field(default=None)  # type: ignore[assignment]
    handoff: Literal["text", "kv_cache"] = "kv_cache"
    max_tokens_per_expert: int = 256
    write_intermediate_to_memory: bool = True
    memory_retrieve_k: int = 2
    fallback_to_text: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.router is None:
            raise ValueError(
                "RoutingPipeline requires a `router`. "
                "Use Router.from_experts(experts) to build one."
            )

    def run(self, prompt: str, **kwargs: Any) -> PipelineResult:
        """Run the routing pipeline.

        Step 1: Route the prompt to get an ordered list of expert names.
        Step 2: Invoke the selected experts in order, using either text
        or KV-cache handoff.
        """
        selected_names = self.router.route(prompt, self.experts)
        if not selected_names:
            logger.warning("Router returned no experts for prompt. Using fallback.")
            # Pick the first synthesizer, else first expert.
            for e in self.experts:
                if e.role.value == "synthesizer":
                    selected_names = [e.name]
                    break
            if not selected_names:
                selected_names = [self.experts[0].name]

        metrics = AggregateMetrics(cost=self.cost)
        per_expert: list = []
        current_cache: KVCache | None = None
        current_text = prompt
        text_mode = self.handoff == "text"

        if self.write_intermediate_to_memory:
            self.memory.write(MemoryEntry(text=prompt, role="user", expert_name="user"))

        for i, name in enumerate(selected_names):
            expert = self.get_expert(name)
            is_last = i == len(selected_names) - 1
            max_tokens = kwargs.pop("max_tokens", self.max_tokens_per_expert)

            if i == 0:
                # First expert: encode prompt (KV mode) or just pass text.
                if not text_mode and not expert.capabilities.supports_kv_cache:
                    # Backend doesn't support KV — start in text mode.
                    text_mode = True
                    current_cache = None
                if not text_mode and expert.capabilities.supports_kv_cache:
                    try:
                        # Augment with memory before encoding.
                        augmented = prompt
                        if self.memory_retrieve_k > 0:
                            entries = self.memory.retrieve(prompt, k=self.memory_retrieve_k)
                            if entries:
                                ctx = "\n".join(f"- [{e.role}] {e.text}" for e in entries)
                                augmented = f"Context from shared memory:\n{ctx}\n\n---\n\n{prompt}"
                                metrics.memory_entries_read += self.memory_retrieve_k
                        current_cache = expert.backend.encode(augmented)
                        output, out_cache, m = expert.invoke(
                            cache=current_cache, max_tokens=max_tokens
                        )
                    except Exception as e:
                        logger.warning(
                            "Expert %r encode failed (%s). Falling back to text.",
                            name,
                            e,
                        )
                        text_mode = True
                        current_cache = None
                        output, out_cache, m = self._invoke_expert(
                            expert=expert,
                            prompt=prompt,
                            max_tokens=max_tokens,
                            memory_retrieve_k=self.memory_retrieve_k,
                        )
                else:
                    output, out_cache, m = self._invoke_expert(
                        expert=expert,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        memory_retrieve_k=self.memory_retrieve_k,
                    )
            else:
                if text_mode or current_cache is None:
                    output, out_cache, m = self._invoke_expert(
                        expert=expert,
                        prompt=current_text,
                        max_tokens=max_tokens,
                        memory_retrieve_k=0,
                    )
                else:
                    output, out_cache, m = expert.invoke(cache=current_cache, max_tokens=max_tokens)
                    if not m.cache_transfer_succeeded:
                        logger.warning(
                            "KV transfer to expert %r failed. Falling back to text.",
                            name,
                        )
                        if not self.fallback_to_text:
                            raise
                        output, out_cache, m = self._invoke_expert(
                            expert=expert,
                            prompt=current_text,
                            max_tokens=max_tokens,
                            memory_retrieve_k=0,
                        )
                        text_mode = True

            metrics.add(m)
            per_expert.append(m)

            if self.write_intermediate_to_memory:
                src = current_cache.source_text if current_cache is not None else current_text
                self.memory.write(
                    MemoryEntry(
                        text=src,
                        role=expert.role.value,
                        expert_name=expert.name,
                        confidence=0.9 if not is_last else 1.0,
                    )
                )
                metrics.memory_entries_written += 1

            current_cache = out_cache
            current_text = output

        metrics.finalize(distinct_experts=[self.get_expert(n) for n in selected_names])
        return PipelineResult(
            output=current_text,
            metrics=metrics,
            per_expert=per_expert,
            selected_experts=list(selected_names),
            metadata={
                "phase": 3,
                "handoff": self.handoff,
                "router": getattr(self.router, "name", "unknown"),
                "text_mode_fallback": text_mode,
                "selected_experts": list(selected_names),
            },
        )
