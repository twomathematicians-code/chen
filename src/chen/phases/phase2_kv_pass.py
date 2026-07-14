"""Phase 2: Latent state (KV-cache) passing.

The key differentiator of CHEN. Instead of passing *text* between
experts, we pass the *KV-cache* — the mathematical memory of the prompt.
This is supposed to preserve more nuance than re-encoding text.

Variable isolated: Does passing KV-caches preserve more nuance than text?

The pipeline runs the same sequence of experts as Phase 1, but each
expert receives the previous expert's KV-cache (via the backend's
``transfer_cache``). If a transfer fails
(:class:`~chen.core.kv_cache.IncompatibleCacheError`), the pipeline logs
a warning and falls back to text handoff for that step, so the pipeline
is robust to backends that don't support cross-family transfer.

The result's ``metrics.latent_nuance_score`` reports the fraction of
KV transfers that succeeded. A real Phase 2 experiment should also
compute a KL-divergence-based nuance probe (see ``ARCHITECTURE.md`` §5);
that's left to the benchmark harness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from chen.core.kv_cache import KVCache
from chen.core.memory import MemoryEntry
from chen.core.pipeline import (
    AggregateMetrics,
    Pipeline,
    PipelineResult,
)

logger = logging.getLogger(__name__)


@dataclass
class KVPassPipeline(Pipeline):
    """Phase 2: KV-cache handoff.

    Attributes:
        sequence: Ordered list of expert names. The first expert encodes
            the prompt to a KV-cache; each subsequent expert receives the
            previous expert's output KV-cache (transferred via the
            backend's ``transfer_cache``).
        max_tokens_per_expert: Token budget per expert.
        write_intermediate_to_memory: If True, write each expert's source
            text (the text the KV-cache was encoded from) to shared memory.
        memory_retrieve_k: Number of memory entries to retrieve and
            prepend to the *first* expert's prompt only. Later experts
            get their context via the KV-cache, not via prepended memory.
        fallback_to_text: If True (default), fall back to text handoff
            when a KV transfer fails. If False, raise the error.
    """

    sequence: list[str] = field(default_factory=list)
    max_tokens_per_expert: int = 256
    write_intermediate_to_memory: bool = True
    memory_retrieve_k: int = 2
    fallback_to_text: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.sequence:
            self.sequence = [e.name for e in self.experts]
        missing = [n for n in self.sequence if n not in self.experts_by_name]
        if missing:
            raise ValueError(
                f"KVPassPipeline.sequence references unknown experts: {missing}. "
                f"Available: {list(self.experts_by_name)}"
            )
        # Verify all experts in the sequence support KV-cache.
        for name in self.sequence:
            e = self.experts_by_name[name]
            if not e.capabilities.supports_kv_cache:
                logger.warning(
                    "Expert %r (backend %r) does not support KV-cache. "
                    "Phase 2 will fall back to text handoff for this expert.",
                    name,
                    type(e.backend).__name__,
                )

    def run(self, prompt: str, **kwargs: Any) -> PipelineResult:
        """Run the KV-cache-pass pipeline.

        Step 1: Expert 1 encodes the prompt to a KV-cache and decodes
        its output from that cache. Step 2: Expert 2 receives Expert 1's
        output cache (transferred), and decodes from it. And so on.
        """
        metrics = AggregateMetrics(cost=self.cost)
        per_expert: list = []
        current_cache: KVCache | None = None
        current_text = prompt
        text_mode = False  # set True if any step falls back to text

        if self.write_intermediate_to_memory:
            self.memory.write(MemoryEntry(text=prompt, role="user", expert_name="user"))

        for i, name in enumerate(self.sequence):
            expert = self.get_expert(name)
            is_last = i == len(self.sequence) - 1
            max_tokens = kwargs.pop("max_tokens", self.max_tokens_per_expert)

            if i == 0:
                # First expert: encode the prompt.
                if not expert.capabilities.supports_kv_cache:
                    # Backend can't produce a KV-cache — start in text mode.
                    text_mode = True
                    current_cache = None
                else:
                    try:
                        current_cache = expert.backend.encode(prompt)
                    except Exception as e:
                        logger.warning(
                            "Expert %r encode failed (%s). Falling back to text.",
                            name,
                            e,
                        )
                        text_mode = True
                        current_cache = None

                # Augment the prompt with memory for the first expert only.
                augmented = prompt
                if self.memory_retrieve_k > 0 and not text_mode:
                    entries = self.memory.retrieve(prompt, k=self.memory_retrieve_k)
                    if entries:
                        ctx = "\n".join(f"- [{e.role}] {e.text}" for e in entries)
                        augmented = f"Context from shared memory:\n{ctx}\n\n---\n\n{prompt}"
                        metrics.memory_entries_read += self.memory_retrieve_k
                    # Re-encode with the augmented prompt if memory was added.
                    if augmented != prompt and not text_mode:
                        try:
                            current_cache = expert.backend.encode(augmented)
                        except Exception:
                            pass  # use the unaugmented cache

                output, out_cache, m = expert.invoke(
                    prompt=augmented if text_mode else None,
                    cache=current_cache if not text_mode else None,
                    max_tokens=max_tokens,
                )
            else:
                # Subsequent expert: receive the previous expert's cache.
                if text_mode or current_cache is None:
                    output, out_cache, m = self._invoke_expert(
                        expert=expert,
                        prompt=current_text,
                        max_tokens=max_tokens,
                        memory_retrieve_k=0,  # later experts get context via cache
                    )
                else:
                    output, out_cache, m = expert.invoke(
                        cache=current_cache,
                        max_tokens=max_tokens,
                    )
                    if not m.cache_transfer_succeeded:
                        logger.warning(
                            "KV transfer to expert %r failed. Falling back to text.",
                            name,
                        )
                        if not self.fallback_to_text:
                            raise
                        # Re-invoke with text.
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
                # Write the source_text of the cache (or the prompt in text mode).
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

        metrics.finalize(distinct_experts=[self.get_expert(n) for n in self.sequence])
        return PipelineResult(
            output=current_text,
            metrics=metrics,
            per_expert=per_expert,
            selected_experts=list(self.sequence),
            metadata={
                "phase": 2,
                "handoff": "kv_cache",
                "text_mode_fallback": text_mode,
                "sequence": list(self.sequence),
            },
        )
