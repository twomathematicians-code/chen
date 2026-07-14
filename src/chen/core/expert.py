"""Micro-expert wrapper.

An :class:`Expert` is a ``(name, role, backend)`` triple. The role is a
semantic tag used by the router; the backend is the inference engine
that actually runs the model. The expert is the unit of cost accounting —
every :meth:`invoke` call records tokens, latency, and params loaded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from chen.backends.base import BackendCapabilities, InferenceBackend
from chen.core.kv_cache import KVCache


class ExpertRole(str, Enum):
    """Semantic role tag for an expert. Used by the router for routing decisions.

    The set is intentionally small and stable — new roles should be added
    only when they enable a genuinely new routing pattern. The ``SYNTHESIZER``
    role has special meaning: routers ensure the pipeline always ends with
    a Synthesizer (so the final output is natural-language text, not a
    latent state).
    """

    ANALYST = "analyst"
    REASONER = "reasoner"
    SYNTHESIZER = "synthesizer"
    CODER = "coder"
    ROUTER = "router"
    TRANSLATOR = "translator"
    GENERALIST = "generalist"


@dataclass
class ExpertMetrics:
    """Per-invocation metrics for one expert.

    Recorded by :meth:`Expert.invoke` and aggregated by the pipeline.
    """

    expert_name: str
    role: ExpertRole
    params_m: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cache_transfer_ms: float = 0.0
    used_kv_cache: bool = False
    cache_transfer_succeeded: bool = True

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Expert:
    """A micro-expert: a name, a role, and a backend.

    Attributes:
        name: Short human-readable identifier (e.g. ``"analyst"``).
            Must be unique within a pipeline.
        role: Semantic role tag. Used by the router.
        backend: The inference backend to invoke.
        description: Optional longer description, for logging and UIs.
        tags: Free-form tags for additional routing hints
            (e.g. ``{"math", "python"}``).
    """

    name: str
    role: ExpertRole
    backend: InferenceBackend
    description: str = ""
    tags: set[str] = field(default_factory=set)

    # Cached capabilities
    _capabilities: BackendCapabilities | None = field(default=None, repr=False, compare=False)

    @property
    def capabilities(self) -> BackendCapabilities:
        if self._capabilities is None:
            self._capabilities = self.backend.capabilities
        return self._capabilities

    @property
    def params_m(self) -> int:
        """Model size in millions of parameters. Delegates to backend."""
        # Some backends expose `params_m` as a property, others as
        # `resolved_params_m`. Try both.
        if hasattr(self.backend, "params_m"):
            try:
                return int(self.backend.params_m)
            except (TypeError, ValueError):
                pass
        if hasattr(self.backend, "resolved_params_m"):
            try:
                return int(self.backend.resolved_params_m)
            except (TypeError, ValueError):
                pass
        return 0

    def invoke(
        self,
        prompt: str | None = None,
        cache: KVCache | None = None,
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> tuple[str, KVCache | None, ExpertMetrics]:
        """Run this expert on either a text prompt or a transferred KV-cache.

        Exactly one of ``prompt`` or ``cache`` must be provided. If both
        are provided, ``cache`` wins (the cache's source_text is used as
        the prompt for cost accounting).

        Args:
            prompt: Text prompt (Phase 1 path).
            cache: KV-cache from a previous expert (Phase 2 path). The
                cache will be transferred to this expert's backend first.
            max_tokens: Maximum tokens to generate.
            **kwargs: Forwarded to the backend.

        Returns:
            A tuple of ``(output_text, output_cache, metrics)``. The
            ``output_cache`` is None if the backend doesn't support KV-cache
            or if generation failed.
        """
        if prompt is None and cache is None:
            raise ValueError("Expert.invoke: one of `prompt` or `cache` is required.")
        if cache is not None and not self.capabilities.supports_kv_cache:
            # Backend can't use a cache — fall back to text from the cache.
            prompt = cache.source_text
            cache = None

        # Effective prompt for cost accounting
        effective_prompt = cache.source_text if cache is not None else (prompt or "")

        used_kv = cache is not None
        cache_transfer_ms = 0.0
        cache_transfer_succeeded = True
        transferred_cache: KVCache | None = cache

        if cache is not None:
            t0 = time.perf_counter()
            try:
                transferred_cache = self.backend.transfer_cache(cache)
            except Exception:
                cache_transfer_succeeded = False
                transferred_cache = None
                # Fall back to text
                effective_prompt = cache.source_text
                used_kv = False
            cache_transfer_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        output: str
        if used_kv and transferred_cache is not None:
            output = self.backend.decode(transferred_cache, max_tokens=max_tokens, **kwargs)
        else:
            output = self.backend.generate(effective_prompt, max_tokens=max_tokens, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Produce an output cache (for downstream experts) when supported.
        output_cache: KVCache | None = None
        if self.capabilities.supports_kv_cache:
            try:
                # If we got a cache in, we can re-encode the combined text
                # (input + output) for the next expert. This is the mock
                # path; real HF backend would extend the cache in-place.
                combined = effective_prompt + "\n" + output
                output_cache = self.backend.encode(combined)
            except Exception:
                output_cache = None

        # Token accounting — best-effort. Backends may or may not expose
        # count_tokens; if not, fall back to a 4-chars-per-token estimate.
        if hasattr(self.backend, "count_tokens"):
            try:
                in_tok = int(self.backend.count_tokens(effective_prompt))
                out_tok = int(self.backend.count_tokens(output))
            except Exception:
                in_tok = max(1, len(effective_prompt) // 4)
                out_tok = max(1, len(output) // 4)
        else:
            in_tok = max(1, len(effective_prompt) // 4)
            out_tok = max(1, len(output) // 4)

        metrics = ExpertMetrics(
            expert_name=self.name,
            role=self.role,
            params_m=self.params_m,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            cache_transfer_ms=cache_transfer_ms,
            used_kv_cache=used_kv,
            cache_transfer_succeeded=cache_transfer_succeeded,
        )
        return output, output_cache, metrics
