"""Backend protocol & shared types.

Every inference backend in CHEN implements the :class:`InferenceBackend`
protocol. The protocol is intentionally minimal — four methods plus a
``params_m`` property — so that new backends (TGI, Triton, ONNX Runtime,
etc.) can be added without touching the core pipeline.

The protocol is split into two halves:

1. **Text-level** operations (:meth:`generate`). These work for every
   backend and are sufficient for Phase 1 (static cascade).
2. **Latent-level** operations (:meth:`encode`, :meth:`decode`,
   :meth:`transfer_cache`). These expose the KV-cache and are required
   for Phase 2 (KV-pass). Backends that don't support latent ops should
   report ``BackendCapabilities.supports_kv_cache = False`` and the
   pipeline will fall back to text handoff with a logged warning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    # Avoid circular import at runtime — KVCache is only used for type hints.
    from chen.core.kv_cache import KVCache


@dataclass(frozen=True)
class BackendCapabilities:
    """Declares what a backend can do. Used by the pipeline to pick the
    right handoff strategy and to give helpful errors.

    Attributes:
        supports_kv_cache: If True, ``encode``/``decode``/``transfer_cache``
            are implemented and Phase 2 can run on this backend.
        supports_streaming: If True, the backend can yield tokens one at a
            time. Currently informational; streaming pipeline is roadmap.
        supports_batching: If True, multiple prompts can be processed in
            one forward pass. Currently informational.
        deterministic: If True, repeated calls with the same prompt
            produce byte-identical output. Required for unit tests.
    """

    supports_kv_cache: bool = False
    supports_streaming: bool = False
    supports_batching: bool = False
    deterministic: bool = False


@runtime_checkable
class InferenceBackend(Protocol):
    """Pluggable inference backend protocol.

    All sizes are in millions of parameters (3_000 = 3B). Tokens are
    counted via the backend's own tokenizer; the pipeline trusts the
    count for cost calculation.
    """

    @property
    def params_m(self) -> int:
        """Model size in millions of parameters (3_000 = 3B)."""
        ...

    @property
    def capabilities(self) -> BackendCapabilities:
        """What this backend can do."""
        ...

    @property
    def model_id(self) -> str:
        """A stable identifier for this backend instance (model name or path)."""
        ...

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs: Any) -> str:
        """Generate text from a text prompt. Phase 1 path."""
        ...

    def encode(self, prompt: str, **kwargs: Any) -> KVCache:
        """Run prefill on the prompt and return its KV-cache. Phase 2 path.

        Raises:
            NotImplementedError: if ``capabilities.supports_kv_cache`` is False.
        """
        ...

    def decode(self, cache: KVCache, max_tokens: int = 256, **kwargs: Any) -> str:
        """Continue generation from a KV-cache. Phase 2 path.

        Raises:
            NotImplementedError: if ``capabilities.supports_kv_cache`` is False.
            IncompatibleCacheError: if the cache cannot be used by this backend.
        """
        ...

    def transfer_cache(self, cache: KVCache) -> KVCache:
        """Adapt a KV-cache produced by another backend for use by this backend.

        For same-family backends this is typically a no-op (or a layer-count
        adjustment). For cross-family backends this applies a learned
        projection. The default implementation returns the cache unchanged.

        Raises:
            NotImplementedError: if ``capabilities.supports_kv_cache`` is False.
            IncompatibleCacheError: if no projection is available and the
                cache shapes are incompatible.
        """
        ...


class BackendError(Exception):
    """Base class for backend errors."""


class BackendNotAvailableError(BackendError):
    """Raised when a backend is requested but its dependencies are not installed."""


class BackendConfigError(BackendError):
    """Raised when a backend is configured incorrectly (e.g. missing model path)."""
