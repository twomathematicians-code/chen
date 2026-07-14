"""llama.cpp backend — stub for v0.1.

llama.cpp runs GGUF-quantized models on CPU or Apple Metal, making it the
right choice for Mac M-series deployment or CPU-only servers. Its Python
binding (``llama-cpp-python``) exposes a limited KV-cache API via
``llama_get_kv_cache`` and ``llama_kv_cache_seq_*`` functions, but
extracting a serializable cache suitable for cross-process transfer is
not straightforward.

This stub registers the backend so the registry finds it, supports
text-level :meth:`generate` (Phase 1), and raises
:class:`NotImplementedError` for the latent-level operations needed by
Phase 2.

Implementation hints for a contributor:

1. Use ``Llama(model_path=..., n_ctx=..., n_gpu_layers=...)`` to load.
2. For ``encode``, run ``llama.decode(...)`` on the prompt tokens, then
   call ``llama._ctx.get_kv_cache()`` to retrieve the cache tensor.
3. For ``decode``, copy the tensor back into the target model's KV cache
   via ``llama._ctx.set_kv_cache(...)``. The Q/K/V layout in llama.cpp
   is interleaved — check ``llama.h`` for the exact memory layout.
4. ``transfer_cache`` between different GGUF models requires matching
   ``n_layer`` and ``n_embd``; cross-arch transfer is not supported.

See: https://github.com/abetlen/llama-cpp-python
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from chen.backends.base import (
    BackendCapabilities,
    BackendNotAvailableError,
)
from chen.core.kv_cache import KVCache

try:  # pragma: no cover - exercised only when llama-cpp-python is installed
    from llama_cpp import Llama  # type: ignore

    _LLAMA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LLAMA_AVAILABLE = False


def _require_llama() -> None:
    if not _LLAMA_AVAILABLE:
        raise BackendNotAvailableError(
            "llama.cpp backend requires the 'llama-cpp' extra. Install with:\n"
            "    pip install llama-cpp-python\n"
            "You will also need a GGUF model file. See .env.example."
        )


@dataclass
class LlamaCppBackend:
    """llama.cpp backend (stub for v0.1).

    Attributes:
        model_path: Path to a .gguf model file on disk.
        n_ctx: Context window size in tokens.
        n_gpu_layers: Number of layers to offload to GPU (Metal/CUDA).
        params_m: Model size in millions of parameters (manual override).
    """

    model_path: str = ""
    n_ctx: int = 4096
    n_gpu_layers: int = 0
    params_m: int = 8_000

    _llm: Any = None

    def __post_init__(self) -> None:
        if not self.model_path:
            self.model_path = os.environ.get(
                "CHEN_LLAMA_MODEL_PATH", "./models/llama-3-8b-instruct.Q4_K_M.gguf"
            )

    @property
    def model_id(self) -> str:
        return f"llama_cpp:{os.path.basename(self.model_path)}"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_kv_cache=False,  # TODO: implement KV extraction
            supports_streaming=True,
            supports_batching=False,
            deterministic=False,
        )

    def _not_implemented(self, method: str) -> NotImplementedError:
        return NotImplementedError(
            f"LlamaCppBackend.{method}() is not implemented in v0.1. "
            f"This is a stub. See chen/backends/llama_cpp.py for implementation hints."
        )

    def _ensure_loaded(self) -> Any:
        if self._llm is not None:
            return self._llm
        _require_llama()
        self._llm = Llama(  # pragma: no cover - requires llama-cpp-python
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )
        return self._llm

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs: Any) -> str:
        llm = self._ensure_loaded()
        out = llm(
            prompt,
            max_tokens=max_tokens,
            echo=False,
            **kwargs,
        )
        return out["choices"][0]["text"]  # pragma: no cover

    def encode(self, prompt: str, **kwargs: Any) -> KVCache:
        raise self._not_implemented("encode")

    def decode(self, cache: KVCache, max_tokens: int = 256, **kwargs: Any) -> str:
        raise self._not_implemented("decode")

    def transfer_cache(self, cache: KVCache) -> KVCache:
        raise self._not_implemented("transfer_cache")
