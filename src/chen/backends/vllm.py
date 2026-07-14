"""vLLM backend — stub for v0.1.

vLLM is a high-throughput inference engine that uses PagedAttention for
memory-efficient KV-cache management. Implementing Phase 2 KV-pass
between vLLM instances is non-trivial because PagedAttention stores
KV-cache in non-contiguous blocks — extracting and re-injecting a
cache requires the (still-evolving) ``SequenceGroup`` API.

This stub:

* Registers itself as the ``vllm`` backend so the registry can find it.
* Raises :class:`NotImplementedError` with a helpful message for every
  method except :meth:`params_m` (which works once a model is loaded).
* Provides a scaffold for a future contributor to fill in.

Implementation hints for a contributor:

1. Use ``LLMEngine.add_request()`` to submit the prompt.
2. After the engine processes the prefill, walk its internal
   ``scheduler.block_manager`` to extract the KV blocks for the sequence.
3. Serialize the blocks to a flat numpy array (be careful with the
   block-table indirection — see vLLM's ``BlockSpaceManager``).
4. On the target model, allocate blocks and copy the array back in.
5. ``transfer_cache`` between different vLLM instances requires matching
   ``block_size`` and ``head_dim``.

See: https://github.com/vllm-project/vllm/blob/main/vllm/core/block_manager.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chen.backends.base import (
    BackendCapabilities,
    BackendNotAvailableError,
)
from chen.core.kv_cache import KVCache

try:  # pragma: no cover - exercised only when vllm is installed
    import vllm  # type: ignore  # noqa: F401

    _VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _VLLM_AVAILABLE = False


def _require_vllm() -> None:
    if not _VLLM_AVAILABLE:
        raise BackendNotAvailableError(
            "vLLM backend requires the 'vllm' extra. Install with:\n"
            "    pip install vllm\n"
            "Note: vLLM requires a CUDA-capable GPU."
        )


@dataclass
class VLLMBackend:
    """vLLM backend (stub for v0.1).

    Attributes:
        model_id: HuggingFace model id or local path.
        tensor_parallel_size: Number of GPUs to shard across.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        params_m: Model size in millions of parameters (manual override).
    """

    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    params_m: int = 8_000

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_kv_cache=False,  # TODO: implement PagedAttention KV extraction
            supports_streaming=True,
            supports_batching=True,
            deterministic=False,
        )

    def _not_implemented(self, method: str) -> NotImplementedError:
        return NotImplementedError(
            f"VLLMBackend.{method}() is not implemented in v0.1. "
            f"This is a stub. See chen/backends/vllm.py for implementation hints. "
            f"If you'd like to contribute this backend, please open an issue "
            f"at https://github.com/your-org/chen/issues."
        )

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs: Any) -> str:
        _require_vllm()
        # Minimal generate path using vLLM's LLM class. This works for Phase 1.
        from vllm import LLM, SamplingParams  # type: ignore

        llm = LLM(
            model=self.model_id,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        sampling = SamplingParams(max_tokens=max_tokens, **kwargs)
        outputs = llm.generate([prompt], sampling)
        return outputs[0].outputs[0].text

    def encode(self, prompt: str, **kwargs: Any) -> KVCache:
        raise self._not_implemented("encode")

    def decode(self, cache: KVCache, max_tokens: int = 256, **kwargs: Any) -> str:
        raise self._not_implemented("decode")

    def transfer_cache(self, cache: KVCache) -> KVCache:
        raise self._not_implemented("transfer_cache")
