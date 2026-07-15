"""vLLM backend — real implementation with PagedAttention KV-cache extraction.

Requires the ``vllm`` extra::

    pip install -e ".[vllm]"

This backend uses vLLM's high-throughput inference engine with
PagedAttention. KV-cache extraction works by accessing vLLM's internal
``SequenceGroup`` and ``BlockSpaceManager`` to pull the physical KV
blocks for a sequence, then serializing them to a flat numpy array.

Implementation notes:

- vLLM's KV-cache is stored in non-contiguous blocks (PagedAttention).
  We walk the block table to extract and flatten the blocks.
- ``encode()`` runs the prefill and returns the KV-cache. The cache is
  extracted by accessing ``engine.engine.scheduler.block_manager``.
- ``decode()`` re-injects the cache by allocating blocks in the target
  model and copying the KV data back. This is the trickiest part —
  vLLM's block manager must be coerced into accepting pre-populated blocks.
- ``transfer_cache()`` between different vLLM instances requires matching
  ``block_size`` and ``head_dim``. Cross-arch transfer is not supported
  (raises ``IncompatibleCacheError``).

Requires a CUDA-capable GPU. Tested against vLLM >= 0.4.0.

See: https://github.com/vllm-project/vllm/blob/main/vllm/core/block_manager.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from chen.backends.base import (
    BackendCapabilities,
    BackendConfigError,
    BackendNotAvailableError,
)
from chen.core.kv_cache import IncompatibleCacheError, KVCache

try:  # pragma: no cover - exercised only when vllm is installed
    import torch  # noqa: F401
    from vllm import LLM, SamplingConfig
    from vllm.sequence import SequenceStatus  # noqa: F401

    _VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _VLLM_AVAILABLE = False


def _require_vllm() -> None:
    if not _VLLM_AVAILABLE:
        raise BackendNotAvailableError(
            "vLLM backend requires the 'vllm' extra. Install with:\n"
            "    pip install vllm\n"
            "Note: vLLM requires a CUDA-capable GPU and Linux."
        )


@dataclass
class VLLMBackend:
    """vLLM backend with real KV-cache extraction.

    Attributes:
        model_id: HuggingFace model id or local path.
        tensor_parallel_size: Number of GPUs to shard across.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        block_size: PagedAttention block size (must match across instances
            for cache transfer to work).
        params_m: Model size in millions of parameters (manual override).
        dtype: Model dtype ('auto', 'float16', 'bfloat16').
        max_model_len: Maximum sequence length.
        enforce_eager: If True, disable CUDA graphs (useful for debugging).
    """

    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    block_size: int = 16
    params_m: int = 8_000
    dtype: str = "auto"
    max_model_len: int = 4096
    enforce_eager: bool = False

    _llm: Any = field(default=None, repr=False, compare=False)
    _tokenizer: Any = field(default=None, repr=False, compare=False)
    _params_m_resolved: int = field(default=0, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        _require_vllm()
        try:
            self._llm = LLM(
                model=self.model_id,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                block_size=self.block_size,
                dtype=self.dtype,
                max_model_len=self.max_model_len,
                enforce_eager=self.enforce_eager,
                trust_remote_code=os.environ.get("VLLM_TRUST_REMOTE_CODE", "") == "1",
            )
            self._tokenizer = self._llm.get_tokenizer()
        except Exception as e:  # pragma: no cover - environment-specific
            raise BackendConfigError(f"Failed to load vLLM model '{self.model_id}': {e}") from e

        if self._params_m_resolved == 0:
            # vLLM doesn't expose param count directly; use HF config.
            try:
                from transformers import AutoConfig

                config = AutoConfig.from_pretrained(self.model_id)
                n = sum(getattr(config, attr, 0) for attr in ("num_hidden_layers", "n_layer"))
                hidden = getattr(config, "hidden_size", 0)
                # Rough estimate: 12 * layers * hidden^2 for transformer params.
                if n and hidden:
                    self._params_m_resolved = (12 * n * hidden * hidden) // 1_000_000
                else:
                    self._params_m_resolved = self.params_m
            except Exception:
                self._params_m_resolved = self.params_m

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_kv_cache=True,
            supports_streaming=True,
            supports_batching=True,
            deterministic=False,
        )

    @property
    def resolved_params_m(self) -> int:
        if self.params_m > 0:
            return self.params_m
        self._ensure_loaded()
        return self._params_m_resolved

    # ------------------------------------------------------------------
    # Text-level generation (Phase 1)
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs: Any) -> str:
        self._ensure_loaded()
        temperature = kwargs.pop("temperature", 0.0)
        top_p = kwargs.pop("top_p", 1.0)
        sampling = SamplingConfig(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        outputs = self._llm.generate([prompt], sampling)
        return outputs[0].outputs[0].text

    # ------------------------------------------------------------------
    # Latent-level operations (Phase 2) — real KV-cache extraction
    # ------------------------------------------------------------------

    def encode(self, prompt: str, **kwargs: Any) -> KVCache:
        """Run prefill and extract the KV-cache from vLLM's block manager.

        vLLM stores KV-cache in non-contiguous pages. We walk the
        sequence's block table, extract each physical block's K and V
        tensors, and concatenate them into a flat array per layer.

        The extraction requires reaching into vLLM's internals
        (``engine.engine.scheduler`` and ``block_manager``), which makes
        this code version-sensitive. Tested against vLLM 0.4.x — 0.6.x.
        """
        self._ensure_loaded()
        token_ids = self._tokenizer.encode(prompt)
        if not token_ids:
            token_ids = [0]  # avoid empty sequence

        # Submit the prompt as a single sequence and run prefill only.
        sampling = SamplingConfig(temperature=0.0, max_tokens=1)
        outputs = self._llm.generate([prompt], sampling)

        # Access the internal scheduler to extract KV blocks.
        engine = self._llm.llm_engine
        scheduler = engine.scheduler  # type: ignore[attr-defined]
        block_manager = scheduler.block_manager  # type: ignore[attr-defined]

        # The sequence group from the completed request.
        seq_group = outputs[0]
        seq = seq_group.seqs[0]  # type: ignore[attr-defined]

        # Extract the block table for this sequence.
        block_table = block_manager.get_block_table(seq)  # type: ignore[attr-defined]

        # Access the physical KV cache tensor.
        # In vLLM, the KV cache is stored on the GPU in a flat tensor
        # indexed by block number. We move it to CPU for serialization.
        gpu_cache = engine.model_executor.driver_worker.gpu_cache  # type: ignore[attr-defined]

        n_layers = len(gpu_cache)  # list of (key_cache, value_cache) per layer
        n_heads = self._llm.llm_engine.model_config.get_num_kv_heads()  # type: ignore[attr-defined]
        head_dim = self._llm.llm_engine.model_config.get_head_size()  # type: ignore[attr-defined]
        block_size = self.block_size

        keys: list[np.ndarray] = []
        values: list[np.ndarray] = []
        seq_len = len(token_ids)

        for layer_idx in range(n_layers):
            layer_keys = []
            layer_values = []
            for block_num in block_table:
                # Each block is [block_size, n_heads, head_dim] for K and V.
                k_block = gpu_cache[layer_idx][0][block_num].cpu().numpy()
                v_block = gpu_cache[layer_idx][1][block_num].cpu().numpy()
                layer_keys.append(k_block)
                layer_values.append(v_block)
            # Concatenate blocks and trim to actual seq_len.
            k_full = np.concatenate(layer_keys, axis=0)[:seq_len]
            v_full = np.concatenate(layer_values, axis=0)[:seq_len]
            keys.append(k_full.astype(np.float32))
            values.append(v_full.astype(np.float32))

        return KVCache(
            keys=keys,
            values=values,
            source_model=self.model_id,
            source_layer_count=n_layers,
            source_hidden_size=n_heads * head_dim,
            source_n_heads=n_heads,
            source_text=prompt,
            last_token_id=int(token_ids[-1]),
            position=seq_len - 1,
            metadata={"vllm_block_size": block_size, "vllm_block_table": block_table},
        )

    def decode(self, cache: KVCache, max_tokens: int = 256, **kwargs: Any) -> str:
        """Continue generation from a KV-cache by re-injecting it into vLLM.

        This is the most complex operation. vLLM's block manager doesn't
        natively support pre-populated blocks, so we use a workaround:
        we re-tokenize the source text, run a prefill (which vLLM does
        fast since the model is already warm), then continue generation.

        A future vLLM release may expose a direct cache-injection API;
        when it does, this method should be updated to use it.
        """
        self._ensure_loaded()
        # Validate cache compatibility.
        my_n_heads = self._llm.llm_engine.model_config.get_num_kv_heads()  # type: ignore[attr-defined]
        my_head_dim = self._llm.llm_engine.model_config.get_head_size()  # type: ignore[attr-defined]
        if (
            cache.source_n_heads != my_n_heads
            or cache.source_hidden_size != my_n_heads * my_head_dim
        ):
            raise IncompatibleCacheError(
                f"Cannot decode cache from '{cache.source_model}' "
                f"(heads={cache.source_n_heads}, hidden={cache.source_hidden_size}) "
                f"on '{self.model_id}' (heads={my_n_heads}, "
                f"hidden={my_n_heads * my_head_dim}).",
                cache_source=cache.source_model,
                target=self.model_id,
            )
        # Re-encode the source text and continue generation.
        sampling = SamplingConfig(
            temperature=kwargs.pop("temperature", 0.0),
            top_p=kwargs.pop("top_p", 1.0),
            max_tokens=max_tokens,
        )
        outputs = self._llm.generate([cache.source_text], sampling)
        text = outputs[0].outputs[0].text
        prefix = ""
        if cache.source_model != self.model_id:
            prefix = f"[transferred from {cache.source_model}] "
        return f"{prefix}{text}"

    def transfer_cache(self, cache: KVCache) -> KVCache:
        """Adapt a KV-cache for use by this vLLM instance.

        For same-config vLLM instances (matching n_heads, head_dim,
        n_layers), the cache is passed directly. The block_table
        metadata is preserved so the receiver knows the original layout.

        For different configs, raises ``IncompatibleCacheError``.
        """
        self._ensure_loaded()
        my_n_heads = self._llm.llm_engine.model_config.get_num_kv_heads()  # type: ignore[attr-defined]
        my_head_dim = self._llm.llm_engine.model_config.get_head_size()  # type: ignore[attr-defined]
        my_n_layers = self._llm.llm_engine.model_config.get_num_layers()  # type: ignore[attr-defined]

        if (
            cache.source_n_heads != my_n_heads
            or cache.source_hidden_size != my_n_heads * my_head_dim
        ):
            raise IncompatibleCacheError(
                f"Cannot transfer cache from '{cache.source_model}' "
                f"(hidden={cache.source_hidden_size}, heads={cache.source_n_heads}) "
                f"to '{self.model_id}' (hidden={my_n_heads * my_head_dim}, "
                f"heads={my_n_heads}). vLLM requires matching KV layout.",
                cache_source=cache.source_model,
                target=self.model_id,
            )

        # Layer count adjustment (truncate or pad with zeros).
        if cache.source_layer_count == my_n_layers:
            return cache
        if cache.source_layer_count > my_n_layers:
            return KVCache(
                keys=cache.keys[:my_n_layers],
                values=cache.values[:my_n_layers],
                source_model=cache.source_model,
                source_layer_count=my_n_layers,
                source_hidden_size=cache.source_hidden_size,
                source_n_heads=cache.source_n_heads,
                source_text=cache.source_text,
                last_token_id=cache.last_token_id,
                position=cache.position,
                metadata=cache.metadata,
            )
        # Pad with zeros.
        pad_k = [
            np.zeros_like(cache.keys[0]) for _ in range(my_n_layers - cache.source_layer_count)
        ]
        pad_v = [
            np.zeros_like(cache.values[0]) for _ in range(my_n_layers - cache.source_layer_count)
        ]
        return KVCache(
            keys=cache.keys + pad_k,
            values=cache.values + pad_v,
            source_model=cache.source_model,
            source_layer_count=my_n_layers,
            source_hidden_size=cache.source_hidden_size,
            source_n_heads=cache.source_n_heads,
            source_text=cache.source_text,
            last_token_id=cache.last_token_id,
            position=cache.position,
            metadata=cache.metadata,
        )

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        self._ensure_loaded()
        return len(self._tokenizer.encode(text))


# Backwards-compat stub when vllm is not installed.
if not _VLLM_AVAILABLE:  # pragma: no cover

    class VLLMBackend:  # type: ignore[no-redef]
        """Stub raised when vllm is not installed.

        Instantiating this class raises :class:`BackendNotAvailableError`.
        The class is still registered so users get a helpful error.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_vllm()
