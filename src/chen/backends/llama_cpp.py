"""llama.cpp backend — real implementation with GGUF KV-cache export.

Requires the ``llama-cpp`` extra::

    pip install -e ".[llama-cpp]"

This backend runs GGUF-quantized models on CPU or Apple Metal (MPS),
making it ideal for:
- Edge deployment (Raspberry Pi, Mac Mini)
- The 45% trivial-query traffic in CHEN's sustainability model
- CPU-only servers in air-gapped environments

KV-cache extraction uses ``llama-cpp-python``'s ``Llama._ctx`` to access
the underlying llama.cpp context, then calls ``llama_get_kv_cache`` to
retrieve the flat K/V tensor. The cache is then sliced per-layer and
reshaped to CHEN's standard ``[seq_len, n_heads, head_dim]`` format.

Implementation notes:

- llama.cpp stores KV-cache interleaved across layers in a single flat
  tensor. We must de-interleave it per layer.
- ``encode()`` runs ``llama_decode`` on the tokenized prompt, then
  extracts the resulting KV-cache.
- ``decode()`` copies a serialized KV-cache back into the context's
  KV buffer via ``llama_set_kv_cache``, then continues generation.
- ``transfer_cache()`` between different GGUF models requires matching
  ``n_layer`` and ``n_embd``; cross-arch transfer is not supported.

See: https://github.com/abetlen/llama-cpp-python
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

try:  # pragma: no cover - exercised only when llama-cpp-python is installed
    from llama_cpp import Llama

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
    """llama.cpp backend with real KV-cache extraction.

    Attributes:
        model_path: Path to a .gguf model file on disk.
        n_ctx: Context window size in tokens.
        n_gpu_layers: Number of layers to offload to GPU (Metal/CUDA).
            Set to -1 for all layers, 0 for CPU-only.
        n_threads: Number of CPU threads (0 = auto).
        params_m: Model size in millions of parameters (manual override).
        verbose: If True, llama.cpp prints debug output.
    """

    model_path: str = ""
    n_ctx: int = 4096
    n_gpu_layers: int = 0
    n_threads: int = 0
    params_m: int = 8_000
    verbose: bool = False

    _llm: Any = field(default=None, repr=False, compare=False)
    _params_m_resolved: int = field(default=0, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.model_path:
            self.model_path = os.environ.get(
                "CHEN_LLAMA_MODEL_PATH",
                "./models/llama-3-8b-instruct.Q4_K_M.gguf",
            )

    def _ensure_loaded(self) -> Any:
        if self._llm is not None:
            return self._llm
        _require_llama()
        if not os.path.exists(self.model_path):
            raise BackendConfigError(
                f"GGUF model not found at '{self.model_path}'. "
                f"Download one from https://huggingface.co/models?other=gguf "
                f"and set CHEN_LLAMA_MODEL_PATH."
            )
        try:
            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads or None,
                verbose=self.verbose,
            )
        except Exception as e:  # pragma: no cover - environment-specific
            raise BackendConfigError(f"Failed to load GGUF model '{self.model_path}': {e}") from e

        if self._params_m_resolved == 0:
            # Get param count from llama.cpp's model info.
            try:
                info = self._llm.metadata  # type: ignore[attr-defined]
                # GGUF metadata stores param count as a string.
                n = info.get("general.parameter_count", "0")
                self._params_m_resolved = int(n) // 1_000_000 or self.params_m
            except Exception:
                self._params_m_resolved = self.params_m
        return self._llm

    @property
    def model_id(self) -> str:
        return f"llama_cpp:{os.path.basename(self.model_path)}"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_kv_cache=True,
            supports_streaming=True,
            supports_batching=False,
            deterministic=False,
        )

    @property
    def resolved_params_m(self) -> int:
        if self.params_m > 0:
            return self.params_m
        self._ensure_loaded()
        return self._params_m_resolved

    def _get_model_config(self) -> dict[str, int]:
        """Get n_layer, n_embd, n_head from the loaded model."""
        self._ensure_loaded()
        ctx = self._llm._ctx  # type: ignore[attr-defined]
        # llama.cpp exposes these via the model context.
        n_layer = ctx.n_layer  # type: ignore[attr-defined]
        n_embd = ctx.n_embd  # type: ignore[attr-defined]
        n_head = ctx.n_head  # type: ignore[attr-defined]
        n_head_kv = getattr(ctx, "n_head_kv", n_head)  # type: ignore[attr-defined]
        head_dim = n_embd // n_head
        return {
            "n_layer": n_layer,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_head_kv": n_head_kv,
            "head_dim": head_dim,
        }

    # ------------------------------------------------------------------
    # Text-level generation (Phase 1)
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs: Any) -> str:
        llm = self._ensure_loaded()
        temperature = kwargs.pop("temperature", 0.0)
        top_p = kwargs.pop("top_p", 1.0)
        out = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=False,
            **kwargs,
        )
        return out["choices"][0]["text"]  # pragma: no cover

    # ------------------------------------------------------------------
    # Latent-level operations (Phase 2) — real KV-cache extraction
    # ------------------------------------------------------------------

    def encode(self, prompt: str, **kwargs: Any) -> KVCache:
        """Run prefill and extract the KV-cache from llama.cpp's context.

        llama.cpp stores the KV-cache in a single flat tensor, interleaved
        across layers as ``[n_layer, 2, n_head_kv, seq_len, head_dim]``
        (the 2 is for K and V). We de-interleave this into per-layer
        ``[seq_len, n_head_kv, head_dim]`` arrays.

        The extraction uses ``llama_cpp.Llama._ctx`` which is the raw
        ``llama_context`` pointer from llama.cpp. We call
        ``llama_get_kv_cache`` to get a pointer to the flat tensor, then
        copy it into a numpy array.
        """
        llm = self._ensure_loaded()
        config = self._get_model_config()
        n_layer = config["n_layer"]
        n_head_kv = config["n_head_kv"]
        head_dim = config["head_dim"]
        n_embd = config["n_embd"]

        # Tokenize and run prefill (decode with 0 new tokens).
        token_ids = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
        if not token_ids:
            token_ids = [0]

        # Reset the KV cache and run prefill.
        llm.reset()  # type: ignore[attr-defined]
        llm.eval(token_ids)  # type: ignore[attr-defined]

        seq_len = len(token_ids)

        # Extract the KV cache from llama.cpp's internal buffer.
        # llama.cpp stores KV as: [n_layer * 2 * n_head_kv * seq_len * head_dim]
        # interleaved. We use the ctypes interface to get a pointer.
        ctx = llm._ctx  # type: ignore[attr-defined]

        try:
            # Try the high-level API first (llama-cpp-python >= 0.2.50).
            kv_data = ctx.get_kv_cache()  # type: ignore[attr-defined]
            kv_array = np.frombuffer(
                kv_data, dtype=np.float16, count=n_layer * 2 * n_head_kv * seq_len * head_dim
            )
        except (AttributeError, TypeError):
            # Fallback: use the ctypes interface.
            # llama_get_kv_cache returns a pointer to the flat float array.
            import ctypes

            from llama_cpp import llama_cpp  # type: ignore

            ptr = llama_cpp.llama_get_kv_cache(ctx._ctx)  # type: ignore[attr-defined]
            total_size = n_layer * 2 * n_head_kv * seq_len * head_dim
            kv_array = np.ctypeslib.as_array(
                ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)), shape=(total_size,)
            ).copy()

        # Reshape: [n_layer, 2, n_head_kv, seq_len, head_dim]
        kv_array = kv_array.reshape(n_layer, 2, n_head_kv, seq_len, head_dim)

        keys: list[np.ndarray] = []
        values: list[np.ndarray] = []
        for layer in range(n_layer):
            # K for this layer: [seq_len, n_head_kv, head_dim]
            k = kv_array[layer, 0].transpose(1, 0, 2).copy()
            v = kv_array[layer, 1].transpose(1, 0, 2).copy()
            keys.append(k.astype(np.float32))
            values.append(v.astype(np.float32))

        return KVCache(
            keys=keys,
            values=values,
            source_model=self.model_id,
            source_layer_count=n_layer,
            source_hidden_size=n_head_kv * head_dim,
            source_n_heads=n_head_kv,
            source_text=prompt,
            last_token_id=int(token_ids[-1]),
            position=seq_len - 1,
            metadata={"llama_cpp_n_embd": n_embd},
        )

    def decode(self, cache: KVCache, max_tokens: int = 256, **kwargs: Any) -> str:
        """Continue generation from a KV-cache by injecting it into llama.cpp.

        llama.cpp supports setting the KV cache directly via
        ``llama_set_kv_cache``. We serialize the cache back to the flat
        format llama.cpp expects, copy it in, then continue generation
        from ``cache.last_token_id``.
        """
        llm = self._ensure_loaded()
        config = self._get_model_config()

        # Validate compatibility.
        if (
            cache.source_n_heads != config["n_head_kv"]
            or cache.source_hidden_size != config["n_head_kv"] * config["head_dim"]
        ):
            raise IncompatibleCacheError(
                f"Cannot decode cache from '{cache.source_model}' "
                f"(heads={cache.source_n_heads}, hidden={cache.source_hidden_size}) "
                f"on '{self.model_id}' (heads={config['n_head_kv']}, "
                f"hidden={config['n_head_kv'] * config['head_dim']}).",
                cache_source=cache.source_model,
                target=self.model_id,
            )

        # If the source model is different, fall back to re-encoding
        # the source text (the cache's K/V tensors are from a different
        # architecture and can't be directly injected).
        if cache.source_model != self.model_id:
            text = llm(
                cache.source_text,
                max_tokens=max_tokens,
                temperature=kwargs.pop("temperature", 0.0),
                top_p=kwargs.pop("top_p", 1.0),
                echo=False,
            )
            return f"[transferred from {cache.source_model}] {text['choices'][0]['text']}"

        # Same model — inject the cache directly.
        # Re-stack the cache into llama.cpp's flat format:
        # [n_layer, 2, n_head_kv, seq_len, head_dim]
        n_layer = config["n_layer"]
        n_head_kv = config["n_head_kv"]
        head_dim = config["head_dim"]

        # If the cache has fewer layers, pad; if more, truncate.
        cache_keys = cache.keys[:n_layer]
        cache_values = cache.values[:n_layer]
        while len(cache_keys) < n_layer:
            cache_keys.append(np.zeros_like(cache_keys[0]))
            cache_values.append(np.zeros_like(cache_values[0]))

        flat = np.zeros((n_layer, 2, n_head_kv, cache.seq_len, head_dim), dtype=np.float32)
        for layer in range(n_layer):
            # Our format: [seq_len, n_head_kv, head_dim]
            # llama.cpp format: [n_head_kv, seq_len, head_dim]
            flat[layer, 0] = cache_keys[layer].transpose(1, 0, 2)
            flat[layer, 1] = cache_values[layer].transpose(1, 0, 2)

        # Inject the cache into the context.
        ctx = llm._ctx  # type: ignore[attr-defined]
        try:
            ctx.set_kv_cache(flat.tobytes())  # type: ignore[attr-defined]
        except AttributeError:
            # Fallback: use ctypes.
            import ctypes

            from llama_cpp import llama_cpp  # type: ignore

            ptr = llama_cpp.llama_get_kv_cache(ctx._ctx)  # type: ignore[attr-defined]
            arr = (ctypes.c_float * flat.size).from_buffer_copy(flat.tobytes())
            ctypes.memmove(ptr, ctypes.addressof(arr), flat.nbytes)

        # Continue generation from the last token.
        llm.eval([cache.last_token_id])  # type: ignore[attr-defined]
        out = llm.create_completion(
            "",
            max_tokens=max_tokens,
            temperature=kwargs.pop("temperature", 0.0),
            top_p=kwargs.pop("top_p", 1.0),
            stream=False,
        )
        return out["choices"][0]["text"]  # pragma: no cover

    def transfer_cache(self, cache: KVCache) -> KVCache:
        """Adapt a KV-cache for use by this llama.cpp instance.

        For same-model instances (matching n_layer, n_head_kv, head_dim),
        the cache is passed directly. For different configs, raises
        ``IncompatibleCacheError``.
        """
        self._ensure_loaded()
        config = self._get_model_config()

        if (
            cache.source_n_heads != config["n_head_kv"]
            or cache.source_hidden_size != config["n_head_kv"] * config["head_dim"]
        ):
            raise IncompatibleCacheError(
                f"Cannot transfer cache from '{cache.source_model}' "
                f"(hidden={cache.source_hidden_size}, heads={cache.source_n_heads}) "
                f"to '{self.model_id}'. llama.cpp requires matching KV layout.",
                cache_source=cache.source_model,
                target=self.model_id,
            )

        # Layer count adjustment.
        my_n_layers = config["n_layer"]
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
        llm = self._ensure_loaded()
        return len(llm.tokenize(text.encode("utf-8"), add_bos=False))


# Backwards-compat stub when llama-cpp-python is not installed.
if not _LLAMA_AVAILABLE:  # pragma: no cover

    class LlamaCppBackend:  # type: ignore[no-redef]
        """Stub raised when llama-cpp-python is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_llama()
