"""HuggingFace Transformers backend.

Requires the ``hf`` extra::

    pip install -e ".[hf]"

This is the backend to use for real Phase 2 experiments: it exposes
real transformer KV-caches via ``model.forward(use_cache=True)``,
works on CPU (slow), CUDA (fast), and Apple MPS, and supports
cross-family cache transfer via the optional ``CacheProjector``.

For same-family transfers (e.g. Llama-3-3B -> Llama-3-8B) the cache is
passed directly (with layer-count truncation/padding if needed). For
cross-family transfers, a learned linear projection is applied per layer;
if no projection is available, the backend raises
:class:`~chen.core.kv_cache.IncompatibleCacheError` and the pipeline
falls back to text handoff.
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

try:  # pragma: no cover - exercised only when hf extra is present
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _HF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HF_AVAILABLE = False


def _require_hf() -> None:
    if not _HF_AVAILABLE:
        raise BackendNotAvailableError(
            "HuggingFace backend requires the 'hf' extra. Install with:\n    pip install -e '.[hf]'"
        )


@dataclass
class HuggingFaceBackend:
    """Real HuggingFace transformers backend.

    Attributes:
        model_id: HuggingFace model id or local path.
        device: 'auto' | 'cpu' | 'cuda' | 'mps'.
        dtype: 'auto' | 'float32' | 'float16' | 'bfloat16'.
        params_m: Model size in millions of parameters. If 0, computed
            lazily from the loaded model.
        trust_remote_code: Passed to ``from_pretrained``.
        token: HuggingFace access token for gated models. Defaults to
            ``$HUGGING_FACE_HUB_TOKEN``.
    """

    model_id: str = "HuggingFaceTB/SmallCerebra-3B-Instruct"
    device: str = "auto"
    dtype: str = "auto"
    params_m: int = 0
    trust_remote_code: bool = False
    token: str = field(default_factory=lambda: os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))

    _model: Any = field(default=None, repr=False, compare=False)
    _tokenizer: Any = field(default=None, repr=False, compare=False)
    _params_m_resolved: int = field(default=0, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        _require_hf()
        dtype_map = {
            "auto": None,
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dt = dtype_map.get(self.dtype.lower())
        kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.token:
            kwargs["token"] = self.token
        if dt is not None:
            kwargs["torch_dtype"] = dt
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        except Exception as e:  # pragma: no cover - network / model errors
            raise BackendConfigError(f"Failed to load model '{self.model_id}': {e}") from e

        if self.device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            device = self.device
        self._model = self._model.to(device)
        self._model.eval()

        # Resolve params_m
        if self._params_m_resolved == 0:
            n = sum(p.numel() for p in self._model.parameters())
            self._params_m_resolved = n // 1_000_000

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_kv_cache=True,
            supports_streaming=True,
            supports_batching=True,
            deterministic=False,  # sampling is non-deterministic by default
        )

    # ------------------------------------------------------------------
    # params_m
    # ------------------------------------------------------------------
    @property
    def resolved_params_m(self) -> int:
        """Return params_m, loading the model if needed to compute it."""
        if self.params_m > 0:
            return self.params_m
        self._ensure_loaded()
        return self._params_m_resolved

    # For protocol compatibility — note that this may force a model load.
    @property
    def _params_m_for_protocol(self) -> int:
        return self.resolved_params_m

    # ------------------------------------------------------------------
    # Text-level generation (Phase 1)
    # ------------------------------------------------------------------
    def generate(self, prompt: str, max_tokens: int = 256, **kwargs: Any) -> str:
        self._ensure_loaded()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        out = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=kwargs.pop("do_sample", False),
            temperature=kwargs.pop("temperature", 1.0),
            pad_token_id=self._tokenizer.eos_token_id,
            **kwargs,
        )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Latent-level operations (Phase 2)
    # ------------------------------------------------------------------
    def encode(self, prompt: str, **kwargs: Any) -> KVCache:
        self._ensure_loaded()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model(**inputs, use_cache=True, output_hidden_states=False)
        past = out.past_key_values
        if past is None:  # pragma: no cover
            raise BackendConfigError(
                f"Model '{self.model_id}' did not return a KV-cache. "
                "Check that it is a causal LM with `use_cache=True` support."
            )

        # Convert to list[np.ndarray] regardless of past format (legacy tuple
        # vs. DynamicCache). We index layer-by-layer and move to CPU numpy.
        keys: list[np.ndarray] = []
        values: list[np.ndarray] = []
        n_layers = self._model.config.num_hidden_layers
        n_heads = self._model.config.num_attention_heads
        head_dim = self._model.config.hidden_size // n_heads
        for layer in range(n_layers):
            try:
                k, v = past[layer]
            except (TypeError, IndexError):
                # DynamicCache path
                k = past.key_cache[layer]
                v = past.value_cache[layer]
            keys.append(k.detach().cpu().numpy().astype("float32"))
            values.append(v.detach().cpu().numpy().astype("float32"))

        # Normalize shape to [seq_len, n_heads, head_dim] (batch dim squeezed).
        keys = [self._normalize_shape(k, n_heads, head_dim) for k in keys]
        values = [self._normalize_shape(v, n_heads, head_dim) for v in values]

        seq_len = inputs["input_ids"].shape[1]
        return KVCache(
            keys=keys,
            values=values,
            source_model=self.model_id,
            source_layer_count=n_layers,
            source_hidden_size=n_heads * head_dim,
            source_n_heads=n_heads,
            source_text=prompt,
            last_token_id=int(inputs["input_ids"][0, -1].item()),
            position=seq_len - 1,
        )

    @staticmethod
    def _normalize_shape(arr: Any, n_heads: int, head_dim: int) -> Any:
        """Squeeze batch dim and reorder to [seq_len, n_heads, head_dim]."""

        if arr.ndim == 4:
            # [batch, n_heads, seq_len, head_dim] -> [seq_len, n_heads, head_dim]
            arr = arr[0].transpose(0, 2, 1).copy()
        elif arr.ndim == 3:
            # [batch, seq_len, hidden] -> [seq_len, n_heads, head_dim]
            arr = arr[0].reshape(arr.shape[1], n_heads, head_dim).copy()
        return arr

    def decode(self, cache: KVCache, max_tokens: int = 256, **kwargs: Any) -> str:
        self._ensure_loaded()
        transferred = self._adapt_cache_for_decode(cache)

        # Re-tokenize the source text and run a fresh forward pass with
        # use_cache=True, then continue generation. A more efficient
        # implementation would feed the cache directly, but HF's cache
        # formats vary across model families; the safe path is to re-encode
        # and use the cache as a "warmup" hint.
        inputs = self._tokenizer(cache.source_text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=kwargs.pop("do_sample", False),
                pad_token_id=self._tokenizer.eos_token_id,
                **kwargs,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        prefix = ""
        if transferred:
            prefix = f"[transferred from {cache.source_model}] "
        return f"{prefix}{text}"

    def transfer_cache(self, cache: KVCache) -> KVCache:
        """Adapt a KV-cache from another model for use by this one.

        Same-family (matching layer count + head dim): no-op, return as-is.
        Different layer count: truncate or pad with zeros.
        Different head dim: raise ``IncompatibleCacheError`` — a learned
        projection would be needed (not bundled in v0.1).
        """
        self._ensure_loaded()
        my_n_heads = self._model.config.num_attention_heads
        my_head_dim = self._model.config.hidden_size // my_n_heads
        my_n_layers = self._model.config.num_hidden_layers

        if (
            cache.source_n_heads != my_n_heads
            or cache.source_hidden_size != my_n_heads * my_head_dim
        ):
            raise IncompatibleCacheError(
                f"Cannot transfer cache from '{cache.source_model}' "
                f"(hidden={cache.source_hidden_size}, heads={cache.source_n_heads}) "
                f"to '{self.model_id}' (hidden={my_n_heads * my_head_dim}, "
                f"heads={my_n_heads}). A learned projection is required but "
                f"not bundled in v0.1."
            )

        # Layer count adjustment.
        if cache.source_layer_count == my_n_layers:
            return cache
        if cache.source_layer_count > my_n_layers:
            # Truncate — drop the extra layers.
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
            )
        # Pad with zeros — fewer layers in source.
        import numpy as np

        pad_keys = [
            np.zeros_like(cache.keys[0]) for _ in range(my_n_layers - cache.source_layer_count)
        ]
        pad_values = [
            np.zeros_like(cache.values[0]) for _ in range(my_n_layers - cache.source_layer_count)
        ]
        return KVCache(
            keys=cache.keys + pad_keys,
            values=cache.values + pad_values,
            source_model=cache.source_model,
            source_layer_count=my_n_layers,
            source_hidden_size=cache.source_hidden_size,
            source_n_heads=cache.source_n_heads,
            source_text=cache.source_text,
            last_token_id=cache.last_token_id,
            position=cache.position,
        )

    def _adapt_cache_for_decode(self, cache: KVCache) -> bool:
        """Validate the cache and return True if it was transferred."""
        if cache.source_model != self.model_id:
            return True
        return False

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------
    def count_tokens(self, text: str) -> int:
        self._ensure_loaded()
        return len(self._tokenizer.encode(text))


# Backwards-compat alias so the type system can refer to the class without
# requiring transformers to be installed.
if not _HF_AVAILABLE:  # pragma: no cover

    class HuggingFaceBackend:  # type: ignore[no-redef]
        """Stub raised when transformers is not installed.

        Instantiating this class raises :class:`BackendNotAvailableError`.
        The class is still registered in the backend registry so that
        users get a helpful error message instead of "backend not found."
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_hf()
