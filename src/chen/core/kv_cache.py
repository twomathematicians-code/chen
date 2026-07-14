"""KV-cache dataclass & transfer protocol.

The :class:`KVCache` is the lingua franca between experts in CHEN. It
holds the per-layer keys and values of a transformer's attention cache,
along with provenance metadata (source model, layer count, hidden size,
head count) needed to validate or adapt the cache for a different model.

See :mod:`chen.backends.base` for the protocol that backends implement
to produce/consume KV-caches, and see ``ARCHITECTURE.md`` §5 for the
full transfer protocol spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class KVCache:
    """Per-layer key/value tensors with provenance.

    Attributes:
        keys: Per-layer key tensors. ``keys[i]`` has shape
            ``[seq_len, n_heads, head_dim]``.
        values: Per-layer value tensors, same shape as ``keys``.
        source_model: Model id that produced this cache.
        source_layer_count: Number of layers (``len(keys)``).
        source_hidden_size: ``n_heads * head_dim`` of the source model.
        source_n_heads: Number of attention heads in the source model.
        source_text: The text that was encoded to produce this cache.
            Stored for fallback (re-encoding) and debugging.
        last_token_id: Token id of the last token in the cache. Used by
            some backends to continue generation.
        position: Absolute position of the last token. Used for RoPE.
        metadata: Free-form dict for backend-specific extensions
            (e.g. ``{"vllm_block_table": [...]}``).
    """

    keys: list[np.ndarray]
    values: list[np.ndarray]
    source_model: str
    source_layer_count: int
    source_hidden_size: int
    source_n_heads: int
    source_text: str
    last_token_id: int = 0
    position: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.keys) != len(self.values):
            raise ValueError(
                f"KVCache: keys and values have different layer counts "
                f"({len(self.keys)} vs {len(self.values)})."
            )
        if self.source_layer_count != len(self.keys):
            # Trust the actual length; warn via metadata.
            self.metadata["layer_count_mismatch"] = (
                f"declared={self.source_layer_count}, actual={len(self.keys)}"
            )
            self.source_layer_count = len(self.keys)

    @property
    def seq_len(self) -> int:
        """Sequence length encoded in this cache (tokens)."""
        if not self.keys:
            return 0
        return int(self.keys[0].shape[0]) if self.keys[0].ndim >= 1 else 0

    @property
    def n_heads(self) -> int:
        """Number of attention heads (inferred from the first layer)."""
        if not self.keys or self.keys[0].ndim < 2:
            return 0
        return int(self.keys[0].shape[1])

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        if not self.keys or self.keys[0].ndim < 3:
            return 0
        return int(self.keys[0].shape[2])

    @property
    def bytes_size(self) -> int:
        """Approximate in-memory size of the cache (bytes)."""
        total = 0
        for k in self.keys:
            total += k.nbytes
        for v in self.values:
            total += v.nbytes
        return total

    def is_compatible_with(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
    ) -> bool:
        """Return True if this cache can be used by a model with the given
        layer/head/dim shape without a learned projection."""
        return (
            self.source_layer_count == n_layers
            and self.n_heads == n_heads
            and self.head_dim == head_dim
        )

    def summary(self) -> str:
        """One-line human-readable summary, for logs."""
        return (
            f"KVCache(model={self.source_model!r}, "
            f"layers={self.source_layer_count}, "
            f"heads={self.n_heads}, "
            f"head_dim={self.head_dim}, "
            f"seq_len={self.seq_len}, "
            f"size={self.bytes_size / 1024:.1f} KB)"
        )


class IncompatibleCacheError(Exception):
    """Raised when a KV-cache cannot be transferred to a target backend.

    The pipeline catches this and falls back to text handoff with a
    logged warning. The original exception is preserved as ``__cause__``.
    """

    def __init__(self, message: str, *, cache_source: str = "", target: str = ""):
        super().__init__(message)
        self.cache_source = cache_source
        self.target = target
