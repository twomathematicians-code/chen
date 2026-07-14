"""Deterministic mock backend for tests and CPU-only demos.

The MockBackend exists so that the entire CHEN pipeline — including Phase 2
KV-cache handoff — runs in under five seconds on a CPU, with no model
downloads and no GPU. Every unit test in the repo uses this backend.

It is **not** a real language model. Generation is a deterministic function
of the prompt and a fixed random seed: the prompt is hashed to a fixed-dim
embedding, the embedding is run through a tiny two-layer FFN, and the
output is a deterministic template-based string that contains enough
structure for downstream experts to operate on.

The KV-cache is real in the sense that it is a numpy array of the right
shape, transferred faithfully between backends, and decodable back to
text — it just isn't a transformer's KV-cache. This is exactly what we
need to test the routing, pipeline, and KPI machinery without depending
on a 4GB model download.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from chen.backends.base import BackendCapabilities
from chen.core.kv_cache import KVCache


def _stable_hash(text: str, seed: int = 0) -> bytes:
    """Return a stable 32-byte hash of ``text`` + ``seed``."""
    h = hashlib.blake2b(digest_size=32, key=seed.to_bytes(8, "little"))
    h.update(text.encode("utf-8"))
    return h.digest()


def _hash_to_embedding(text: str, dim: int, seed: int = 0) -> np.ndarray:
    """Deterministically project text to a fixed-dim float32 vector in [0, 1)."""
    out = np.zeros(dim, dtype=np.float32)
    # Walk the prompt in 8-byte chunks, hash each chunk with position,
    # and fold into the embedding.
    encoded = text.encode("utf-8")
    for i in range(0, max(len(encoded), 1), 8):
        chunk = encoded[i : i + 8]
        h = hashlib.blake2b(chunk, digest_size=8, key=(seed + i).to_bytes(8, "little"))
        idx = (i // 8) % dim
        out[idx] = (int.from_bytes(h.digest(), "little") % 10000) / 10000.0
    return out


@dataclass
class MockBackend:
    """Deterministic mock backend for tests and CPU-only demos.

    Attributes:
        params_m: Model size in millions of parameters. Used for cost calcs.
        model_id: A stable identifier (defaults to ``mock-<params_m>B``).
        n_layers: Number of fake attention layers in the KV-cache.
        n_heads: Number of fake attention heads per layer.
        head_dim: Dimension of each fake head.
        seed: Random seed for deterministic generation.
        role_hint: Optional role tag (e.g. "analyst") that biases the
            generated output so different experts produce visibly different
            text. This makes the cascade observable in tests.
    """

    params_m: int = 3_000
    model_id: str = ""
    n_layers: int = 4
    n_heads: int = 8
    head_dim: int = 64
    seed: int = 42
    role_hint: str = "expert"

    _cache: dict[str, KVCache] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.model_id:
            self.model_id = f"mock-{self.params_m // 1000}B-{self.role_hint}"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_kv_cache=True,
            supports_streaming=False,
            supports_batching=False,
            deterministic=True,
        )

    # ------------------------------------------------------------------
    # Text-level generation (Phase 1)
    # ------------------------------------------------------------------
    def generate(self, prompt: str, max_tokens: int = 256, **kwargs: Any) -> str:
        """Return a deterministic, structured mock response.

        The response includes:
          - The role hint (so you can see which expert spoke).
          - A short hash of the prompt (so you can verify determinism).
          - The first 80 chars of the prompt (so downstream experts see it).
          - A fake "reasoning chain" of length proportional to ``params_m``.

        This is intentionally not a real LLM — but it has enough structure
        that the cascade and KPI machinery can be exercised end-to-end.
        """
        del max_tokens, kwargs  # unused — output is template-based
        emb = _hash_to_embedding(prompt, dim=32, seed=self.seed)
        prompt_hash = hashlib.blake2b(prompt.encode("utf-8"), digest_size=4).hexdigest()
        reasoning_steps = max(1, self.params_m // 2_000)
        steps = "\n".join(
            f"  step {i + 1}: latent[{i}] = {emb[i % 32]:.4f}" for i in range(reasoning_steps)
        )
        snippet = prompt.strip().replace("\n", " ")[:80]
        return (
            f"[{self.role_hint}@{self.params_m // 1000}B] "
            f"prompt_id={prompt_hash}\n"
            f"input: {snippet!r}\n"
            f"reasoning ({reasoning_steps} steps):\n{steps}\n"
            f"output: processed by {self.model_id}"
        )

    # ------------------------------------------------------------------
    # Latent-level operations (Phase 2)
    # ------------------------------------------------------------------
    def encode(self, prompt: str, **kwargs: Any) -> KVCache:
        """Run a fake prefill: hash the prompt into per-layer K/V tensors."""
        del kwargs
        seq_len = max(1, len(prompt) // 4)  # ~4 chars per token
        keys: list[np.ndarray] = []
        values: list[np.ndarray] = []
        for layer in range(self.n_layers):
            k = _hash_to_embedding(
                f"{prompt}|layer={layer}|kind=key",
                dim=self.n_heads * self.head_dim * seq_len,
                seed=self.seed + layer,
            ).reshape(seq_len, self.n_heads, self.head_dim)
            v = _hash_to_embedding(
                f"{prompt}|layer={layer}|kind=val",
                dim=self.n_heads * self.head_dim * seq_len,
                seed=self.seed + layer + 1000,
            ).reshape(seq_len, self.n_heads, self.head_dim)
            keys.append(k.astype(np.float32))
            values.append(v.astype(np.float32))
        cache = KVCache(
            keys=keys,
            values=values,
            source_model=self.model_id,
            source_layer_count=self.n_layers,
            source_hidden_size=self.n_heads * self.head_dim,
            source_n_heads=self.n_heads,
            source_text=prompt,
            last_token_id=int.from_bytes(
                hashlib.blake2b(prompt.encode(), digest_size=4).digest(), "little"
            )
            % 32_000,
            position=seq_len - 1,
        )
        self._cache[prompt] = cache
        return cache

    def decode(self, cache: KVCache, max_tokens: int = 256, **kwargs: Any) -> str:
        """Recover text from a KV-cache.

        For the mock backend, "decoding" means: hash the cache's source_text
        together with the cache's provenance, and produce a structured
        response. If the cache was transferred from another mock backend,
        the response notes that the cache came from a different model.
        """
        del max_tokens, kwargs
        transferred = cache.source_model != self.model_id
        suffix = f" (transferred from {cache.source_model})" if transferred else ""
        return (
            f"[{self.role_hint}@{self.params_m // 1000}B] "
            f"decoded from cache ({cache.source_layer_count} layers, "
            f"seq_len={cache.seq_len}){suffix}\n"
            f"source: {cache.source_text[:80]!r}\n"
            f"output: decoded by {self.model_id}"
        )

    def transfer_cache(self, cache: KVCache) -> KVCache:
        """Adapt a KV-cache from another mock backend for use by this one.

        For same-shape backends (matching ``n_layers``, ``n_heads``,
        ``head_dim``) this is a no-op — the cache is returned unchanged
        (with provenance preserved so the receiver knows it was transferred).

        For different-shape backends we *re-encode* the source text with
        this backend's shape. This is the mock-backend analog of a learned
        cross-family projection: it loses the original latent (because the
        mock's latents are arbitrary anyway) but preserves the source text,
        which is what downstream decoding needs.
        """
        same_shape = (
            cache.source_layer_count == self.n_layers
            and cache.source_n_heads == self.n_heads
            and cache.source_hidden_size == self.n_heads * self.head_dim
        )
        if same_shape:
            # No-op transfer — pass the cache through with provenance intact.
            return cache
        # Shape mismatch — re-encode source text with our shape.
        return self.encode(cache.source_text)

    # ------------------------------------------------------------------
    # Tokenization (mocked)
    # ------------------------------------------------------------------
    def count_tokens(self, text: str) -> int:
        """Cheap deterministic token count (~4 chars per token)."""
        return max(1, len(text) // 4)
