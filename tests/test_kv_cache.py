"""Tests for the KV-cache dataclass."""

from __future__ import annotations

import numpy as np
import pytest

from chen.core.kv_cache import IncompatibleCacheError, KVCache


def _make_cache(
    n_layers: int = 4,
    seq_len: int = 10,
    n_heads: int = 8,
    head_dim: int = 64,
    source_model: str = "test-model",
) -> KVCache:
    keys = [np.zeros((seq_len, n_heads, head_dim), dtype=np.float32) for _ in range(n_layers)]
    values = [np.zeros((seq_len, n_heads, head_dim), dtype=np.float32) for _ in range(n_layers)]
    return KVCache(
        keys=keys,
        values=values,
        source_model=source_model,
        source_layer_count=n_layers,
        source_hidden_size=n_heads * head_dim,
        source_n_heads=n_heads,
        source_text="test prompt",
    )


class TestKVCacheConstruction:
    def test_basic_construction(self):
        c = _make_cache()
        assert c.source_layer_count == 4
        assert c.seq_len == 10
        assert c.n_heads == 8
        assert c.head_dim == 64

    def test_mismatched_keys_values_raises(self):
        with pytest.raises(ValueError, match="different layer counts"):
            KVCache(
                keys=[np.zeros((1, 1, 1))],
                values=[np.zeros((1, 1, 1)), np.zeros((1, 1, 1))],
                source_model="m",
                source_layer_count=1,
                source_hidden_size=1,
                source_n_heads=1,
                source_text="",
            )

    def test_layer_count_mismatch_logs_in_metadata(self):
        c = KVCache(
            keys=[np.zeros((1, 1, 1)), np.zeros((1, 1, 1))],
            values=[np.zeros((1, 1, 1)), np.zeros((1, 1, 1))],
            source_model="m",
            source_layer_count=5,  # wrong
            source_hidden_size=1,
            source_n_heads=1,
            source_text="",
        )
        assert c.source_layer_count == 2  # auto-corrected
        assert "layer_count_mismatch" in c.metadata


class TestKVCacheProperties:
    def test_bytes_size(self):
        c = _make_cache(n_layers=4, seq_len=10, n_heads=8, head_dim=64)
        # 4 layers * 2 (keys+values) * 10 * 8 * 64 * 4 bytes = 163840 bytes
        assert c.bytes_size == 4 * 2 * 10 * 8 * 64 * 4

    def test_is_compatible_with_matching(self):
        c = _make_cache(n_layers=4, n_heads=8, head_dim=64)
        assert c.is_compatible_with(4, 8, 64) is True

    def test_is_compatible_with_mismatching(self):
        c = _make_cache(n_layers=4, n_heads=8, head_dim=64)
        assert c.is_compatible_with(8, 8, 64) is False
        assert c.is_compatible_with(4, 16, 64) is False
        assert c.is_compatible_with(4, 8, 32) is False

    def test_summary_is_human_readable(self):
        c = _make_cache()
        s = c.summary()
        assert "KVCache" in s
        assert "model=" in s
        assert "layers=4" in s


class TestIncompatibleCacheError:
    def test_raises_with_message(self):
        with pytest.raises(IncompatibleCacheError, match="cannot transfer"):
            raise IncompatibleCacheError("cannot transfer cache: shape mismatch")

    def test_preserves_source_and_target(self):
        try:
            raise IncompatibleCacheError("x", cache_source="model-a", target="model-b")
        except IncompatibleCacheError as e:
            assert e.cache_source == "model-a"
            assert e.target == "model-b"
