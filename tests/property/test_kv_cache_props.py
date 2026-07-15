"""Property-based tests for the KVCache dataclass.

These tests verify invariants that should hold for *any* valid KVCache
construction, regardless of the specific shapes or values.
"""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

from chen.core.kv_cache import KVCache

# Strategies for generating KV-cache shapes.
n_layers_st = st.integers(min_value=1, max_value=12)
seq_len_st = st.integers(min_value=1, max_value=64)
n_heads_st = st.integers(min_value=1, max_value=16)
head_dim_st = st.integers(min_value=8, max_value=128)


def _make_cache(n_layers: int, seq_len: int, n_heads: int, head_dim: int) -> KVCache:
    keys = [np.random.randn(seq_len, n_heads, head_dim).astype(np.float32) for _ in range(n_layers)]
    values = [
        np.random.randn(seq_len, n_heads, head_dim).astype(np.float32) for _ in range(n_layers)
    ]
    return KVCache(
        keys=keys,
        values=values,
        source_model="test",
        source_layer_count=n_layers,
        source_hidden_size=n_heads * head_dim,
        source_n_heads=n_heads,
        source_text="test",
    )


class TestKVCacheProperties:
    @given(n_layers_st, seq_len_st, n_heads_st, head_dim_st)
    def test_keys_and_values_have_same_length(self, n_layers, seq_len, n_heads, head_dim):
        cache = _make_cache(n_layers, seq_len, n_heads, head_dim)
        assert len(cache.keys) == len(cache.values) == n_layers

    @given(n_layers_st, seq_len_st, n_heads_st, head_dim_st)
    def test_seq_len_property_matches_first_layer(self, n_layers, seq_len, n_heads, head_dim):
        cache = _make_cache(n_layers, seq_len, n_heads, head_dim)
        assert cache.seq_len == seq_len

    @given(n_layers_st, seq_len_st, n_heads_st, head_dim_st)
    def test_n_heads_property_matches(self, n_layers, seq_len, n_heads, head_dim):
        cache = _make_cache(n_layers, seq_len, n_heads, head_dim)
        assert cache.n_heads == n_heads

    @given(n_layers_st, seq_len_st, n_heads_st, head_dim_st)
    def test_head_dim_property_matches(self, n_layers, seq_len, n_heads, head_dim):
        cache = _make_cache(n_layers, seq_len, n_heads, head_dim)
        assert cache.head_dim == head_dim

    @given(n_layers_st, seq_len_st, n_heads_st, head_dim_st)
    def test_bytes_size_positive(self, n_layers, seq_len, n_heads, head_dim):
        cache = _make_cache(n_layers, seq_len, n_heads, head_dim)
        assert cache.bytes_size > 0
        # 4 bytes per float32, n_layers * 2 (keys+values) * seq_len * n_heads * head_dim
        expected = n_layers * 2 * seq_len * n_heads * head_dim * 4
        assert cache.bytes_size == expected

    @given(n_layers_st, seq_len_st, n_heads_st, head_dim_st)
    def test_is_compatible_with_itself(self, n_layers, seq_len, n_heads, head_dim):
        cache = _make_cache(n_layers, seq_len, n_heads, head_dim)
        assert cache.is_compatible_with(n_layers, n_heads, head_dim) is True

    @given(n_layers_st, n_layers_st, seq_len_st, n_heads_st, head_dim_st)
    def test_is_compatible_with_different_layers(self, l1, l2, seq_len, n_heads, head_dim):
        assume(l1 != l2)
        cache = _make_cache(l1, seq_len, n_heads, head_dim)
        assert cache.is_compatible_with(l2, n_heads, head_dim) is False

    @given(seq_len_st, n_heads_st, head_dim_st)
    def test_summary_contains_essential_fields(self, seq_len, n_heads, head_dim):
        cache = _make_cache(4, seq_len, n_heads, head_dim)
        s = cache.summary()
        assert "KVCache" in s
        assert "layers=4" in s
        assert "heads=" in s
