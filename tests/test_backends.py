"""Tests for the MockBackend."""

from __future__ import annotations

from chen.backends.mock import MockBackend
from chen.core.kv_cache import KVCache


class TestMockBackendBasic:
    def test_generate_returns_nonempty_string(self, mock_backend_small):
        out = mock_backend_small.generate("hello world")
        assert isinstance(out, str)
        assert len(out) > 0

    def test_generate_is_deterministic(self, mock_backend_small):
        a = mock_backend_small.generate("hello world")
        b = mock_backend_small.generate("hello world")
        assert a == b

    def test_generate_different_prompts_give_different_outputs(self, mock_backend_small):
        a = mock_backend_small.generate("hello world")
        b = mock_backend_small.generate("goodbye world")
        assert a != b

    def test_generate_includes_role_hint(self, mock_backend_small):
        out = mock_backend_small.generate("test prompt")
        assert "analyst" in out.lower()

    def test_generate_includes_params_hint(self, mock_backend_large):
        out = mock_backend_large.generate("test prompt")
        assert "8B" in out

    def test_count_tokens_positive(self, mock_backend_small):
        n = mock_backend_small.count_tokens("hello world")
        assert n > 0
        assert isinstance(n, int)


class TestMockBackendCapabilities:
    def test_supports_kv_cache(self, mock_backend_small):
        assert mock_backend_small.capabilities.supports_kv_cache is True

    def test_deterministic(self, mock_backend_small):
        assert mock_backend_small.capabilities.deterministic is True

    def test_params_m(self, mock_backend_small, mock_backend_large):
        assert mock_backend_small.params_m == 3_000
        assert mock_backend_large.params_m == 8_000

    def test_model_id_auto_generated(self):
        b = MockBackend(params_m=3_000, role_hint="analyst")
        assert "3B" in b.model_id
        assert "analyst" in b.model_id


class TestMockBackendKVCache:
    def test_encode_returns_kv_cache(self, mock_backend_small):
        cache = mock_backend_small.encode("hello world")
        assert isinstance(cache, KVCache)
        assert cache.source_model == mock_backend_small.model_id
        assert cache.source_text == "hello world"
        assert len(cache.keys) == mock_backend_small.n_layers
        assert len(cache.values) == mock_backend_small.n_layers

    def test_cache_seq_len_proportional_to_prompt(self, mock_backend_small):
        short = mock_backend_small.encode("hi")
        long = mock_backend_small.encode("hello " * 100)
        assert long.seq_len > short.seq_len

    def test_decode_from_own_cache(self, mock_backend_small):
        cache = mock_backend_small.encode("test prompt")
        out = mock_backend_small.decode(cache)
        assert isinstance(out, str)
        assert "decoded" in out.lower()

    def test_decode_marks_transferred_cache(self, mock_backend_small, mock_backend_large):
        cache = mock_backend_small.encode("test prompt")
        # Transfer from small to large: shapes differ -> re-encode path.
        transferred = mock_backend_large.transfer_cache(cache)
        out = mock_backend_large.decode(transferred)
        assert "transferred" in out.lower()

    def test_transfer_cache_same_shape_is_noop(self):
        a = MockBackend(params_m=3_000, role_hint="a", n_layers=4, n_heads=8, head_dim=64)
        b = MockBackend(params_m=3_000, role_hint="b", n_layers=4, n_heads=8, head_dim=64)
        cache = a.encode("test prompt")
        transferred = b.transfer_cache(cache)
        # Same shape -> no re-encoding, cache returned as-is.
        assert transferred is cache or transferred.source_text == cache.source_text

    def test_transfer_cache_different_shape_re_encodes(self):
        a = MockBackend(params_m=3_000, role_hint="a", n_layers=4, n_heads=8, head_dim=64)
        b = MockBackend(params_m=8_000, role_hint="b", n_layers=8, n_heads=16, head_dim=32)
        cache = a.encode("test prompt")
        transferred = b.transfer_cache(cache)
        # Re-encoded to match b's shape.
        assert transferred.source_layer_count == b.n_layers
        assert transferred.source_n_heads == b.n_heads


class TestMockBackendEdgeCases:
    def test_empty_prompt(self, mock_backend_small):
        out = mock_backend_small.generate("")
        assert isinstance(out, str)
        # Should not raise.
        cache = mock_backend_small.encode("")
        assert cache.seq_len >= 1  # at least one position

    def test_unicode_prompt(self, mock_backend_small):
        out = mock_backend_small.generate("héllo 世界 🌍")
        assert isinstance(out, str)

    def test_very_long_prompt(self, mock_backend_small):
        long_prompt = "word " * 10_000
        out = mock_backend_small.generate(long_prompt)
        assert isinstance(out, str)
        cache = mock_backend_small.encode(long_prompt)
        assert cache.seq_len > 1000
