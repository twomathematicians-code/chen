"""Tests for the Expert wrapper."""

from __future__ import annotations

import pytest

from chen.backends.mock import MockBackend
from chen.core.expert import Expert, ExpertMetrics, ExpertRole


class TestExpertConstruction:
    def test_basic_construction(self, mock_backend_small):
        e = Expert(name="a", role=ExpertRole.ANALYST, backend=mock_backend_small)
        assert e.name == "a"
        assert e.role == ExpertRole.ANALYST
        assert e.backend is mock_backend_small

    def test_default_description_and_tags(self, mock_backend_small):
        e = Expert(name="a", role=ExpertRole.ANALYST, backend=mock_backend_small)
        assert e.description == ""
        assert e.tags == set()

    def test_capabilities_cached(self, mock_backend_small):
        e = Expert(name="a", role=ExpertRole.ANALYST, backend=mock_backend_small)
        cap1 = e.capabilities
        cap2 = e.capabilities
        assert cap1 is cap2

    def test_params_m_delegates_to_backend(self, mock_backend_small, mock_backend_large):
        small = Expert(name="s", role=ExpertRole.ANALYST, backend=mock_backend_small)
        large = Expert(name="l", role=ExpertRole.REASONER, backend=mock_backend_large)
        assert small.params_m == 3_000
        assert large.params_m == 8_000


class TestExpertInvokeText:
    def test_invoke_with_prompt_returns_output_and_metrics(self, mock_backend_small):
        e = Expert(name="a", role=ExpertRole.ANALYST, backend=mock_backend_small)
        out, cache, m = e.invoke(prompt="hello world", max_tokens=64)
        assert isinstance(out, str)
        assert len(out) > 0
        assert cache is not None  # MockBackend produces an output cache
        assert isinstance(m, ExpertMetrics)
        assert m.expert_name == "a"
        assert m.role == ExpertRole.ANALYST
        assert m.params_m == 3_000
        assert m.input_tokens > 0
        assert m.output_tokens > 0
        assert m.latency_ms >= 0
        assert m.used_kv_cache is False

    def test_invoke_requires_prompt_or_cache(self, mock_backend_small):
        e = Expert(name="a", role=ExpertRole.ANALYST, backend=mock_backend_small)
        with pytest.raises(ValueError, match="one of `prompt` or `cache`"):
            e.invoke()

    def test_invoke_is_deterministic(self, mock_backend_small):
        e = Expert(name="a", role=ExpertRole.ANALYST, backend=mock_backend_small)
        a, _, _ = e.invoke(prompt="hello world")
        b, _, _ = e.invoke(prompt="hello world")
        assert a == b


class TestExpertInvokeKVCache:
    def test_invoke_with_cache_uses_kv_path(self, mock_backend_small):
        e = Expert(name="a", role=ExpertRole.ANALYST, backend=mock_backend_small)
        cache = mock_backend_small.encode("hello world")
        out, out_cache, m = e.invoke(cache=cache, max_tokens=64)
        assert isinstance(out, str)
        assert m.used_kv_cache is True
        assert m.cache_transfer_succeeded is True
        assert m.cache_transfer_ms >= 0

    def test_invoke_falls_back_to_text_on_unsupported_kv(self):
        """If the backend doesn't support KV-cache, expert should use text."""
        from chen.backends.base import BackendCapabilities

        class TextOnlyBackend(MockBackend):
            @property
            def capabilities(self) -> BackendCapabilities:
                return BackendCapabilities(
                    supports_kv_cache=False,
                    deterministic=True,
                )

        backend = TextOnlyBackend(params_m=3_000, role_hint="text-only")
        e = Expert(name="a", role=ExpertRole.ANALYST, backend=backend)
        # Pass a cache; expert should detect unsupported and use source_text.
        other_cache = MockBackend(params_m=3_000).encode("hello world")
        out, _, m = e.invoke(cache=other_cache)
        assert isinstance(out, str)
        # used_kv_cache should be False because we fell back to text.
        assert m.used_kv_cache is False


class TestExpertMetrics:
    def test_total_tokens(self):
        m = ExpertMetrics(
            expert_name="a",
            role=ExpertRole.ANALYST,
            params_m=3_000,
            input_tokens=10,
            output_tokens=20,
            latency_ms=5.0,
        )
        assert m.total_tokens == 30
