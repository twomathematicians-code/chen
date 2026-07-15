"""Tests for the reproducibility utilities."""

from __future__ import annotations

import pytest

from chen.reproducibility import (
    RunContext,
    hash_config,
    seed_everything,
    track_run,
)


class TestHashConfig:
    def test_returns_sha256_hex(self):
        h = hash_config({"phase": 1})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_order_independent(self):
        a = hash_config({"a": 1, "b": 2})
        b = hash_config({"b": 2, "a": 1})
        assert a == b

    def test_different_configs_different_hashes(self):
        a = hash_config({"phase": 1})
        b = hash_config({"phase": 2})
        assert a != b

    def test_handles_nested_dicts(self):
        a = hash_config({"a": {"b": 1}})
        b = hash_config({"a": {"b": 1}})
        assert a == b

    def test_handles_lists(self):
        a = hash_config({"seq": ["a", "b", "c"]})
        b = hash_config({"seq": ["a", "b", "c"]})
        assert a == b

    def test_list_order_matters(self):
        a = hash_config({"seq": ["a", "b"]})
        b = hash_config({"seq": ["b", "a"]})
        assert a != b


class TestSeedEverything:
    def test_does_not_raise(self):
        seed_everything(42)
        seed_everything(0)
        seed_everything(2**31 - 1)

    def test_seeds_python_random(self):
        import random

        seed_everything(42)
        a = random.random()
        seed_everything(42)
        b = random.random()
        assert a == b

    def test_seeds_numpy(self):
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not installed")
        seed_everything(42)
        a = np.random.random()
        seed_everything(42)
        b = np.random.random()
        assert a == b


class TestRunContext:
    def test_from_config(self):
        ctx = RunContext.from_config({"phase": 1, "backend": "mock"})
        assert ctx.config == {"phase": 1, "backend": "mock"}
        assert len(ctx.config_hash) == 64
        assert ctx.seed == 42

    def test_custom_seed(self):
        ctx = RunContext.from_config({"phase": 1}, seed=123)
        assert ctx.seed == 123


class TestTrackRun:
    def test_yields_context(self):
        with track_run({"phase": 1}) as ctx:
            assert isinstance(ctx, RunContext)
            assert ctx.config == {"phase": 1}
            assert len(ctx.config_hash) == 64

    def test_propagates_exception(self):
        with pytest.raises(ValueError, match="boom"):
            with track_run({"phase": 1}):
                raise ValueError("boom")

    def test_seeds_inside_context(self):
        import random

        with track_run({"phase": 1}, seed=42):
            a = random.random()
        with track_run({"phase": 1}, seed=42):
            b = random.random()
        assert a == b
