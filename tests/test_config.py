"""Tests for config & cost model."""

from __future__ import annotations

import os

import pytest

from chen.core.config import CostModel, Settings


class TestCostModel:
    def test_cost_for_input_only(self):
        cm = CostModel(input_per_1m=1.0, output_per_1m=2.0)
        # 1M input tokens, 0 output -> $1
        assert cm.cost_for(1_000_000, 0) == pytest.approx(1.0)

    def test_cost_for_output_only(self):
        cm = CostModel(input_per_1m=1.0, output_per_1m=2.0)
        # 0 input, 1M output -> $2
        assert cm.cost_for(0, 1_000_000) == pytest.approx(2.0)

    def test_cost_for_mixed(self):
        cm = CostModel(input_per_1m=1.0, output_per_1m=2.0)
        # 500K input + 500K output -> $0.5 + $1 = $1.5
        assert cm.cost_for(500_000, 500_000) == pytest.approx(1.5)

    def test_cost_for_zero_tokens(self):
        cm = CostModel()
        assert cm.cost_for(0, 0) == 0.0

    def test_param_tax_included_when_requested(self):
        cm = CostModel(input_per_1m=0.0, output_per_1m=0.0, per_1m_params_per_1k_tokens=0.001)
        # 70_000M params, 1K tokens -> 70_000 * 1 * 0.001 = 70
        cost = cm.cost_for(500, 500, params_m=70_000, include_param_tax=True)
        assert cost == pytest.approx(70.0, rel=0.01)

    def test_param_tax_excluded_by_default(self):
        cm = CostModel(input_per_1m=0.0, output_per_1m=0.0)
        cost = cm.cost_for(500, 500, params_m=70_000)
        assert cost == 0.0


class TestSettings:
    def test_defaults(self, monkeypatch):
        # Clear all CHEN_ env vars so we get real defaults.
        for k in list(os.environ):
            if k.startswith("CHEN_"):
                monkeypatch.delenv(k, raising=False)
        s = Settings.from_env()
        assert s.default_backend == "mock"
        assert s.memory_backend == "in_memory"
        assert s.router_max_experts == 3

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("CHEN_DEFAULT_BACKEND", "hf")
        monkeypatch.setenv("CHEN_HF_DEVICE", "cuda")
        monkeypatch.setenv("CHEN_ROUTER_MAX_EXPERTS", "5")
        s = Settings.from_env()
        assert s.default_backend == "hf"
        assert s.hf_device == "cuda"
        assert s.router_max_experts == 5

    def test_with_overrides(self):
        s = Settings()
        s2 = s.with_overrides(default_backend="hf")
        assert s2.default_backend == "hf"
        # Original unchanged.
        assert s.default_backend == "mock"

    def test_cost_model_embedded(self):
        s = Settings()
        assert isinstance(s.cost, CostModel)
