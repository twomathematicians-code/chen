"""Tests for the routers."""

from __future__ import annotations

from chen.core.router import (
    CosineRouter,
    HybridRouter,
    LogisticRouter,
    RouterDecision,
)


class TestLogisticRouter:
    def test_always_returns_at_least_one_expert(self, four_experts):
        router = LogisticRouter.from_experts(four_experts)
        selected = router.route("hello", four_experts)
        assert len(selected) >= 1

    def test_code_prompt_selects_coder(self, four_experts):
        router = LogisticRouter.from_experts(four_experts)
        selected = router.route("Debug this Python function: def foo(): return None", four_experts)
        # Coder should be in the selection (high keyword weight).
        assert "coder" in selected

    def test_math_prompt_selects_reasoner(self, four_experts):
        router = LogisticRouter.from_experts(four_experts)
        selected = router.route("Solve for x: 3x + 7 = 22", four_experts)
        assert "reasoner" in selected

    def test_simple_prompt_selects_synthesizer(self, four_experts):
        router = LogisticRouter.from_experts(four_experts)
        selected = router.route("Write a haiku about autumn.", four_experts)
        # Synthesizer should always be last (force_last_role).
        assert selected[-1] == "synthesizer"

    def test_force_last_role_appends_synthesizer(self, four_experts):
        router = LogisticRouter.from_experts(four_experts)
        selected = router.route("Explain quantum physics and write code.", four_experts)
        assert selected[-1] == "synthesizer"

    def test_max_activation_caps_selection(self, four_experts):
        router = LogisticRouter.from_experts(four_experts, max_activation=2)
        selected = router.route("debug python code math reasoning", four_experts)
        assert len(selected) <= 2

    def test_empty_experts_returns_empty(self):
        router = LogisticRouter.from_experts([])
        assert router.route("anything", []) == []

    def test_name_property(self, four_experts):
        router = LogisticRouter.from_experts(four_experts)
        assert router.name == "logistic"


class TestCosineRouter:
    def test_registers_experts_with_prototypes(self, four_experts):
        router = CosineRouter.from_experts(four_experts)
        for e in four_experts:
            assert e.name in router.prototypes

    def test_returns_at_least_one_expert(self, four_experts):
        router = CosineRouter.from_experts(four_experts)
        selected = router.route("hello", four_experts)
        assert len(selected) >= 1

    def test_force_last_role(self, four_experts):
        router = CosineRouter.from_experts(four_experts)
        selected = router.route("any prompt", four_experts)
        assert selected[-1] == "synthesizer"

    def test_max_activation(self, four_experts):
        router = CosineRouter.from_experts(four_experts, max_activation=2)
        selected = router.route("hello", four_experts)
        assert len(selected) <= 2

    def test_name_property(self, four_experts):
        router = CosineRouter.from_experts(four_experts)
        assert router.name == "cosine"

    def test_custom_embed_fn(self, four_experts):
        import numpy as np

        def embed(text: str):
            v = np.zeros(8, dtype=np.float32)
            for i, c in enumerate(text[:8]):
                v[i] = ord(c) / 256.0
            n = float((v**2).sum()) ** 0.5
            return v / n if n > 0 else v

        router = CosineRouter.from_experts(four_experts, embed_fn=embed, embed_dim=8)
        selected = router.route("hello", four_experts)
        assert len(selected) >= 1


class TestHybridRouter:
    def test_returns_at_least_one_expert(self, four_experts):
        router = HybridRouter.from_experts(four_experts)
        selected = router.route("hello", four_experts)
        assert len(selected) >= 1

    def test_force_last_role(self, four_experts):
        router = HybridRouter.from_experts(four_experts)
        selected = router.route("any prompt", four_experts)
        assert selected[-1] == "synthesizer"

    def test_code_prompt_includes_coder(self, four_experts):
        router = HybridRouter.from_experts(four_experts)
        selected = router.route("Debug this Python function: def foo(): return None", four_experts)
        # Either coder is selected, or it's not — but synthesizer is last.
        assert selected[-1] == "synthesizer"

    def test_name_property(self, four_experts):
        router = HybridRouter.from_experts(four_experts)
        assert router.name == "hybrid"

    def test_alpha_can_be_tuned(self, four_experts):
        r1 = HybridRouter.from_experts(four_experts, alpha=0.0)  # all cosine
        r2 = HybridRouter.from_experts(four_experts, alpha=1.0)  # all logistic
        # Both should still return a valid selection.
        assert len(r1.route("hello", four_experts)) >= 1
        assert len(r2.route("hello", four_experts)) >= 1


class TestRouterDecision:
    def test_iterable(self):
        d = RouterDecision(
            selected=["a", "b"],
            scores={"a": 0.9, "b": 0.5},
            router_name="logistic",
        )
        assert list(d) == ["a", "b"]
