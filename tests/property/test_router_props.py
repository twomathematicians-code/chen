"""Property-based tests for the router.

Verifies that router outputs satisfy basic invariants regardless of
the prompt or expert pool composition.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from chen.backends.mock import MockBackend
from chen.core.expert import Expert, ExpertRole
from chen.core.router import (
    CosineRouter,
    HybridRouter,
    LogisticRouter,
)

# Strategies for prompt generation.
word_st = st.text(
    alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
    min_size=1,
    max_size=10,
)
prompt_st = st.lists(word_st, min_size=1, max_size=20).map(" ".join)


def _build_random_experts(n: int) -> list[Expert]:
    """Build n experts with random roles."""
    roles = list(ExpertRole)
    experts = []
    for i in range(n):
        experts.append(
            Expert(
                name=f"expert_{i}",
                role=roles[i % len(roles)],
                backend=MockBackend(params_m=3_000, role_hint=f"e{i}"),
            )
        )
    return experts


class TestRouterInvariants:
    @given(prompt_st, st.integers(min_value=1, max_value=8))
    def test_logistic_always_returns_at_least_one_expert(self, prompt, n_experts):
        experts = _build_random_experts(n_experts)
        # Ensure at least one synthesizer (force_last_role expects one)
        if not any(e.role == ExpertRole.SYNTHESIZER for e in experts):
            experts[0] = Expert(
                name=experts[0].name,
                role=ExpertRole.SYNTHESIZER,
                backend=experts[0].backend,
            )
        router = LogisticRouter.from_experts(experts)
        selected = router.route(prompt, experts)
        assert len(selected) >= 1

    @given(prompt_st, st.integers(min_value=2, max_value=8))
    def test_logistic_respects_max_activation(self, prompt, n_experts):
        experts = _build_random_experts(n_experts)
        if not any(e.role == ExpertRole.SYNTHESIZER for e in experts):
            experts[0] = Expert(
                name=experts[0].name,
                role=ExpertRole.SYNTHESIZER,
                backend=experts[0].backend,
            )
        router = LogisticRouter.from_experts(experts, max_activation=2)
        selected = router.route(prompt, experts)
        assert len(selected) <= 2

    @given(prompt_st, st.integers(min_value=2, max_value=8))
    def test_logistic_last_expert_is_synthesizer_when_available(self, prompt, n_experts):
        experts = _build_random_experts(n_experts)
        # Ensure there's a synthesizer in the pool.
        experts[-1] = Expert(
            name="synth",
            role=ExpertRole.SYNTHESIZER,
            backend=MockBackend(params_m=3_000, role_hint="synth"),
        )
        router = LogisticRouter.from_experts(experts)
        selected = router.route(prompt, experts)
        # Last selected should be a synthesizer.
        last_name = selected[-1]
        last_expert = next(e for e in experts if e.name == last_name)
        assert last_expert.role == ExpertRole.SYNTHESIZER

    @given(prompt_st, st.integers(min_value=1, max_value=8))
    def test_cosine_always_returns_at_least_one_expert(self, prompt, n_experts):
        experts = _build_random_experts(n_experts)
        if not any(e.role == ExpertRole.SYNTHESIZER for e in experts):
            experts[0] = Expert(
                name=experts[0].name,
                role=ExpertRole.SYNTHESIZER,
                backend=experts[0].backend,
            )
        router = CosineRouter.from_experts(experts)
        selected = router.route(prompt, experts)
        assert len(selected) >= 1

    @given(prompt_st, st.integers(min_value=1, max_value=8))
    def test_hybrid_always_returns_at_least_one_expert(self, prompt, n_experts):
        experts = _build_random_experts(n_experts)
        if not any(e.role == ExpertRole.SYNTHESIZER for e in experts):
            experts[0] = Expert(
                name=experts[0].name,
                role=ExpertRole.SYNTHESIZER,
                backend=experts[0].backend,
            )
        router = HybridRouter.from_experts(experts)
        selected = router.route(prompt, experts)
        assert len(selected) >= 1

    @given(prompt_st)
    def test_router_output_only_contains_known_expert_names(self, prompt):
        experts = _build_random_experts(4)
        experts.append(
            Expert(
                name="synth",
                role=ExpertRole.SYNTHESIZER,
                backend=MockBackend(params_m=3_000, role_hint="synth"),
            )
        )
        known_names = {e.name for e in experts}
        for router_cls in [LogisticRouter, CosineRouter, HybridRouter]:
            router = router_cls.from_experts(experts)
            selected = router.route(prompt, experts)
            for name in selected:
                assert name in known_names, f"Router returned unknown name: {name}"

    @given(prompt_st)
    def test_deterministic_routing(self, prompt):
        """Same prompt + same experts → same routing decision."""
        experts = _build_random_experts(4)
        experts.append(
            Expert(
                name="synth",
                role=ExpertRole.SYNTHESIZER,
                backend=MockBackend(params_m=3_000, role_hint="synth"),
            )
        )
        router = LogisticRouter.from_experts(experts)
        a = router.route(prompt, experts)
        b = router.route(prompt, experts)
        assert a == b
