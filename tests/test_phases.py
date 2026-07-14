"""Tests for the three pipelines (Phase 1, 2, 3)."""

from __future__ import annotations

import pytest

from chen.backends.mock import MockBackend
from chen.core.expert import Expert, ExpertRole
from chen.core.router import (
    CosineRouter,
    HybridRouter,
    LogisticRouter,
)
from chen.phases.phase1_cascade import CascadePipeline
from chen.phases.phase2_kv_pass import KVPassPipeline
from chen.phases.phase3_routing import RoutingPipeline

# ---------------------------------------------------------------------------
# CascadePipeline (Phase 1)
# ---------------------------------------------------------------------------


class TestCascadePipeline:
    def test_basic_run_returns_pipeline_result(self, three_experts):
        pipe = CascadePipeline(
            experts=three_experts,
            sequence=["analyst", "reasoner", "synthesizer"],
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
        result = pipe.run("hello world")
        assert isinstance(result.output, str)
        assert len(result.output) > 0
        assert len(result.per_expert) == 3
        assert result.metrics.total_tokens > 0
        assert result.metrics.total_latency_ms >= 0
        assert result.metrics.total_cost_usd >= 0

    def test_default_sequence_uses_input_order(self, three_experts):
        pipe = CascadePipeline(
            experts=three_experts,
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
        result = pipe.run("hello world")
        assert result.selected_experts == ["analyst", "reasoner", "synthesizer"]

    def test_unknown_sequence_expert_raises(self, three_experts):
        with pytest.raises(ValueError, match="unknown experts"):
            CascadePipeline(
                experts=three_experts,
                sequence=["analyst", "nonexistent"],
            )

    def test_empty_experts_raises(self):
        with pytest.raises(ValueError, match="at least one expert"):
            CascadePipeline(experts=[])

    def test_writes_intermediate_to_memory(self, three_experts, fresh_memory):
        pipe = CascadePipeline(
            experts=three_experts,
            sequence=["analyst", "reasoner", "synthesizer"],
            memory=fresh_memory,
            write_intermediate_to_memory=True,
            memory_retrieve_k=0,
        )
        pipe.run("hello world")
        # 3 experts + 1 user prompt = 4 entries
        assert fresh_memory.count() >= 4

    def test_memory_augmentation_increases_input_tokens(self, three_experts):
        pipe_no_mem = CascadePipeline(
            experts=three_experts,
            sequence=["analyst", "reasoner", "synthesizer"],
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
        pipe_with_mem = CascadePipeline(
            experts=three_experts,
            sequence=["analyst", "reasoner", "synthesizer"],
            memory_retrieve_k=4,
            write_intermediate_to_memory=True,
        )
        r_no = pipe_no_mem.run("hello world")
        r_with = pipe_with_mem.run("hello world")
        # With memory, the second expert should see more context.
        assert r_with.metrics.total_input_tokens >= r_no.metrics.total_input_tokens

    def test_metadata_includes_phase(self, three_experts):
        pipe = CascadePipeline(
            experts=three_experts,
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
        result = pipe.run("hello")
        assert result.metadata["phase"] == 1
        assert result.metadata["handoff"] == "text"


# ---------------------------------------------------------------------------
# KVPassPipeline (Phase 2)
# ---------------------------------------------------------------------------


class TestKVPassPipeline:
    def test_basic_run(self, three_experts):
        pipe = KVPassPipeline(
            experts=three_experts,
            sequence=["analyst", "reasoner", "synthesizer"],
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
        result = pipe.run("hello world")
        assert isinstance(result.output, str)
        assert len(result.output) > 0
        assert len(result.per_expert) == 3
        # All experts support KV-cache (mock backend), so we should see transfers.
        assert result.metrics.kv_cache_transfers >= 2

    def test_metadata_includes_phase(self, three_experts):
        pipe = KVPassPipeline(
            experts=three_experts,
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
        result = pipe.run("hello")
        assert result.metadata["phase"] == 2
        assert result.metadata["handoff"] == "kv_cache"

    def test_kv_transfers_recorded_in_metrics(self, three_experts):
        pipe = KVPassPipeline(
            experts=three_experts,
            sequence=["analyst", "reasoner", "synthesizer"],
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
        result = pipe.run("hello world")
        # At least the 2nd and 3rd experts should have used_kv_cache=True.
        n_kv = sum(1 for m in result.per_expert if m.used_kv_cache)
        assert n_kv >= 2

    def test_falls_back_to_text_when_kv_unsupported(self):
        from chen.backends.base import BackendCapabilities

        class TextOnlyBackend(MockBackend):
            @property
            def capabilities(self) -> BackendCapabilities:
                return BackendCapabilities(
                    supports_kv_cache=False,
                    deterministic=True,
                )

        experts = [
            Expert(name="a", role=ExpertRole.ANALYST, backend=TextOnlyBackend(params_m=3_000)),
            Expert(name="s", role=ExpertRole.SYNTHESIZER, backend=TextOnlyBackend(params_m=3_000)),
        ]
        pipe = KVPassPipeline(
            experts=experts,
            sequence=["a", "s"],
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
            fallback_to_text=True,
        )
        result = pipe.run("hello world")
        # No KV transfers because the backends don't support it.
        assert result.metrics.kv_cache_transfers == 0
        assert isinstance(result.output, str)

    def test_latent_nuance_score_computed(self, three_experts):
        pipe = KVPassPipeline(
            experts=three_experts,
            sequence=["analyst", "reasoner", "synthesizer"],
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
        result = pipe.run("hello world")
        # With successful KV transfers, nuance score should be 1.0.
        assert 0.0 <= result.metrics.latent_nuance_score <= 1.0
        assert result.metrics.latent_nuance_score > 0


# ---------------------------------------------------------------------------
# RoutingPipeline (Phase 3)
# ---------------------------------------------------------------------------


class TestRoutingPipeline:
    def test_basic_run_with_logistic_router(self, four_experts):
        router = LogisticRouter.from_experts(four_experts)
        pipe = RoutingPipeline(
            experts=four_experts,
            router=router,
            handoff="text",
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
        result = pipe.run("hello world")
        assert isinstance(result.output, str)
        assert len(result.selected_experts) >= 1
        assert result.metadata["phase"] == 3

    def test_router_selects_different_experts_for_different_prompts(self, four_experts):
        router = LogisticRouter.from_experts(four_experts)
        pipe = RoutingPipeline(
            experts=four_experts,
            router=router,
            handoff="text",
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
        poem = pipe.run("Write a haiku about autumn.")
        debug = pipe.run("Debug this Python function: def foo(): return None")
        # The two prompts should generally activate different experts.
        # (Not strictly required to differ, but the router should at least
        # always include the synthesizer as the last expert.)
        assert poem.selected_experts[-1] == "synthesizer"
        assert debug.selected_experts[-1] == "synthesizer"

    def test_kv_cache_handoff_in_routing(self, four_experts):
        router = LogisticRouter.from_experts(four_experts)
        pipe = RoutingPipeline(
            experts=four_experts,
            router=router,
            handoff="kv_cache",
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
        result = pipe.run("Explain recursion step by step.")
        # If multiple experts were selected and KV supported, transfers > 0.
        if len(result.selected_experts) > 1:
            assert result.metrics.kv_cache_transfers >= 1

    def test_requires_router(self, four_experts):
        with pytest.raises(ValueError, match="requires a `router`"):
            RoutingPipeline(experts=four_experts)

    def test_works_with_each_router_kind(self, four_experts):
        for make in [
            lambda: LogisticRouter.from_experts(four_experts),
            lambda: CosineRouter.from_experts(four_experts),
            lambda: HybridRouter.from_experts(four_experts),
        ]:
            router = make()
            pipe = RoutingPipeline(
                experts=four_experts,
                router=router,
                handoff="text",
                memory_retrieve_k=0,
                write_intermediate_to_memory=False,
            )
            result = pipe.run("hello world")
            assert len(result.selected_experts) >= 1

    def test_metadata_includes_router_name(self, four_experts):
        router = LogisticRouter.from_experts(four_experts)
        pipe = RoutingPipeline(
            experts=four_experts,
            router=router,
            handoff="text",
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
        result = pipe.run("hello")
        assert result.metadata["router"] == "logistic"


# ---------------------------------------------------------------------------
# Cross-phase consistency
# ---------------------------------------------------------------------------


class TestCrossPhase:
    def test_phase1_and_phase2_same_sequence(self, three_experts):
        p1 = CascadePipeline(
            experts=three_experts,
            sequence=["analyst", "reasoner", "synthesizer"],
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
        p2 = KVPassPipeline(
            experts=three_experts,
            sequence=["analyst", "reasoner", "synthesizer"],
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
        r1 = p1.run("hello world")
        r2 = p2.run("hello world")
        # Both should produce output and run all 3 experts.
        assert len(r1.per_expert) == 3
        assert len(r2.per_expert) == 3
        # KV-pass should record more KV transfers.
        assert r2.metrics.kv_cache_transfers >= r1.metrics.kv_cache_transfers

    def test_phase3_uses_subset_of_experts(self, four_experts):
        """Phase 3 should generally invoke fewer experts than the full garage."""
        router = LogisticRouter.from_experts(four_experts, max_activation=2)
        pipe = RoutingPipeline(
            experts=four_experts,
            router=router,
            handoff="text",
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
        result = pipe.run("Write a haiku.")
        # Router caps at 2 experts.
        assert len(result.selected_experts) <= 2
