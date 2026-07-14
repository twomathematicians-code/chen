"""Tests for the benchmarks layer."""

from __future__ import annotations

import pytest

from chen.backends.mock import MockBackend
from chen.benchmarks.kpis import BaselineMetrics, KPIReport, KPIs
from chen.benchmarks.runner import BenchmarkRunner
from chen.benchmarks.tasks import (
    TASK_REGISTRY,
    BenchmarkTask,
    exact_match_grader,
    keyword_coverage_grader,
    numeric_grader,
    register_task,
)
from chen.core.expert import Expert, ExpertRole
from chen.phases.phase1_cascade import CascadePipeline

# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------


class TestGraders:
    def test_exact_match_found(self):
        assert exact_match_grader("Paris is great", "Paris") == 1.0

    def test_exact_match_not_found(self):
        assert exact_match_grader("London is great", "Paris") == 0.0

    def test_exact_match_case_insensitive(self):
        assert exact_match_grader("PARIS is great", "paris") == 1.0

    def test_numeric_exact(self):
        assert numeric_grader("The answer is 42", "42") == 1.0

    def test_numeric_within_5pct(self):
        assert numeric_grader("The answer is 99", "100") == 0.5

    def test_numeric_far_off(self):
        assert numeric_grader("The answer is 50", "100") == 0.0

    def test_numeric_no_number_in_output(self):
        assert numeric_grader("no number here", "42") == 0.0

    def test_keyword_coverage_all_found(self):
        assert keyword_coverage_grader("cats and dogs", "cats,dogs") == 1.0

    def test_keyword_coverage_partial(self):
        assert keyword_coverage_grader("cats and birds", "cats,dogs") == 0.5

    def test_keyword_coverage_empty_expected(self):
        assert keyword_coverage_grader("anything", "") == 0.0


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


class TestBenchmarkTask:
    def test_registry_has_builtins(self):
        assert "math_arithmetic" in TASK_REGISTRY
        assert "code_python_basics" in TASK_REGISTRY
        assert "qa_factual" in TASK_REGISTRY
        assert "summarization" in TASK_REGISTRY
        assert "reasoning_logical" in TASK_REGISTRY

    def test_register_custom_task(self):
        task = BenchmarkTask(
            name="test_custom",
            description="custom test task",
            samples=[("hello", "world")],
            grader=exact_match_grader,
        )
        register_task(task)
        try:
            assert "test_custom" in TASK_REGISTRY
            assert TASK_REGISTRY["test_custom"] is task
        finally:
            TASK_REGISTRY.pop("test_custom", None)

    def test_each_task_has_samples(self):
        for name, task in TASK_REGISTRY.items():
            assert len(task.samples) > 0, f"task {name!r} has no samples"


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------


class TestKPIs:
    def test_compute_returns_report(self):
        kpis = KPIs()
        # CHEN: 10 queries, total 10K tokens, total cost $0.002 (very cheap
        # because we only invoke 14B params total, not 70B).
        report = kpis.compute(
            chen_total_tokens=10_000,
            chen_total_cost_usd=0.002,
            chen_latency_ms=200.0,
            chen_accuracy=0.8,
            chen_params_invoked_m=15_000,  # 3 experts of 3+8+3 = 14, plus re-invoke
            chen_distinct_params_m=14_000,  # 3+8+3
            baseline=BaselineMetrics(
                params_m=70_000,
                accuracy=0.85,
                latency_ms=1500.0,
                input_tokens=200,
                output_tokens=300,
            ),
            n_queries=10,
        )
        assert isinstance(report, KPIReport)
        # EPU = 70_000 / 14_000 = 5.0
        assert report.epu == pytest.approx(5.0, rel=0.01)
        assert report.epu_target_met is True  # 5.0 >= 3.0
        # CHEN: $0.002 / 10K * 1M = $0.20/1M tokens.
        # Baseline: 200*0.15/1M + 300*0.60/1M = $0.00021 per query, 500 tokens
        # per query -> $0.42/1M tokens. CHEN is >50% cheaper.
        assert report.chen_cost_per_1m < report.baseline_cost_per_1m
        assert report.cost_reduction_pct > 50.0

    def test_summary_is_human_readable(self):
        kpis = KPIs()
        report = kpis.compute(
            chen_total_tokens=10_000,
            chen_total_cost_usd=0.05,
            chen_latency_ms=200.0,
            chen_accuracy=0.8,
            chen_params_invoked_m=15_000,
            chen_distinct_params_m=14_000,
            baseline=BaselineMetrics(params_m=70_000),
        )
        s = report.summary()
        assert "EPU" in s
        assert "Cost" in s
        assert "Latency" in s

    def test_as_dict_returns_flat_dict(self):
        kpis = KPIs()
        report = kpis.compute(
            chen_total_tokens=10_000,
            chen_total_cost_usd=0.05,
            chen_latency_ms=200.0,
            chen_accuracy=0.8,
            chen_params_invoked_m=15_000,
            chen_distinct_params_m=14_000,
            baseline=BaselineMetrics(params_m=70_000),
        )
        d = report.as_dict()
        assert "epu" in d
        assert "chen_cost_per_1m" in d
        assert isinstance(d["epu"], float)


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------


class TestBenchmarkRunner:
    @pytest.fixture
    def simple_pipeline(self):
        experts = [
            Expert(
                name="generalist",
                role=ExpertRole.SYNTHESIZER,
                backend=MockBackend(params_m=3_000, role_hint="synth"),
            ),
        ]
        return CascadePipeline(
            experts=experts,
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )

    def test_run_task_returns_result(self, simple_pipeline):
        runner = BenchmarkRunner(
            pipeline=simple_pipeline,
            baseline=BaselineMetrics(params_m=70_000, accuracy=0.5),
        )
        result = runner.run_task("math_arithmetic", max_tokens=64)
        assert result.task_name == "math_arithmetic"
        assert len(result.per_sample) == 5
        assert 0.0 <= result.avg_accuracy <= 1.0
        assert result.total_tokens > 0
        assert result.kpi_report is not None

    def test_run_task_by_name(self, simple_pipeline):
        runner = BenchmarkRunner(pipeline=simple_pipeline)
        result = runner.run_task("qa_factual", max_tokens=64)
        assert result.task_name == "qa_factual"

    def test_run_unknown_task_raises(self, simple_pipeline):
        runner = BenchmarkRunner(pipeline=simple_pipeline)
        with pytest.raises(KeyError, match="Unknown task"):
            runner.run_task("nonexistent")

    def test_run_all_runs_every_task(self, simple_pipeline):
        runner = BenchmarkRunner(
            pipeline=simple_pipeline,
            baseline=BaselineMetrics(params_m=70_000, accuracy=0.5),
        )
        results = runner.run_all(max_tokens=32)
        assert len(results) == len(TASK_REGISTRY)
        for r in results:
            assert r.kpi_report is not None

    def test_summary_includes_kpi_report(self, simple_pipeline):
        runner = BenchmarkRunner(
            pipeline=simple_pipeline,
            baseline=BaselineMetrics(params_m=70_000, accuracy=0.5),
        )
        result = runner.run_task("qa_factual", max_tokens=64)
        s = result.summary()
        assert "Benchmark: qa_factual" in s
        assert "EPU" in s
