"""Tests for the standard benchmark tasks (MMLU, HumanEval, GSM8K)."""

from __future__ import annotations

from chen.benchmarks.tasks import (
    GSM8K_TASK,
    HUMANEVAL_TASK,
    MMLU_TASK,
    TASK_REGISTRY,
    _gsm8k_grader,
    _humaneval_grader,
    _mmlu_grader,
)


class TestMMLUGrader:
    def test_correct_answer_with_label(self):
        assert _mmlu_grader("The answer is B", "B") == 1.0

    def test_correct_answer_with_colon(self):
        assert _mmlu_grader("Answer: B", "B") == 1.0

    def test_correct_answer_in_parens(self):
        assert _mmlu_grader("The answer is (B)", "B") == 1.0

    def test_wrong_answer(self):
        assert _mmlu_grader("The answer is C", "B") == 0.0

    def test_partial_credit_for_ambiguous(self):
        # Just the letter B somewhere — ambiguous
        assert _mmlu_grader("I think B might be right", "B") == 0.5

    def test_empty_expected(self):
        assert _mmlu_grader("anything", "") == 0.0


class TestHumanEvalGrader:
    def test_exact_match(self):
        assert _humaneval_grader("def add(a, b):\n    return a + b", "return a + b") == 1.0

    def test_no_match(self):
        assert _humaneval_grader("return a - b", "return a + b") == 0.0

    def test_empty_expected(self):
        assert _humaneval_grader("anything", "") == 0.0


class TestGSM8KGrader:
    def test_exact_number_match(self):
        assert _gsm8k_grader("The answer is 18", "18") == 1.0

    def test_number_with_commas(self):
        assert _gsm8k_grader("The answer is 70,000", "70000") == 1.0

    def test_wrong_number(self):
        assert _gsm8k_grader("The answer is 42", "18") == 0.0

    def test_empty_expected(self):
        assert _gsm8k_grader("anything", "") == 0.0


class TestStandardTasksRegistered:
    def test_mmlu_registered(self):
        assert "mmlu" in TASK_REGISTRY
        assert TASK_REGISTRY["mmlu"] is MMLU_TASK

    def test_humaneval_registered(self):
        assert "humaneval" in TASK_REGISTRY
        assert TASK_REGISTRY["humaneval"] is HUMANEVAL_TASK

    def test_gsm8k_registered(self):
        assert "gsm8k" in TASK_REGISTRY
        assert TASK_REGISTRY["gsm8k"] is GSM8K_TASK

    def test_mmlu_has_5_samples(self):
        assert len(MMLU_TASK.samples) == 5

    def test_humaneval_has_5_samples(self):
        assert len(HUMANEVAL_TASK.samples) == 5

    def test_gsm8k_has_5_samples(self):
        assert len(GSM8K_TASK.samples) == 5

    def test_mmlu_tags(self):
        assert "mmlu" in MMLU_TASK.tags
        assert "knowledge" in MMLU_TASK.tags

    def test_humaneval_tags(self):
        assert "humaneval" in HUMANEVAL_TASK.tags
        assert "code" in HUMANEVAL_TASK.tags

    def test_gsm8k_tags(self):
        assert "gsm8k" in GSM8K_TASK.tags
        assert "math" in GSM8K_TASK.tags
