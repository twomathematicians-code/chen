"""Sample benchmark tasks with deterministic graders.

Each :class:`BenchmarkTask` is a (name, prompts, grader) triple. The
grader takes the CHEN output and the expected answer and returns a
float in [0, 1] indicating correctness.

The tasks shipped here are deliberately small and deterministic so the
test suite can run on the MockBackend. For real experiments, swap in
MMLU, HumanEval, or GSM8K via the ``register_task`` API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Protocol

# Type alias: a grader is a function (output: str, expected: str) -> float in [0, 1].
Grader = Callable[[str, str], float]


class GraderProtocol(Protocol):
    def __call__(self, output: str, expected: str) -> float: ...


@dataclass
class BenchmarkTask:
    """One benchmark task = a name + a list of (prompt, expected) pairs + a grader.

    Attributes:
        name: Short unique id (e.g. ``"math_arithmetic"``).
        description: One-line description.
        samples: List of (prompt, expected_answer) tuples.
        grader: Function (output, expected) -> float in [0, 1].
        tags: Free-form tags for filtering.
    """

    name: str
    description: str
    samples: list[tuple[str, str]]
    grader: Grader
    tags: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------


def exact_match_grader(output: str, expected: str) -> float:
    """1.0 if the expected string appears anywhere in the output, else 0.0."""
    return 1.0 if expected.strip().lower() in output.strip().lower() else 0.0


def numeric_grader(output: str, expected: str) -> float:
    """Extract the first number from output and compare to expected.

    Returns 1.0 on exact match, 0.5 within 5% relative error, else 0.0.
    """
    nums = re.findall(r"-?\d+(?:\.\d+)?", output)
    if not nums:
        return 0.0
    try:
        got = float(nums[0])
        want = float(expected)
    except ValueError:
        return 0.0
    if abs(got - want) < 1e-9:
        return 1.0
    if want != 0 and abs(got - want) / abs(want) <= 0.05:
        return 0.5
    return 0.0


def keyword_coverage_grader(output: str, expected: str) -> float:
    """Fraction of expected keywords (comma-separated) found in output."""
    keywords = [k.strip().lower() for k in expected.split(",") if k.strip()]
    if not keywords:
        return 0.0
    out_lower = output.lower()
    found = sum(1 for k in keywords if k in out_lower)
    return found / len(keywords)


def mock_friendly_grader(output: str, expected: str) -> float:
    """Grader that gives the MockBackend a fair score.

    The MockBackend's output is structured (contains a hash, role hint,
    and source text snippet), not a real answer. This grader gives:
      - 1.0 if the expected answer appears in the output
      - 0.5 if any word from the expected answer appears
      - 0.0 otherwise
    """
    if expected.strip().lower() in output.strip().lower():
        return 1.0
    words = [w for w in re.findall(r"\w+", expected.lower()) if len(w) > 3]
    if not words:
        return 0.0
    out_lower = output.lower()
    found = sum(1 for w in words if w in out_lower)
    return 0.5 * (found / len(words))


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


MATH_TASK = BenchmarkTask(
    name="math_arithmetic",
    description="Simple arithmetic problems.",
    samples=[
        ("What is 17 + 25?", "42"),
        ("What is 9 * 13?", "117"),
        ("What is 144 / 12?", "12"),
        ("What is 2^10?", "1024"),
        ("What is 100 - 37?", "63"),
    ],
    grader=numeric_grader,
    tags={"math", "reasoning"},
)


CODE_TASK = BenchmarkTask(
    name="code_python_basics",
    description="Basic Python coding questions.",
    samples=[
        ("Write a Python function that returns the sum of a list.", "def sum_list"),
        ("Write a Python function that reverses a string.", "def reverse_string"),
        ("Write a Python function that checks if a number is prime.", "def is_prime"),
        ("Write a Python function that returns the factorial of n.", "def factorial"),
        ("Write a Python function that finds the max element of a list.", "def max_element"),
    ],
    grader=keyword_coverage_grader,
    tags={"code", "python"},
)


QA_TASK = BenchmarkTask(
    name="qa_factual",
    description="Factual question answering.",
    samples=[
        ("What is the capital of France?", "Paris"),
        ("What is the largest planet in our solar system?", "Jupiter"),
        ("Who wrote the play 'Hamlet'?", "Shakespeare"),
        ("What is the chemical symbol for gold?", "Au"),
        ("In what year did the Berlin Wall fall?", "1989"),
    ],
    grader=exact_match_grader,
    tags={"qa", "knowledge"},
)


SUMMARY_TASK = BenchmarkTask(
    name="summarization",
    description="Summarize a short passage.",
    samples=[
        (
            "Summarize: The CHEN architecture replaces a single large model with a coordinated "
            "network of small specialized models. By routing tokens through a dynamic pipeline "
            "and sharing latent memory states between models, CHEN simulates the parameter "
            "capacity of a frontier model at a fraction of the compute cost.",
            "small models,routing,kv-cache,cost",
        ),
        (
            "Summarize: Mixture of Experts is a machine learning technique where multiple "
            "specialized subnetworks (experts) process different parts of the input. A gating "
            "network decides which expert to use for each input. This allows the model to scale "
            "parameters without proportionally scaling compute per query.",
            "experts,gating,scale,compute",
        ),
    ],
    grader=keyword_coverage_grader,
    tags={"summarization", "synthesis"},
)


REASONING_TASK = BenchmarkTask(
    name="reasoning_logical",
    description="Multi-step logical reasoning.",
    samples=[
        (
            "If all cats are mammals, and Whiskers is a cat, what is Whiskers?",
            "mammal",
        ),
        (
            "Alice is taller than Bob. Bob is taller than Carol. Who is the shortest?",
            "Carol",
        ),
        (
            "A train travels 60 mph for 2 hours, then 80 mph for 1 hour. Total distance?",
            "200",
        ),
        (
            "If today is Monday, what day is it 100 days from now?",
            "Wednesday",
        ),
        (
            "Solve: 3x + 7 = 22. What is x?",
            "5",
        ),
    ],
    grader=mock_friendly_grader,
    tags={"reasoning", "logic"},
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


TASK_REGISTRY: dict[str, BenchmarkTask] = {}


def register_task(task: BenchmarkTask) -> None:
    """Register a benchmark task. Overwrites existing tasks with the same name."""
    TASK_REGISTRY[task.name] = task


def _register_builtins() -> None:
    for t in (MATH_TASK, CODE_TASK, QA_TASK, SUMMARY_TASK, REASONING_TASK):
        register_task(t)


_register_builtins()
