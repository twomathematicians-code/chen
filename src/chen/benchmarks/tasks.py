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
# Standard benchmark task configs (MMLU, HumanEval, GSM8K)
#
# These are reproduction configs — the sample sets here are small subsets
# for quick verification. For real benchmarks, download the full datasets
# from HuggingFace and pass them to BenchmarkRunner.
# ---------------------------------------------------------------------------


def _mmlu_grader(output: str, expected: str) -> float:
    """MMLU grader: expected is the letter (A/B/C/D). 1.0 if found, 0.0 otherwise."""
    expected = expected.strip().upper()
    if not expected:
        return 0.0
    # Look for the letter at the start of a word or after "answer is".
    import re

    out_upper = output.upper()
    # Check for "Answer: X" or "The answer is X" patterns.
    patterns = [
        rf"ANSWER\s*[:\-]?\s*{expected}\b",
        rf"ANSWER\s+IS\s*[:\-]?\s*{expected}\b",
        rf"\({expected}\)",
        rf"^\s*{expected}\s*[\.\,]",
    ]
    for p in patterns:
        if re.search(p, out_upper):
            return 1.0
    # Fallback: first occurrence of the letter as a standalone word.
    if re.search(rf"\b{expected}\b", out_upper):
        return 0.5  # partial credit — ambiguous
    return 0.0


def _humaneval_grader(output: str, expected: str) -> float:
    """HumanEval grader: expected is the function signature. 1.0 if found."""
    expected = expected.strip()
    if not expected:
        return 0.0
    return 1.0 if expected in output else 0.0


def _gsm8k_grader(output: str, expected: str) -> float:
    """GSM8K grader: expected is the final numeric answer. 1.0 if found."""
    expected = expected.strip()
    if not expected:
        return 0.0
    # GSM8K answers end with "#### N" — extract the number.
    nums = expected.replace(",", "").split()
    target = nums[-1] if nums else expected
    # Check if the target number appears in the output.
    import re

    out_nums = re.findall(r"[\d,]+\.?\d*", output.replace(",", ""))
    for n in out_nums:
        if n == target:
            return 1.0
    # Also check if target appears as a substring.
    if target in output:
        return 1.0
    return 0.0


# MMLU — Massive Multitask Language Understanding
# Subset of 5 questions across 5 subjects. Full dataset: 14,042 questions.
# https://huggingface.co/datasets/cais/mmlu
MMLU_TASK = BenchmarkTask(
    name="mmlu",
    description="MMLU — Massive Multitask Language Understanding (subset).",
    samples=[
        # Elementary Mathematics
        (
            "Question: What is 7 + 6?\nA) 12\nB) 13\nC) 14\nD) 15\nAnswer:",
            "B",
        ),
        # College Chemistry
        (
            "Question: Which of the following is the ground state electron configuration of oxygen?\n"
            "A) 1s2 2s2 2p4\nB) 1s2 2s2 2p6\nC) 1s2 2s2 2p2\nD) 1s2 2s2 2p3\nAnswer:",
            "A",
        ),
        # High School History
        (
            "Question: In what year did World War II end?\n"
            "A) 1943\nB) 1944\nC) 1945\nD) 1946\nAnswer:",
            "C",
        ),
        # Professional Law
        (
            "Question: Which amendment to the US Constitution guarantees freedom of speech?\n"
            "A) First\nB) Second\nC) Fourth\nD) Fifth\nAnswer:",
            "A",
        ),
        # College Computer Science
        (
            "Question: What is the time complexity of binary search on a sorted array?\n"
            "A) O(n)\nB) O(n log n)\nC) O(log n)\nD) O(1)\nAnswer:",
            "C",
        ),
    ],
    grader=_mmlu_grader,
    tags={"mmlu", "knowledge", "multiple_choice"},
)


# HumanEval — Code generation benchmark
# Subset of 5 problems. Full dataset: 164 problems.
# https://huggingface.co/datasets/openai_humaneval
HUMANEVAL_TASK = BenchmarkTask(
    name="humaneval",
    description="HumanEval — code generation benchmark (subset).",
    samples=[
        (
            "Complete the following Python function:\n\n"
            "def add(a, b):\n"
            '    """Return the sum of two numbers."""\n',
            "return a + b",
        ),
        (
            "Complete the following Python function:\n\n"
            "def is_even(n):\n"
            '    """Return True if n is even, False otherwise."""\n',
            "return n % 2 == 0",
        ),
        (
            "Complete the following Python function:\n\n"
            "def factorial(n):\n"
            '    """Return the factorial of n."""\n',
            "return 1 if n <= 1 else n * factorial(n - 1)",
        ),
        (
            "Complete the following Python function:\n\n"
            "def reverse_string(s):\n"
            '    """Return the reversed string."""\n',
            "return s[::-1]",
        ),
        (
            "Complete the following Python function:\n\n"
            "def max_of_list(lst):\n"
            '    """Return the maximum element in the list."""\n',
            "return max(lst)",
        ),
    ],
    grader=_humaneval_grader,
    tags={"humaneval", "code", "python"},
)


# GSM8K — Grade School Math 8K
# Subset of 5 problems. Full dataset: 8,500 problems.
# https://huggingface.co/datasets/openai/gsm8k
GSM8K_TASK = BenchmarkTask(
    name="gsm8k",
    description="GSM8K — grade school math problems (subset).",
    samples=[
        (
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning "
            "and bakes muffins for her friends every day with four. She sells the remainder "
            "at the farmers' market daily for $2 per fresh duck egg. How much in dollars "
            "does she make every day at the farmers' market?",
            "18",
        ),
        (
            "A robe takes 2 bolts of blue fiber and half that much white fiber. "
            "How many bolts in total does it take?",
            "3",
        ),
        (
            "Josh decides to try flipping a house. He buys a house for $80,000 and then "
            "puts in $50,000 in repairs. This increased the value of the house by 150%. "
            "How much profit did he make?",
            "70000",
        ),
        (
            "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. "
            "How many total meters does he run a week?",
            "540",
        ),
        (
            "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, "
            "containing seeds, mealworms and vegetables to help keep them healthy. "
            "She gives the chickens their feed in three separate meals. In the morning, "
            "she gives her flock of chickens 15 cups of feed. In the afternoon, she gives "
            "her chickens another 25 cups of feed. How many cups of feed does she need to "
            "give her chickens in the final meal of the day if the size of Wendi's flock "
            "is 20 chickens?",
            "20",
        ),
    ],
    grader=_gsm8k_grader,
    tags={"gsm8k", "math", "reasoning"},
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


TASK_REGISTRY: dict[str, BenchmarkTask] = {}


def register_task(task: BenchmarkTask) -> None:
    """Register a benchmark task. Overwrites existing tasks with the same name."""
    TASK_REGISTRY[task.name] = task


def _register_builtins() -> None:
    for t in (
        MATH_TASK,
        CODE_TASK,
        QA_TASK,
        SUMMARY_TASK,
        REASONING_TASK,
        MMLU_TASK,
        HUMANEVAL_TASK,
        GSM8K_TASK,
    ):
        register_task(t)


_register_builtins()
