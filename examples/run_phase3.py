"""Phase 3 demo: Dynamic routing.

Builds a 4-expert garage (Analyst, Reasoner, Coder, Synthesizer) plus
a LogisticRouter, then runs a cheap query (haiku) and an expensive
query (debug C++) to show the router activating different subsets.

Usage::

    python examples/run_phase3.py
"""

from __future__ import annotations

import argparse
import sys

from chen.backends.mock import MockBackend
from chen.core.expert import Expert, ExpertRole
from chen.core.router import (
    CosineRouter,
    HybridRouter,
    LogisticRouter,
)
from chen.phases.phase3_routing import RoutingPipeline

CHEAP_PROMPT = "Write a haiku about autumn leaves."
EXPENSIVE_PROMPT = (
    "Debug this segfault in my C++ allocator. The crash is in "
    "malloc_consolidate when freeing a large block after a realloc."
)
MATH_PROMPT = "What is 17 * 23? Show your reasoning step by step."


def build_experts() -> list[Expert]:
    return [
        Expert(
            name="analyst",
            role=ExpertRole.ANALYST,
            backend=MockBackend(params_m=3_000, role_hint="analyst"),
            tags={"entity", "logic"},
        ),
        Expert(
            name="reasoner",
            role=ExpertRole.REASONER,
            backend=MockBackend(params_m=8_000, role_hint="reasoner"),
            tags={"math", "logic"},
        ),
        Expert(
            name="coder",
            role=ExpertRole.CODER,
            backend=MockBackend(params_m=7_000, role_hint="coder"),
            tags={"code", "python", "c++"},
        ),
        Expert(
            name="synthesizer",
            role=ExpertRole.SYNTHESIZER,
            backend=MockBackend(params_m=3_000, role_hint="synthesizer"),
            tags={"text"},
        ),
    ]


def make_router(kind: str, experts: list[Expert]):
    if kind == "logistic":
        return LogisticRouter.from_experts(experts)
    elif kind == "cosine":
        return CosineRouter.from_experts(experts)
    elif kind == "hybrid":
        return HybridRouter.from_experts(experts)
    raise ValueError(f"Unknown router: {kind}")


def main() -> int:
    parser = argparse.ArgumentParser(description="CHEN Phase 3 demo (dynamic routing).")
    parser.add_argument(
        "--router",
        choices=["logistic", "cosine", "hybrid"],
        default="logistic",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    experts = build_experts()
    router = make_router(args.router, experts)
    pipe = RoutingPipeline(
        experts=experts,
        router=router,
        handoff="kv_cache",
        max_tokens_per_expert=args.max_tokens,
    )

    print("=" * 70)
    print(f"PHASE 3 — Dynamic Routing (router={router.name})")
    print("=" * 70)
    print(f"Available experts: {[e.name for e in experts]}\n")

    for label, prompt in [
        ("CHEAP  (poem)", CHEAP_PROMPT),
        ("MATH   (arithmetic)", MATH_PROMPT),
        ("EXPENSIVE (debug C++)", EXPENSIVE_PROMPT),
    ]:
        result = pipe.run(prompt, max_tokens=args.max_tokens)
        print(f"--- {label} ---")
        print(f"Prompt: {prompt!r}")
        print(f"Selected experts: {result.selected_experts}")
        print(f"Output (truncated): {result.output[:120]}...")
        print(f"Metrics: {result.metrics.summary()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
