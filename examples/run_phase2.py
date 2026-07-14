"""Phase 2 demo: KV-cache passing between experts.

Runs the same 3-expert sequence as Phase 1, but each expert receives the
previous expert's KV-cache instead of text. Compares the latent nuance
score against the text-only baseline.

Usage::

    python examples/run_phase2.py
    python examples/run_phase2.py --prompt "Explain quantum entanglement."
"""

from __future__ import annotations

import argparse
import sys

from chen.backends.mock import MockBackend
from chen.core.expert import Expert, ExpertRole
from chen.phases.phase1_cascade import CascadePipeline
from chen.phases.phase2_kv_pass import KVPassPipeline

DEFAULT_PROMPT = (
    "Explain quantum entanglement to a 12-year-old, then derive Bell's "
    "inequality step by step."
)


def build_experts() -> list[Expert]:
    return [
        Expert(
            name="analyst",
            role=ExpertRole.ANALYST,
            backend=MockBackend(params_m=3_000, role_hint="analyst"),
        ),
        Expert(
            name="reasoner",
            role=ExpertRole.REASONER,
            backend=MockBackend(params_m=8_000, role_hint="reasoner"),
        ),
        Expert(
            name="synthesizer",
            role=ExpertRole.SYNTHESIZER,
            backend=MockBackend(params_m=3_000, role_hint="synthesizer"),
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="CHEN Phase 2 demo (KV-cache passing).")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    experts = build_experts()
    sequence = ["analyst", "reasoner", "synthesizer"]

    # Phase 1 baseline (text handoff)
    cascade = CascadePipeline(
        experts=experts,
        sequence=sequence,
        max_tokens_per_expert=args.max_tokens,
        memory_retrieve_k=0,  # isolate the KV variable
        write_intermediate_to_memory=False,
    )
    cascade_result = cascade.run(args.prompt)

    # Phase 2 (KV-cache handoff)
    kv_pipe = KVPassPipeline(
        experts=experts,
        sequence=sequence,
        max_tokens_per_expert=args.max_tokens,
        memory_retrieve_k=0,
        write_intermediate_to_memory=False,
    )
    kv_result = kv_pipe.run(args.prompt)

    print("=" * 70)
    print("PHASE 2 — KV-Cache Passing vs. Text Handoff")
    print("=" * 70)
    print(f"Prompt: {args.prompt!r}\n")

    print("--- Phase 1 (text handoff) ---")
    print(f"Output: {cascade_result.output[:200]}...")
    print(f"Metrics: {cascade_result.metrics.summary()}\n")

    print("--- Phase 2 (KV-cache handoff) ---")
    print(f"Output: {kv_result.output[:200]}...")
    print(f"Metrics: {kv_result.metrics.summary()}\n")

    print("--- Comparison ---")
    cascade_nuance = cascade_result.metrics.latent_nuance_score
    kv_nuance = kv_result.metrics.latent_nuance_score
    ratio = kv_nuance / cascade_nuance if cascade_nuance > 0 else float("inf")
    print(f"  cascade nuance: {cascade_nuance:.3f}")
    print(f"  kv-pass  nuance: {kv_nuance:.3f}")
    print(f"  KV-pass retained {ratio:.2f}x the nuance of text-only.")
    print(f"  KV transfers: {kv_result.metrics.kv_cache_transfers} "
          f"(failures: {kv_result.metrics.kv_cache_failures})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
