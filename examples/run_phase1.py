"""Phase 1 demo: Static cascading with text handoff.

Runs a 3-expert cascade (Analyst → Reasoner → Synthesizer) on a sample
prompt using the MockBackend, and prints the output + cost/latency.

Usage::

    python examples/run_phase1.py
    python examples/run_phase1.py --prompt "Explain recursion to a 5-year-old."
    python examples/run_phase1.py --backend hf  # requires the hf extra
"""

from __future__ import annotations

import argparse
import sys

from chen.backends.mock import MockBackend
from chen.core.expert import Expert, ExpertRole
from chen.phases.phase1_cascade import CascadePipeline

DEFAULT_PROMPT = (
    "Read this 10-page PDF about renewable energy trends and write a Python "
    "script that loads the data and plots solar vs wind capacity over time."
)


def build_experts(backend: str = "mock") -> list[Expert]:
    if backend == "mock":
        return [
            Expert(
                name="analyst",
                role=ExpertRole.ANALYST,
                backend=MockBackend(params_m=3_000, role_hint="analyst"),
                description="Extracts entities and logic from text.",
                tags={"entity", "logic"},
            ),
            Expert(
                name="reasoner",
                role=ExpertRole.REASONER,
                backend=MockBackend(params_m=8_000, role_hint="reasoner"),
                description="Math and code reasoning.",
                tags={"math", "code"},
            ),
            Expert(
                name="synthesizer",
                role=ExpertRole.SYNTHESIZER,
                backend=MockBackend(params_m=3_000, role_hint="synthesizer"),
                description="Converts latent state to natural language.",
                tags={"text", "natural-language"},
            ),
        ]
    elif backend == "hf":
        from chen.backends.hf import HuggingFaceBackend

        # Real, open, non-gated HuggingFace models (no token required).
        # Swap for larger models if you have a GPU — see .env.example.
        return [
            Expert(
                name="analyst",
                role=ExpertRole.ANALYST,
                backend=HuggingFaceBackend(
                    model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                    params_m=1_700,
                ),
            ),
            Expert(
                name="reasoner",
                role=ExpertRole.REASONER,
                backend=HuggingFaceBackend(
                    model_id="Qwen/Qwen2.5-3B-Instruct",
                    params_m=3_000,
                ),
            ),
            Expert(
                name="synthesizer",
                role=ExpertRole.SYNTHESIZER,
                backend=HuggingFaceBackend(
                    model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                    params_m=1_700,
                ),
            ),
        ]
    else:
        raise ValueError(f"Unknown backend: {backend}")


def main() -> int:
    parser = argparse.ArgumentParser(description="CHEN Phase 1 demo (static cascade).")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="The prompt to process.")
    parser.add_argument(
        "--backend",
        choices=["mock", "hf"],
        default="mock",
        help="Which backend to use. 'mock' runs on CPU in <1s; 'hf' requires the hf extra.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256, help="Token budget per expert."
    )
    parser.add_argument(
        "--no-memory", action="store_true", help="Disable shared memory augmentation."
    )
    args = parser.parse_args()

    experts = build_experts(args.backend)
    pipe = CascadePipeline(
        experts=experts,
        sequence=["analyst", "reasoner", "synthesizer"],
        max_tokens_per_expert=args.max_tokens,
        write_intermediate_to_memory=not args.no_memory,
        memory_retrieve_k=0 if args.no_memory else 2,
    )
    result = pipe.run(args.prompt)

    print("=" * 70)
    print("PHASE 1 — Static Cascade (text handoff)")
    print("=" * 70)
    print(f"Prompt: {args.prompt!r}\n")
    for i, m in enumerate(result.per_expert):
        print(f"--- Expert {i + 1}: {m.expert_name} ({m.role.value}) ---")
        print(f"  params: {m.params_m}M, in_tok: {m.input_tokens}, out_tok: {m.output_tokens}")
        print(f"  latency: {m.latency_ms:.1f} ms, used_kv: {m.used_kv_cache}")
    print("\n--- Final output ---")
    print(result.output)
    print("\n--- Aggregate metrics ---")
    print(result.metrics.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
