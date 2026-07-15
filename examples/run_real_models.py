"""Run CHEN with REAL HuggingFace models (not mock).

Downloads ~3 GB of model weights on first run (cached afterwards).
Works on CPU — takes ~30-60 seconds per query on a modern laptop.
On GPU: ~2-5 seconds per query.

Usage:
    python examples/run_real_models.py                    # default CPU-friendly models
    python examples/run_real_models.py --tier gpu         # larger models, needs GPU
    python examples/run_real_models.py --tier tiny        # super-fast verification
    python examples/run_real_models.py --phase 3          # which phase to run (1, 2, 3)

Requirements:
    pip install -e ".[hf]"
"""

from __future__ import annotations

import argparse
import sys
import time

# Model configurations per tier
TIERS = {
    # Tiny: super-fast verification that real models work end-to-end
    "tiny": {
        "analyst": ("HuggingFaceTB/SmolLM2-360M-Instruct", 360),
        "reasoner": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", 1_700),
        "synthesizer": ("HuggingFaceTB/SmolLM2-360M-Instruct", 360),
        "coder": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", 1_700),
    },
    # CPU: works on any modern laptop with 8GB+ RAM
    "cpu": {
        "analyst": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", 1_700),
        "reasoner": ("Qwen/Qwen2.5-3B-Instruct", 3_000),
        "synthesizer": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", 1_700),
        "coder": ("Qwen/Qwen2.5-3B-Instruct", 3_000),
    },
    # GPU: needs 8GB+ VRAM, uses larger models
    "gpu": {
        "analyst": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", 1_700),
        "reasoner": ("Qwen/Qwen2.5-7B-Instruct", 7_000),
        "synthesizer": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", 1_700),
        "coder": ("Qwen/CodeQwen1.5-7B-Chat", 7_000),
    },
}


def build_experts(tier: str):
    from chen.backends.hf import HuggingFaceBackend
    from chen.core.expert import Expert, ExpertRole

    cfg = TIERS[tier]
    return [
        Expert(
            name="analyst",
            role=ExpertRole.ANALYST,
            backend=HuggingFaceBackend(
                model_id=cfg["analyst"][0], params_m=cfg["analyst"][1]
            ),
        ),
        Expert(
            name="reasoner",
            role=ExpertRole.REASONER,
            backend=HuggingFaceBackend(
                model_id=cfg["reasoner"][0], params_m=cfg["reasoner"][1]
            ),
        ),
        Expert(
            name="coder",
            role=ExpertRole.CODER,
            backend=HuggingFaceBackend(
                model_id=cfg["coder"][0], params_m=cfg["coder"][1]
            ),
        ),
        Expert(
            name="synthesizer",
            role=ExpertRole.SYNTHESIZER,
            backend=HuggingFaceBackend(
                model_id=cfg["synthesizer"][0], params_m=cfg["synthesizer"][1]
            ),
        ),
    ]


def run_phase1(experts, prompt: str, max_tokens: int):
    from chen.phases.phase1_cascade import CascadePipeline

    pipe = CascadePipeline(
        experts=experts,
        sequence=["analyst", "reasoner", "synthesizer"],
        max_tokens_per_expert=max_tokens,
        memory_retrieve_k=0,
        write_intermediate_to_memory=False,
    )
    return pipe.run(prompt)


def run_phase2(experts, prompt: str, max_tokens: int):
    from chen.phases.phase2_kv_pass import KVPassPipeline

    pipe = KVPassPipeline(
        experts=experts,
        sequence=["analyst", "reasoner", "synthesizer"],
        max_tokens_per_expert=max_tokens,
        memory_retrieve_k=0,
        write_intermediate_to_memory=False,
    )
    return pipe.run(prompt)


def run_phase3(experts, prompt: str, max_tokens: int):
    from chen.core.router import LogisticRouter
    from chen.phases.phase3_routing import RoutingPipeline

    router = LogisticRouter.from_experts(experts)
    pipe = RoutingPipeline(
        experts=experts,
        router=router,
        handoff="text",  # safer for cross-family HF models
        max_tokens_per_expert=max_tokens,
        memory_retrieve_k=0,
        write_intermediate_to_memory=False,
    )
    return pipe.run(prompt)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CHEN with real HuggingFace models."
    )
    parser.add_argument(
        "--tier",
        choices=["tiny", "cpu", "gpu"],
        default="cpu",
        help="Model size tier. 'tiny' for first verification, 'cpu' for laptops, 'gpu' for 8GB+ VRAM.",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Which phase to run.",
    )
    parser.add_argument(
        "--prompt",
        default="Explain recursion to a 12-year-old using a real-world analogy.",
        help="The prompt to process.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max tokens per expert.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"CHEN — Real HuggingFace Models (tier={args.tier}, phase={args.phase})")
    print("=" * 70)
    print(f"Prompt: {args.prompt!r}\n")

    # Verify HF extra is installed
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: HuggingFace backend not installed.")
        print("Run:  pip install -e '.[hf]'")
        return 1

    # Show device info
    import torch

    if torch.cuda.is_available():
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("Device: Apple Metal (MPS)")
    else:
        print("Device: CPU (will be slow — consider --tier tiny for first run)")
    print()

    # Build experts (this downloads model weights on first run)
    print(f"Loading {args.tier}-tier models (first run downloads ~3 GB, cached after)...")
    t0 = time.perf_counter()
    experts = build_experts(args.tier)
    print(f"Models loaded in {time.perf_counter() - t0:.1f}s\n")

    # Run the selected phase
    print(f"Running Phase {args.phase}...")
    t0 = time.perf_counter()
    if args.phase == 1:
        result = run_phase1(experts, args.prompt, args.max_tokens)
    elif args.phase == 2:
        result = run_phase2(experts, args.prompt, args.max_tokens)
    else:
        result = run_phase3(experts, args.prompt, args.max_tokens)
    elapsed = time.perf_counter() - t0

    print(f"\n--- Output (Phase {args.phase}, {elapsed:.1f}s total) ---")
    print(result.output)
    print(f"\n--- Metrics ---")
    print(result.metrics.summary())
    if args.phase == 3:
        print(f"\nSelected experts: {result.selected_experts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
