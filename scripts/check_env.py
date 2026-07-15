"""Check your environment — what's installed, what's available, what's missing.

Usage:
    python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys


def check_python() -> None:
    print(f"Python:        {sys.version.split()[0]} ({platform.machine()})")
    if sys.version_info < (3, 9):
        print("  [FAIL] Python 3.9+ required")
    else:
        print("  [OK]   Python 3.9+")


def check_package(name: str, required_for: str) -> bool:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  [OK]   {name:20s} {version:15s} (for {required_for})")
        return True
    except ImportError:
        print(f"  [MISS] {name:20s} {'':15s} (for {required_for})")
        return False


def check_chen() -> None:
    print("\nCHEN package:")
    try:
        import chen

        print(f"  [OK]   chen {chen.__version__}")
        from chen.backends import list_backends

        print(f"  [OK]   backends: {', '.join(list_backends())}")
    except ImportError as e:
        print(f"  [FAIL] cannot import chen: {e}")
        print("         Did you run: pip install -e .   ?")


def check_core_deps() -> None:
    print("\nCore dependencies (always required):")
    check_package("numpy", "arrays / KV-cache")
    check_package("pydantic", "config validation")


def check_hf_stack() -> None:
    print("\nHuggingFace backend dependencies (install with: pip install -e '.[hf]'):")
    has_transformers = check_package("transformers", "model loading")
    has_torch = check_package("torch", "tensor ops / GPU")
    check_package("tokenizers", "tokenization")
    check_package("accelerate", "device placement")

    if has_torch:
        import torch

        print("\nPyTorch compute backends:")
        cuda = torch.cuda.is_available()
        mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        print(f"  CUDA available:  {cuda}")
        print(f"  MPS available:   {mps}")
        print(f"  CPU threads:     {torch.get_num_threads()}")
        if cuda:
            print(f"  GPU:             {torch.cuda.get_device_name(0)}")
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  VRAM:            {mem:.1f} GB")


def check_memory_stack() -> None:
    print("\nShared memory backend dependencies (install with: pip install -e '.[memory]'):")
    check_package("chromadb", "persistent vector store")
    check_package("sentence_transformers", "embeddings")


def check_optional_backends() -> None:
    print("\nOptional inference backends:")
    check_package("vllm", "vLLM backend (CUDA GPU required)")
    check_package("llama_cpp", "llama.cpp backend (CPU/MPS)")


def check_dev_tools() -> None:
    print("\nDev tools (install with: pip install -e '.[dev]'):")
    check_package("pytest", "tests")
    check_package("ruff", "linting")
    check_package("mypy", "type checking")


def check_git() -> None:
    print("\nGit:")
    if shutil.which("git"):
        out = subprocess.run(
            ["git", "--version"], capture_output=True, text=True, check=False
        )
        print(f"  [OK]   {out.stdout.strip()}")
    else:
        print("  [MISS] git not on PATH")


def main() -> int:
    print("=" * 60)
    print("CHEN Environment Check")
    print("=" * 60)
    check_python()
    check_git()
    check_chen()
    check_core_deps()
    check_hf_stack()
    check_memory_stack()
    check_optional_backends()
    check_dev_tools()
    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Install core:    pip install -e .")
    print("  2. Add HF backend:  pip install -e '.[hf]'")
    print("  3. Add dev tools:   pip install -e '.[dev]'")
    print("  4. Run smoke test:  python scripts/smoke_test.py")
    print("  5. Run Phase 1:     python examples/run_phase1.py --backend hf")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
