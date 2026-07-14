"""Pluggable inference backends.

Every expert talks to a backend that implements the ``InferenceBackend``
protocol defined in :mod:`chen.backends.base`. The repo ships with a fully
working :class:`~chen.backends.mock.MockBackend` (deterministic, CPU-only,
no model downloads) and a fully working
:class:`~chen.backends.hf.HuggingFaceBackend` (requires the ``hf`` extra).
The ``vllm`` and ``llama_cpp`` backends are stubs in v0.1.

Select a backend via the ``CHEN_DEFAULT_BACKEND`` env var or by passing an
explicit ``backend=`` to each :class:`~chen.core.expert.Expert`.
"""

from __future__ import annotations

from chen.backends.base import BackendCapabilities, InferenceBackend
from chen.backends.mock import MockBackend
from chen.backends.registry import (
    BACKEND_REGISTRY,
    get_backend,
    list_backends,
    register_backend,
)

__all__ = [
    "InferenceBackend",
    "BackendCapabilities",
    "MockBackend",
    "get_backend",
    "register_backend",
    "list_backends",
    "BACKEND_REGISTRY",
]


def _register_builtins() -> None:
    """Register all built-in backends. Called once on import."""
    register_backend("mock", MockBackend)

    # HuggingFace backend — only register if transformers is installed.
    try:  # pragma: no cover - exercised only when hf extra is present
        from chen.backends.hf import HuggingFaceBackend

        register_backend("hf", HuggingFaceBackend)
    except ImportError:  # pragma: no cover
        pass

    # vLLM and llama.cpp backends are stubs — always registered so that
    # users get a helpful error message instead of a silent "not found".
    try:
        from chen.backends.vllm import VLLMBackend

        register_backend("vllm", VLLMBackend)
    except ImportError:  # pragma: no cover
        pass

    try:
        from chen.backends.llama_cpp import LlamaCppBackend

        register_backend("llama_cpp", LlamaCppBackend)
    except ImportError:  # pragma: no cover
        pass


_register_builtins()
