"""Backend registry.

Maintains a global mapping from backend name to backend class. Built-in
backends (mock, hf, vllm, llama_cpp) are registered automatically when
:mod:`chen.backends` is imported. Users can register custom backends
via :func:`register_backend`.
"""

from __future__ import annotations

from typing import Any, Callable

from chen.backends.base import BackendError, InferenceBackend

# Type alias: a backend class is any callable that, when invoked with
# keyword arguments, returns an object implementing InferenceBackend.
BackendFactory = Callable[..., InferenceBackend]

BACKEND_REGISTRY: dict[str, BackendFactory] = {}


def register_backend(name: str, factory: BackendFactory) -> None:
    """Register a backend under ``name``.

    Args:
        name: The backend's short name (e.g. ``"mock"``, ``"hf"``).
            Case-insensitive — normalized to lowercase.
        factory: A class (or any callable) that, when called with keyword
            arguments, returns a backend instance.

    Raises:
        BackendError: If ``name`` is already registered with a different factory.
    """
    key = name.lower()
    if key in BACKEND_REGISTRY and BACKEND_REGISTRY[key] is not factory:
        raise BackendError(
            f"Backend '{key}' is already registered to "
            f"{BACKEND_REGISTRY[key]!r}. Use a different name."
        )
    BACKEND_REGISTRY[key] = factory


def get_backend(name: str, **kwargs: Any) -> InferenceBackend:
    """Look up and instantiate a backend by name.

    Args:
        name: Backend short name (case-insensitive).
        **kwargs: Forwarded to the backend factory.

    Raises:
        BackendError: If ``name`` is not registered.
    """
    key = name.lower()
    if key not in BACKEND_REGISTRY:
        raise BackendError(
            f"Unknown backend '{name}'. Registered backends: {', '.join(sorted(BACKEND_REGISTRY))}."
        )
    return BACKEND_REGISTRY[key](**kwargs)


def list_backends() -> list[str]:
    """Return the sorted list of registered backend names."""
    return sorted(BACKEND_REGISTRY)
