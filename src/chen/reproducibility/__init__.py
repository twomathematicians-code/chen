"""Reproducibility utilities — config hashing, run tracking, deterministic seeding.

CHEN is research software: reproducibility is a first-class concern. This
module provides:

* :func:`hash_config` — deterministic SHA-256 of a configuration dict.
* :func:`seed_everything` — seed Python, NumPy, and (if installed) PyTorch.
* :class:`RunContext` — context manager that records run metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from chen.observability.logging import get_logger

log = get_logger(__name__)


def hash_config(config: dict[str, Any]) -> str:
    """Deterministic SHA-256 hash of a configuration dict.

    The dict is JSON-serialized with sorted keys before hashing, so the
    hash is order-independent.

    Args:
        config: Any JSON-serializable dict.

    Returns:
        A 64-character hex string.
    """
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (if installed) RNGs.

    Args:
        seed: The seed value to use across all RNGs.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    log.debug("reproducibility.seed_set", seed=seed)


@dataclass
class RunContext:
    """Context for a single reproducible run."""

    config: dict[str, Any]
    config_hash: str
    seed: int

    @classmethod
    def from_config(cls, config: dict[str, Any], seed: int = 42) -> RunContext:
        """Build a RunContext from a config dict, computing its hash."""
        return cls(
            config=config,
            config_hash=hash_config(config),
            seed=seed,
        )


@contextmanager
def track_run(config: dict[str, Any], seed: int = 42) -> Iterator[RunContext]:
    """Context manager that seeds RNGs and logs run start/end.

    Usage::

        with track_run({"phase": 1, "backend": "mock"}) as ctx:
            result = pipeline.run(prompt)
            # ctx.config_hash can be used as a run_id
    """
    ctx = RunContext.from_config(config, seed)
    seed_everything(seed)
    log.info(
        "run.start",
        config_hash=ctx.config_hash,
        seed=seed,
        config=config,
    )
    try:
        yield ctx
    except Exception as e:
        log.error("run.failed", config_hash=ctx.config_hash, error=str(e))
        raise
    finally:
        log.info("run.end", config_hash=ctx.config_hash)
