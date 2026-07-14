"""Configuration & cost model for CHEN.

Settings are read from environment variables (with ``CHEN_`` prefix)
or set explicitly via :class:`Settings`. The :class:`CostModel`
encapsulates per-parameter cost rates used by the KPI harness.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_float(key: str, default: float) -> float:
    v = _env(key)
    try:
        return float(v) if v else default
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    v = _env(key)
    try:
        return int(v) if v else default
    except ValueError:
        return default


@dataclass(frozen=True)
class CostModel:
    """USD cost rates.

    The defaults approximate public list prices for hosted LLM inference
    in early 2025. Override via env vars if you have negotiated rates.

    Attributes:
        input_per_1m: USD per 1M input tokens.
        output_per_1m: USD per 1M output tokens.
        per_1m_params_per_1k_tokens: USD cost of loading 1M parameters
            for 1K tokens of inference. Used to estimate the "parameter
            tax" of a monolith that loads all params for every query.
    """

    input_per_1m: float = field(default_factory=lambda: _env_float("CHEN_COST_INPUT_PER_1M", 0.15))
    output_per_1m: float = field(
        default_factory=lambda: _env_float("CHEN_COST_OUTPUT_PER_1M", 0.60)
    )
    per_1m_params_per_1k_tokens: float = 0.00001

    def cost_for(
        self,
        input_tokens: int,
        output_tokens: int,
        params_m: int = 0,
        include_param_tax: bool = False,
    ) -> float:
        """Compute USD cost for a single expert invocation.

        Args:
            input_tokens: Number of input/prompt tokens.
            output_tokens: Number of generated tokens.
            params_m: Parameters loaded (in millions). Used only if
                ``include_param_tax`` is True.
            include_param_tax: If True, add a per-parameter "monolith tax"
                that approximates the cost of loading ``params_m`` for this
                query. Used when comparing against a monolithic baseline.
        """
        cost = (
            input_tokens / 1_000_000 * self.input_per_1m
            + output_tokens / 1_000_000 * self.output_per_1m
        )
        if include_param_tax and params_m > 0:
            total_tokens = input_tokens + output_tokens
            cost += params_m * (total_tokens / 1_000) * self.per_1m_params_per_1k_tokens
        return cost


@dataclass(frozen=True)
class Settings:
    """Global CHEN settings. Read from env vars at construction time.

    Most users don't need to touch this — the defaults work for the mock
    backend. Override via the corresponding env var (see .env.example).
    """

    default_backend: str = field(
        default_factory=lambda: _env("CHEN_DEFAULT_BACKEND", "mock").lower()
    )
    hf_device: str = field(default_factory=lambda: _env("CHEN_HF_DEVICE", "auto").lower())
    memory_backend: str = field(
        default_factory=lambda: _env("CHEN_MEMORY_BACKEND", "in_memory").lower()
    )
    memory_persist_dir: str = field(
        default_factory=lambda: _env("CHEN_MEMORY_PERSIST_DIR", "./chen_data/memory")
    )
    memory_embedding_model: str = field(
        default_factory=lambda: _env("CHEN_MEMORY_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    log_level: str = field(default_factory=lambda: _env("CHEN_LOG_LEVEL", "INFO"))
    cost: CostModel = field(default_factory=CostModel)

    # Router defaults
    router_max_experts: int = field(default_factory=lambda: _env_int("CHEN_ROUTER_MAX_EXPERTS", 3))
    router_min_experts: int = field(default_factory=lambda: _env_int("CHEN_ROUTER_MIN_EXPERTS", 1))

    @classmethod
    def from_env(cls) -> Settings:
        """Read all settings from environment variables."""
        return cls()

    def with_overrides(self, **kwargs: Any) -> Settings:
        """Return a copy with the given fields overridden."""
        from dataclasses import asdict

        d = asdict(self)
        d.update(kwargs)
        return Settings(**d)  # type: ignore[arg-type]
