"""Circuit breaker for expert backends.

Prevents cascading failures by stopping requests to a failing backend
after a threshold of errors. After a cooldown, the breaker enters
"half-open" state and allows one probe request; if it succeeds, the
breaker closes and traffic resumes.

States:
- ``CLOSED``: Normal operation. Requests pass through. Errors increment
  the failure counter. When failures exceed ``threshold``, the breaker
  opens.
- ``OPEN``: All requests fail fast with ``CircuitBreakerOpenError``.
  After ``cooldown_seconds``, transitions to HALF_OPEN.
- ``HALF_OPEN``: One probe request is allowed. If it succeeds, the
  breaker closes. If it fails, the breaker re-opens.

Usage::

    from chen.observability.circuit_breaker import CircuitBreaker

    breaker = CircuitBreaker(name="reasoner-backend", threshold=5, cooldown_seconds=30)

    try:
        with breaker:
            result = backend.generate(prompt)
    except CircuitBreakerOpenError:
        # Fallback to text handoff or another expert
        result = fallback_generate(prompt)
"""

from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from chen.observability.logging import get_logger

_log = get_logger("chen.observability.circuit_breaker")


class CircuitState(str, enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    """Raised when the circuit breaker is open and requests are rejected."""

    def __init__(self, name: str, state: CircuitState, retry_after: float) -> None:
        super().__init__(f"Circuit breaker '{name}' is {state.value}. Retry in {retry_after:.1f}s.")
        self.name = name
        self.state = state
        self.retry_after = retry_after


@dataclass
class CircuitBreaker:
    """A thread-safe circuit breaker for a single backend.

    Attributes:
        name: Identifier (e.g. "reasoner-backend").
        threshold: Number of failures before opening.
        cooldown_seconds: Time to wait before transitioning OPEN → HALF_OPEN.
        success_threshold: Number of successes in HALF_OPEN before closing.
    """

    name: str
    threshold: int = 5
    cooldown_seconds: float = 30.0
    success_threshold: int = 1

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def state(self) -> CircuitState:
        """Current breaker state (may transition on read if cooldown expired)."""
        with self._lock:
            self._maybe_transition()
            return self._state

    def _maybe_transition(self) -> None:
        """Transition OPEN → HALF_OPEN if cooldown has elapsed."""
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.cooldown_seconds:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                _log.info(
                    "circuit_breaker.transition",
                    name=self.name,
                    old="open",
                    new="half_open",
                )

    def __enter__(self) -> CircuitBreaker:
        with self._lock:
            self._maybe_transition()
            if self._state == CircuitState.OPEN:
                retry = max(
                    0.0,
                    self.cooldown_seconds - (time.time() - self._last_failure_time),
                )
                raise CircuitBreakerOpenError(self.name, self._state, retry)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        with self._lock:
            if exc_type is None:
                self._on_success()
            else:
                self._on_failure()

    def _on_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                _log.info(
                    "circuit_breaker.transition",
                    name=self.name,
                    old="half_open",
                    new="closed",
                )
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0

    def _on_failure(self) -> None:
        self._last_failure_time = time.time()
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            _log.warning(
                "circuit_breaker.transition",
                name=self.name,
                old="half_open",
                new="open",
            )
        elif self._state == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self.threshold:
                self._state = CircuitState.OPEN
                _log.warning(
                    "circuit_breaker.transition",
                    name=self.name,
                    old="closed",
                    new="open",
                    failures=self._failure_count,
                )

    def reset(self) -> None:
        """Manually reset the breaker to CLOSED (for admin endpoints)."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for Prometheus metrics / admin endpoints."""
        with self._lock:
            self._maybe_transition()
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "threshold": self.threshold,
                "cooldown_seconds": self.cooldown_seconds,
                "last_failure_ago": (
                    time.time() - self._last_failure_time if self._last_failure_time > 0 else None
                ),
            }


@dataclass
class CircuitBreakerRegistry:
    """Registry of circuit breakers per backend.

    Provides a singleton-like registry so the HTTP server can query
    breaker state for the /v1/health endpoint.
    """

    _breakers: dict[str, CircuitBreaker] = field(default_factory=dict)

    def get_or_create(
        self,
        name: str,
        threshold: int = 5,
        cooldown_seconds: float = 30.0,
    ) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name, threshold=threshold, cooldown_seconds=cooldown_seconds
            )
        return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:  # noqa: UP045
        return self._breakers.get(name)

    def list_all(self) -> list[dict[str, Any]]:
        return [b.to_dict() for b in self._breakers.values()]

    def reset_all(self) -> None:
        for b in self._breakers.values():
            b.reset()


# Global registry singleton
_global_registry = CircuitBreakerRegistry()


def get_global_registry() -> CircuitBreakerRegistry:
    """Return the global CircuitBreakerRegistry."""
    return _global_registry
