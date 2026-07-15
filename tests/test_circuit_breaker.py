"""Tests for the circuit breaker."""

from __future__ import annotations

import time

import pytest

from chen.observability.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitBreakerRegistry,
    CircuitState,
    get_global_registry,
)


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(name="test")
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(name="test", threshold=3, cooldown_seconds=60)
        for _ in range(3):
            try:
                with cb:
                    raise ValueError("simulated failure")
            except ValueError:
                pass
        assert cb.state == CircuitState.OPEN

    def test_open_rejects_requests(self):
        cb = CircuitBreaker(name="test", threshold=1, cooldown_seconds=60)
        try:
            with cb:
                raise ValueError("fail")
        except ValueError:
            pass
        assert cb.state == CircuitState.OPEN
        with pytest.raises(CircuitBreakerOpenError):
            with cb:
                pass

    def test_transitions_to_half_open_after_cooldown(self):
        cb = CircuitBreaker(name="test", threshold=1, cooldown_seconds=0.1)
        try:
            with cb:
                raise ValueError("fail")
        except ValueError:
            pass
        assert cb.state == CircuitState.OPEN
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_closes_on_success(self):
        cb = CircuitBreaker(name="test", threshold=1, cooldown_seconds=0.1)
        try:
            with cb:
                raise ValueError("fail")
        except ValueError:
            pass
        time.sleep(0.15)
        # Now in HALF_OPEN — a success should close it
        with cb:
            pass
        assert cb.state == CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self):
        cb = CircuitBreaker(name="test", threshold=1, cooldown_seconds=0.1)
        try:
            with cb:
                raise ValueError("fail")
        except ValueError:
            pass
        time.sleep(0.15)
        # In HALF_OPEN — a failure should re-open
        try:
            with cb:
                raise ValueError("fail again")
        except ValueError:
            pass
        assert cb.state == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(name="test", threshold=3, cooldown_seconds=60)
        # 2 failures (under threshold)
        for _ in range(2):
            try:
                with cb:
                    raise ValueError("fail")
            except ValueError:
                pass
        # A success should reset the counter
        with cb:
            pass
        # Now 2 more failures shouldn't open it (counter was reset)
        for _ in range(2):
            try:
                with cb:
                    raise ValueError("fail")
            except ValueError:
                pass
        assert cb.state == CircuitState.CLOSED

    def test_reset(self):
        cb = CircuitBreaker(name="test", threshold=1, cooldown_seconds=60)
        try:
            with cb:
                raise ValueError("fail")
        except ValueError:
            pass
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_to_dict(self):
        cb = CircuitBreaker(name="test", threshold=5, cooldown_seconds=30)
        d = cb.to_dict()
        assert d["name"] == "test"
        assert d["state"] == "closed"
        assert d["threshold"] == 5


class TestCircuitBreakerRegistry:
    def test_get_or_create(self):
        reg = CircuitBreakerRegistry()
        cb1 = reg.get_or_create("backend-1")
        cb2 = reg.get_or_create("backend-1")
        assert cb1 is cb2  # same instance

    def test_get_returns_none_for_unknown(self):
        reg = CircuitBreakerRegistry()
        assert reg.get("unknown") is None

    def test_list_all(self):
        reg = CircuitBreakerRegistry()
        reg.get_or_create("backend-1")
        reg.get_or_create("backend-2")
        all_breakers = reg.list_all()
        assert len(all_breakers) == 2

    def test_reset_all(self):
        reg = CircuitBreakerRegistry()
        cb1 = reg.get_or_create("backend-1", threshold=1)
        cb2 = reg.get_or_create("backend-2", threshold=1)
        # Open both
        for cb in [cb1, cb2]:
            try:
                with cb:
                    raise ValueError("fail")
            except ValueError:
                pass
        reg.reset_all()
        assert cb1.state == CircuitState.CLOSED
        assert cb2.state == CircuitState.CLOSED

    def test_global_registry_singleton(self):
        reg1 = get_global_registry()
        reg2 = get_global_registry()
        assert reg1 is reg2
