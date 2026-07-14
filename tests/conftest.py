"""Shared pytest fixtures for the CHEN test suite."""

from __future__ import annotations

import pytest

from chen.backends.mock import MockBackend
from chen.core.expert import Expert, ExpertRole
from chen.core.memory import InMemoryMemory


@pytest.fixture
def mock_backend_small() -> MockBackend:
    """A small 3B mock backend."""
    return MockBackend(params_m=3_000, role_hint="analyst")


@pytest.fixture
def mock_backend_large() -> MockBackend:
    """A larger 8B mock backend."""
    return MockBackend(params_m=8_000, role_hint="reasoner")


@pytest.fixture
def three_experts() -> list[Expert]:
    """The canonical 3-expert garage: Analyst, Reasoner, Synthesizer."""
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


@pytest.fixture
def four_experts() -> list[Expert]:
    """A 4-expert garage: Analyst, Reasoner, Coder, Synthesizer."""
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
            name="coder",
            role=ExpertRole.CODER,
            backend=MockBackend(params_m=7_000, role_hint="coder"),
        ),
        Expert(
            name="synthesizer",
            role=ExpertRole.SYNTHESIZER,
            backend=MockBackend(params_m=3_000, role_hint="synthesizer"),
        ),
    ]


@pytest.fixture
def fresh_memory() -> InMemoryMemory:
    """A fresh in-memory store (no entries)."""
    return InMemoryMemory()
