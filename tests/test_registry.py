"""Tests for the backend registry."""

from __future__ import annotations

import pytest

from chen.backends import (
    BACKEND_REGISTRY,
    get_backend,
    list_backends,
    register_backend,
)
from chen.backends.base import BackendError
from chen.backends.mock import MockBackend


class TestRegistry:
    def test_mock_backend_registered(self):
        assert "mock" in BACKEND_REGISTRY

    def test_list_backends_includes_mock(self):
        names = list_backends()
        assert "mock" in names

    def test_get_backend_returns_instance(self):
        b = get_backend("mock", params_m=3_000, role_hint="test")
        assert isinstance(b, MockBackend)
        assert b.params_m == 3_000

    def test_get_backend_case_insensitive(self):
        b = get_backend("MOCK")
        assert isinstance(b, MockBackend)

    def test_get_unknown_backend_raises(self):
        with pytest.raises(BackendError, match="Unknown backend"):
            get_backend("nonexistent")

    def test_register_custom_backend(self):
        class MyBackend(MockBackend):
            pass

        register_backend("my-backend", MyBackend)
        try:
            assert "my-backend" in BACKEND_REGISTRY
            b = get_backend("my-backend")
            assert isinstance(b, MyBackend)
        finally:
            BACKEND_REGISTRY.pop("my-backend", None)

    def test_duplicate_registration_same_factory_is_idempotent(self):
        register_backend("mock", MockBackend)  # should not raise
        assert BACKEND_REGISTRY["mock"] is MockBackend

    def test_duplicate_registration_different_factory_raises(self):
        class Other(MockBackend):
            pass

        with pytest.raises(BackendError, match="already registered"):
            register_backend("mock", Other)
