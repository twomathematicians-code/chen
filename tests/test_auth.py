"""Tests for authentication, rate limiting, and authorization."""

from __future__ import annotations

import json

import pytest

from chen.server.auth import APIKey, APIKeyStore, AuthMiddleware, RateLimitMiddleware


@pytest.fixture
def key_store_with_keys(tmp_path):
    """A key store with test keys."""
    keys_file = tmp_path / "keys.json"
    keys_data = [
        {
            "key": "chen_admin_123",
            "role": "admin",
            "name": "Admin Key",
            "rate_limit_per_minute": 100,
        },
        {
            "key": "chen_user_456",
            "role": "user",
            "name": "User Key",
            "rate_limit_per_minute": 60,
        },
        {
            "key": "chen_readonly_789",
            "role": "read-only",
            "name": "Read-Only Key",
            "rate_limit_per_minute": 30,
        },
    ]
    keys_file.write_text(json.dumps(keys_data))
    return APIKeyStore(path=str(keys_file))


@pytest.fixture
def empty_key_store(tmp_path):
    """A key store with no keys (dev mode)."""
    return APIKeyStore(path=str(tmp_path / "nonexistent.json"))


class TestAPIKey:
    def test_default_role_is_user(self):
        key = APIKey(key="test")
        assert key.role == "user"

    def test_custom_role(self):
        key = APIKey(key="test", role="admin")
        assert key.role == "admin"


class TestAPIKeyStore:
    def test_has_keys_false_when_empty(self, empty_key_store):
        assert empty_key_store.has_keys() is False

    def test_has_keys_true_when_populated(self, key_store_with_keys):
        assert key_store_with_keys.has_keys() is True

    def test_get_valid_key(self, key_store_with_keys):
        key = key_store_with_keys.get("chen_admin_123")
        assert key is not None
        assert key.role == "admin"
        assert key.name == "Admin Key"

    def test_get_invalid_key_returns_none(self, key_store_with_keys):
        assert key_store_with_keys.get("invalid_key") is None

    def test_list_keys(self, key_store_with_keys):
        keys = key_store_with_keys.list_keys()
        assert len(keys) == 3

    def test_loads_from_env(self, tmp_path, monkeypatch):
        keys_file = tmp_path / "env_keys.json"
        keys_file.write_text(json.dumps([{"key": "env_key", "role": "user"}]))
        monkeypatch.setenv("CHEN_API_KEYS_FILE", str(keys_file))
        store = APIKeyStore()
        assert store.has_keys()
        assert store.get("env_key") is not None


class TestAuthMiddleware:
    @pytest.fixture
    def app_with_auth(self, key_store_with_keys):
        """Build a minimal ASGI app wrapped with AuthMiddleware."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()

        @app.get("/v1/secret")
        async def secret():
            return {"data": "top secret"}

        @app.get("/v1/health")
        async def health():
            return {"status": "ok"}

        # Wrap with auth middleware manually
        original_app = app  # noqa: F841
        wrapped = AuthMiddleware(app, key_store_with_keys)
        # TestClient expects an ASGI app
        return TestClient(wrapped)

    @pytest.fixture
    def app_no_auth(self, empty_key_store):
        """App with no keys configured (dev mode — auth bypassed)."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()

        @app.get("/v1/secret")
        async def secret():
            return {"data": "top secret"}

        wrapped = AuthMiddleware(app, empty_key_store)
        return TestClient(wrapped)

    def test_public_path_no_auth_needed(self, app_with_auth):
        r = app_with_auth.get("/v1/health")
        assert r.status_code == 200

    def test_protected_path_without_auth_returns_401(self, app_with_auth):
        r = app_with_auth.get("/v1/secret")
        assert r.status_code == 401

    def test_protected_path_with_invalid_auth_returns_401(self, app_with_auth):
        r = app_with_auth.get("/v1/secret", headers={"Authorization": "Bearer invalid_key"})
        assert r.status_code == 401

    def test_protected_path_with_valid_auth_returns_200(self, app_with_auth):
        r = app_with_auth.get("/v1/secret", headers={"Authorization": "Bearer chen_user_456"})
        assert r.status_code == 200

    def test_no_keys_bypasses_auth(self, app_no_auth):
        r = app_no_auth.get("/v1/secret")
        assert r.status_code == 200


class TestRateLimitMiddleware:
    @pytest.fixture
    def rate_limited_app(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"ok": True}

        # Wrap with rate limiting — 3 per minute
        wrapped = RateLimitMiddleware(app, default_limit=3)
        return TestClient(wrapped)

    def test_allows_under_limit(self, rate_limited_app):
        r = rate_limited_app.get("/test")
        assert r.status_code == 200

    def test_blocks_over_limit(self, rate_limited_app):
        # Make 3 requests (the limit)
        for _ in range(3):
            rate_limited_app.get("/test")
        # 4th should be rate limited
        r = rate_limited_app.get("/test")
        assert r.status_code == 429
        assert "Rate limit exceeded" in r.json()["error"]

    def test_rate_limit_returns_retry_after(self, rate_limited_app):
        for _ in range(3):
            rate_limited_app.get("/test")
        r = rate_limited_app.get("/test")
        assert r.status_code == 429
        assert "retry-after" in {k.lower() for k in r.headers.keys()}
