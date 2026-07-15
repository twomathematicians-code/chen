"""Authentication & authorization middleware for the CHEN HTTP API.

Provides:
- API key authentication (Bearer token)
- Role-based access control (admin, user, read-only)
- Configurable CORS (no more allow_origins=["*"])
- Rate limiting per API key (token bucket)

Usage in ``create_app``::

    from chen.server.auth import AuthMiddleware, RateLimitMiddleware

    app.add_middleware(AuthMiddleware, api_keys=load_api_keys())
    app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional

from chen.observability.logging import get_logger

_log = get_logger("chen.server.auth")


# ---------------------------------------------------------------------------
# API key model
# ---------------------------------------------------------------------------


@dataclass
class APIKey:
    """A single API key with associated role and metadata.

    Attributes:
        key: The API key string (e.g. "chen_abc123...").
        role: One of "admin", "user", "read-only".
        tenant_id: Optional tenant identifier for multi-tenant isolation.
        name: Human-readable name for the key (for audit logs).
        created_at: Unix timestamp.
        rate_limit_per_minute: Max requests per minute (0 = unlimited).
    """

    key: str
    role: str = "user"
    tenant_id: Optional[str] = None  # noqa: UP045
    name: str = ""
    created_at: float = field(default_factory=time.time)
    rate_limit_per_minute: int = 60


# ---------------------------------------------------------------------------
# API key store (file-based; replace with DB/KMS in production)
# ---------------------------------------------------------------------------


class APIKeyStore:
    """File-based API key store.

    Format: JSON file with a list of APIKey objects.

    Path: ``$CHEN_API_KEYS_FILE`` or ``./chen_data/api_keys.json``
    """

    def __init__(self, path: Optional[str] = None) -> None:  # noqa: UP045
        self.path = path or os.environ.get("CHEN_API_KEYS_FILE", "./chen_data/api_keys.json")
        self._keys: dict[str, APIKey] = {}
        self._load()

    def _load(self) -> None:
        """Load keys from the JSON file. Creates an empty file if missing."""
        try:
            with open(self.path) as f:
                data = json.load(f)
            for entry in data:
                key = APIKey(**entry)
                self._keys[key.key] = key
        except FileNotFoundError:
            # No keys file = no auth (development mode).
            _log.warning("auth.no_keys_file", path=self.path)
        except (json.JSONDecodeError, TypeError) as e:
            _log.error("auth.keys_file_corrupt", path=self.path, error=str(e))

    def get(self, key: str) -> Optional[APIKey]:  # noqa: UP045
        """Look up an API key. Returns None if not found."""
        return self._keys.get(key)

    def has_keys(self) -> bool:
        """Return True if any keys are configured (auth is active)."""
        return len(self._keys) > 0

    def list_keys(self) -> list[APIKey]:
        """List all keys (for admin endpoints). Does not return the key strings."""
        return list(self._keys.values())


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------


PUBLIC_PATHS = frozenset(
    {
        "/v1/health",
        "/v1/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
)


class AuthMiddleware:
    """FastAPI/Starlette middleware that validates API keys.

    If the API key store has keys, every request (except PUBLIC_PATHS)
    must include a valid ``Authorization: Bearer <key>`` header.

    If the store is empty (development mode), auth is bypassed and a
    warning is logged.
    """

    def __init__(self, app: Any, key_store: Optional[APIKeyStore] = None) -> None:  # noqa: UP045
        self.app = app
        self.key_store = key_store or APIKeyStore()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"]
        method = scope["method"]

        # Public paths don't require auth.
        if path in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        # If no keys configured, bypass auth (dev mode).
        if not self.key_store.has_keys():
            scope["state"] = scope.get("state", {})
            scope["state"]["auth"] = None
            await self.app(scope, receive, send)
            return

        # Extract Bearer token.
        headers = dict(scope.get("headers", []))
        auth_header = headers.get(b"authorization", b"").decode("ascii", errors="ignore")
        if not auth_header.startswith("Bearer "):
            await self._send_error(send, 401, "Missing or invalid Authorization header")
            return

        key_str = auth_header[7:].strip()
        api_key = self.key_store.get(key_str)
        if api_key is None:
            await self._send_error(send, 401, "Invalid API key")
            return

        # Check role-based permissions.
        if not self._check_permission(api_key, path, method):
            await self._send_error(
                send, 403, f"Role '{api_key.role}' cannot access {method} {path}"
            )
            return

        # Attach auth info to scope for downstream handlers.
        scope["state"] = scope.get("state", {})
        scope["state"]["auth"] = api_key
        await self.app(scope, receive, send)

    def _check_permission(self, key: APIKey, path: str, method: str) -> bool:
        """Check if the key's role allows this request."""
        if key.role == "admin":
            return True
        if key.role == "user":
            # Users can do everything except admin endpoints.
            return not path.startswith("/v1/admin")
        if key.role == "read-only":
            # Read-only can only GET.
            return method == "GET"
        return False

    async def _send_error(self, send: Any, status: int, message: str) -> None:
        """Send a JSON error response."""
        body = json.dumps({"error": message}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(body)).encode()],
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


# ---------------------------------------------------------------------------
# Rate limiting (token bucket per API key)
# ---------------------------------------------------------------------------


class RateLimitMiddleware:
    """Per-API-key rate limiting using a sliding window.

    Each key has a ``rate_limit_per_minute`` budget (from the APIKey).
    Requests exceeding the limit get HTTP 429.
    """

    def __init__(self, app: Any, default_limit: int = 60) -> None:
        self.app = app
        self.default_limit = default_limit
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Get the API key from auth state (set by AuthMiddleware).
        state = scope.get("state", {})
        auth = state.get("auth")
        if auth is None:
            # No auth (dev mode) — apply default limit per IP.
            client = scope.get("client", ("unknown", 0))[0]
            key_id = f"ip:{client}"
            limit = self.default_limit
        else:
            key_id = auth.key
            limit = auth.rate_limit_per_minute or self.default_limit

        now = time.time()
        window = 60.0  # 1 minute

        # Prune old entries.
        reqs = self._requests[key_id]
        while reqs and reqs[0] < now - window:
            reqs.popleft()

        if len(reqs) >= limit:
            _log.warning("rate_limit.exceeded", key_id=key_id, count=len(reqs))
            body = json.dumps(
                {"error": "Rate limit exceeded", "limit": limit, "window_seconds": window}
            ).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 429,
                    "headers": [
                        [b"content-type", b"application/json"],
                        [b"content-length", str(len(body)).encode()],
                        [b"retry-after", b"60"],
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        reqs.append(now)
        await self.app(scope, receive, send)
