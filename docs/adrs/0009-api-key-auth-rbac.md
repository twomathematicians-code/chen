# ADR 0009: API key authentication with RBAC

- Status: Accepted
- Date: 2025-07-15
- Deciders: CHEN core team

## Context

v0.2.0's threat model admitted: "anyone who can reach the port can use
it." The HTTP API had no authentication, no authorization, and
`allow_origins=["*"]` in CORS. This is acceptable for local development
but blocks deployment in any regulated environment.

For CHEN to be usable by financial firms, government agencies, and
healthcare providers, it needs:

1. **Authentication** — prove who you are.
2. **Authorization** — what you're allowed to do.
3. **Rate limiting** — prevent abuse and DoS.
4. **Configurable CORS** — restrict origins in production.

## Decision

Implement API key authentication with role-based access control (RBAC),
per-key rate limiting, and configurable CORS.

### Authentication

- API keys are stored in a JSON file (`chen_data/api_keys.json`).
- Clients send `Authorization: Bearer <key>` header.
- `AuthMiddleware` validates the key against the store.
- If no keys file exists, auth is **bypassed** (development mode) and a
  warning is logged.
- Public paths (`/v1/health`, `/v1/metrics`, `/docs`) bypass auth.

### Authorization (RBAC)

Three roles:

| Role | Can access |
|------|------------|
| `admin` | All endpoints (including `/v1/admin/*`) |
| `user` | All endpoints except `/v1/admin/*` |
| `read-only` | GET requests only |

### Rate limiting

- Per-key sliding-window limit (default 60 req/min).
- Configurable per key via `rate_limit_per_minute` field.
- HTTP 429 with `Retry-After: 60` header when exceeded.
- Applied before auth (so unauthenticated requests are also limited).

### CORS

- `CHEN_CORS_ORIGINS` env var (comma-separated).
- Default `*` for development.
- When not wildcard, `allow_credentials=True` and methods/headers are
  restricted.

## Consequences

### Positive

- CHEN can be deployed in regulated environments (fintech, govtech, healthcare).
- Rate limiting protects against DoS and runaway clients.
- RBAC enables multi-team deployments with different access levels.
- CORS configuration prevents cross-origin attacks in production.

### Negative

- File-based key store is simple but not enterprise-grade. For production,
  replace with an HSM, KMS, or HashiCorp Vault integration.
- No OAuth2/OIDC support yet — API keys are the only auth method.
- Rate limiting is in-memory (per-process) — not shared across replicas.
  For multi-replica, use Redis-backed rate limiting.

### Neutral

- The auth middleware is ASGI-compatible and works with any Starlette/FastAPI app.
- The key store interface is pluggable — swap the file-based store for a DB-backed one without changing the middleware.

## Alternatives considered

### Alternative A: OAuth2 / OIDC

Use OAuth2 with JWT tokens from an external identity provider (Auth0, Okta, Keycloak).

**Why not for v0.3.0:** adds a hard dependency on an external IdP. API keys are simpler and sufficient for single-organization deployments. OAuth2 support is planned for v0.4.0 as a pluggable auth provider.

### Alternative B: mTLS (mutual TLS)

Require client certificates at the TLS layer.

**Why not:** mTLS is excellent for service-to-service auth but cumbersome for human/CLI users. API keys are more ergonomic for the CLI use case. mTLS can be added at the load balancer layer without changing CHEN.

### Alternative C: No auth (trust the network)

Rely on network-level isolation (VPN, private subnet).

**Why not:** defense in depth. Network isolation fails; auth is a second layer. Also, some deployment topologies (shared k8s cluster) can't guarantee network isolation.
