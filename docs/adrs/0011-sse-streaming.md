# ADR 0011: SSE streaming for real-time pipeline output

- Status: Accepted
- Date: 2025-07-15
- Deciders: CHEN core team

## Context

v0.2.0's `POST /v1/infer` endpoint was synchronous — the client sends
a request and waits for the entire pipeline to finish before getting a
response. For a 3-expert pipeline with 2-second per-expert latency,
the client waits ~6 seconds with no feedback.

This is problematic for:

- **User experience**: chat UIs feel dead while waiting.
- **Timeouts**: load balancers and proxies may time out before the pipeline finishes.
- **Progress feedback**: users want to see which expert is running, not just the final output.

## Decision

Add a streaming endpoint (`POST /v1/infer/stream`) that uses
Server-Sent Events (SSE) to push per-expert progress to the client in
real time.

### Protocol

SSE is a standard HTTP protocol for server-to-client streaming:

```
POST /v1/infer/stream HTTP/1.1
Content-Type: application/json

{"prompt": "...", "phase": 1, "backend": "mock"}
```

Response (streamed):

```
data: {"type": "expert", "index": 0, "expert_name": "analyst", "role": "analyst", "latency_ms": 1.2}

data: {"type": "expert", "index": 1, "expert_name": "synthesizer", "role": "synthesizer", "latency_ms": 0.8}

event: done
data: {"type": "done", "output": "...", "total_tokens": 145, "run_id": "abc123", ...}
```

### Implementation

- The pipeline runs in a thread pool (`asyncio.run_in_executor`) to
  avoid blocking the FastAPI event loop.
- After the pipeline completes, per-expert metrics are streamed as
  individual SSE events.
- A final `event: done` message contains the full result (output,
  metrics, run_id).
- Response headers disable proxy buffering: `X-Accel-Buffering: no`,
  `Cache-Control: no-cache`.

### Future: token-level streaming

v0.3.0 streams per-expert events (one event per expert). v0.4.0 will
add token-level streaming — each generated token is streamed as an SSE
event, giving true real-time output. This requires the backends to
support `stream=True` in their `generate()` methods.

## Consequences

### Positive

- Clients get real-time feedback — no more 6-second dead waits.
- Proxies won't time out — the connection is kept alive by SSE events.
- Progress is observable — clients can show "Analyst running... Reasoner running... Synthesizer running..."
- SSE is standard HTTP — works through proxies, firewalls, and CDNs without special configuration.

### Negative

- More complex client implementation — must parse SSE stream rather than a single JSON response.
- Pipeline still runs synchronously internally — the thread pool just moves the blocking work off the event loop. True async pipeline is roadmap.
- No backpressure — if the client is slow, events buffer in memory. For high-throughput, consider WebSocket instead.

### Neutral

- SSE is one-directional (server→client). If bidirectional communication is needed (e.g., client cancels mid-pipeline), use WebSocket instead.
- The `stream` field on `InferRequest` is informational — the streaming endpoint is separate (`/v1/infer/stream`). Clients choose which endpoint to call.

## Alternatives considered

### Alternative A: WebSocket

Use WebSocket for bidirectional streaming.

**Why not:** WebSocket is overkill for the common case (client sends one request, server streams response). WebSocket requires a protocol upgrade, doesn't work through all proxies, and is harder to secure. SSE is simpler and sufficient for v0.3.0. WebSocket is roadmap for interactive use cases.

### Alternative B: NDJSON (newline-delimited JSON)

Stream newline-delimited JSON objects instead of SSE.

**Why not:** NDJSON is simpler but lacks the `event:` type field that SSE provides. SSE is also a recognized media type (`text/event-stream`) that proxies handle correctly (disabling buffering, keeping connection alive). NDJSON is more brittle in practice.

### Alternative C: gRPC streaming

Use gRPC server-side streaming.

**Why not:** gRPC requires HTTP/2, protobuf schema management, and a gRPC client. SSE works over HTTP/1.1 with any HTTP client. For a REST-first API, SSE is the right choice. gRPC is roadmap for internal service-to-service communication.
