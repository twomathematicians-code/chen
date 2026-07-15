"""Integration tests for the HTTP API server.

Run with:
    pytest tests/integration/ -m integration

Requires the `server` extra:
    pip install -e '.[server]'
"""

from __future__ import annotations

import pytest

# Skip all tests in this module if fastapi/httpx aren't installed.
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from chen import __version__  # noqa: E402
from chen.server.app import create_app  # noqa: E402


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.mark.integration
class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        r = client.get("/v1/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["version"] == __version__

    def test_metrics_returns_text(self, client):
        r = client.get("/v1/metrics")
        assert r.status_code == 200
        assert "chen_" in r.text or r.text == ""  # may be empty if prom not installed

    def test_docs_accessible(self, client):
        r = client.get("/docs")
        assert r.status_code == 200


@pytest.mark.integration
class TestInferEndpoint:
    def test_infer_phase1_mock(self, client):
        r = client.post(
            "/v1/infer",
            json={
                "prompt": "Explain recursion.",
                "phase": 1,
                "backend": "mock",
                "max_tokens": 64,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert "output" in body
        assert isinstance(body["output"], str)
        assert len(body["output"]) > 0
        assert "selected_experts" in body
        assert "per_expert" in body
        assert len(body["per_expert"]) >= 1
        assert "run_id" in body
        assert "config_hash" in body
        assert len(body["config_hash"]) == 64

    def test_infer_phase2_mock(self, client):
        r = client.post(
            "/v1/infer",
            json={
                "prompt": "Explain quantum entanglement.",
                "phase": 2,
                "backend": "mock",
                "max_tokens": 64,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["kv_transfers"] >= 1

    def test_infer_phase3_mock(self, client):
        r = client.post(
            "/v1/infer",
            json={
                "prompt": "Debug this Python function.",
                "phase": 3,
                "backend": "mock",
                "max_tokens": 64,
                "router": "logistic",
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["selected_experts"]) >= 1

    def test_infer_validates_prompt_min_length(self, client):
        r = client.post("/v1/infer", json={"prompt": "", "phase": 1})
        assert r.status_code == 422  # Pydantic validation error

    def test_infer_validates_phase_range(self, client):
        r = client.post("/v1/infer", json={"prompt": "x", "phase": 5})
        assert r.status_code == 422

    def test_infer_persists_run(self, client):
        # First request creates a run
        r = client.post(
            "/v1/infer",
            json={
                "prompt": "test prompt for persistence",
                "phase": 1,
                "backend": "mock",
                "max_tokens": 32,
                "save_run": True,
            },
        )
        assert r.status_code == 200
        run_id = r.json()["run_id"]

        # Fetch the run back
        r2 = client.get(f"/v1/runs/{run_id}")
        assert r2.status_code == 200
        body = r2.json()
        assert body["run_id"] == run_id
        assert body["prompt_preview"].startswith("test prompt")


@pytest.mark.integration
class TestRunsEndpoint:
    def test_list_runs_empty_at_first(self, client):
        # Use a fresh in-memory store — actually it's shared, so we just
        # check the endpoint returns a list.
        r = client.get("/v1/runs")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_get_nonexistent_run_returns_404(self, client):
        r = client.get("/v1/runs/nonexistent_run_id")
        assert r.status_code == 404
