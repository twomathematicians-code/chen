"""Locust load test for the CHEN HTTP API.

Usage::

    pip install locust
    locust -f tests/load/locustfile.py --host http://localhost:8000

Then open http://localhost:8089 to configure the load test.

SLOs:
    - p50 latency < 2s (Phase 1, mock backend)
    - p99 latency < 5s (Phase 1, mock backend)
    - error rate < 0.1%

For real-backend testing, change BACKEND to "hf" and set up the HF
environment variables.
"""

from __future__ import annotations

import os
import random

from locust import HttpUser, between, task

# Configuration via environment.
BACKEND = os.environ.get("CHEN_LOAD_BACKEND", "mock")
PROMPTS = [
    "Explain recursion to a 12-year-old.",
    "Write a haiku about autumn.",
    "What is the capital of France?",
    "Debug this Python code: def foo(): return None",
    "Summarize: CHEN is a distributed inference architecture.",
    "What is 17 * 23?",
    "Write a Python function that reverses a string.",
    "If today is Monday, what day is it 100 days from now?",
    "Explain quantum entanglement.",
    "Write a short essay about climate change.",
]


class ChenUser(HttpUser):
    """Simulates a user hitting the CHEN API."""

    wait_time = between(1, 3)  # 1-3 seconds between requests

    def on_start(self) -> None:
        """Called when a user starts."""
        # Optional: set up auth header if testing with auth enabled.
        api_key = os.environ.get("CHEN_LOAD_API_KEY", "")
        if api_key:
            self.client.headers.update({"Authorization": f"Bearer {api_key}"})

    @task(3)
    def infer_phase1(self) -> None:
        """Phase 1 — static cascade (most common request)."""
        prompt = random.choice(PROMPTS)
        with self.client.post(
            "/v1/infer",
            json={
                "prompt": prompt,
                "phase": 1,
                "backend": BACKEND,
                "max_tokens": 64,
            },
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                body = response.json()
                if "output" not in body:
                    response.failure("Missing 'output' in response")
            elif response.status_code == 429:
                response.failure("Rate limited")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def infer_phase3(self) -> None:
        """Phase 3 — dynamic routing (less common)."""
        prompt = random.choice(PROMPTS)
        self.client.post(
            "/v1/infer",
            json={
                "prompt": prompt,
                "phase": 3,
                "backend": BACKEND,
                "max_tokens": 64,
                "router": "logistic",
            },
        )

    @task(1)
    def health_check(self) -> None:
        """Health check (lightweight)."""
        self.client.get("/v1/health")

    @task(1)
    def list_runs(self) -> None:
        """List recent runs (read-only)."""
        self.client.get("/v1/runs?limit=5")
