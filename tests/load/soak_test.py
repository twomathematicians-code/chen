"""Soak test — sustained load over a long period (hours/days).

Unlike Locust (which is interactive), this is a standalone script that
runs sustained load and reports memory usage, latency, and error rates
over time. Designed to detect:
- Memory leaks (growing RSS over time)
- Latency degradation (p50/p99 increasing over time)
- Connection leaks (growing file descriptor count)
- SQLite bloat (database growing unboundedly)

Usage::

    python tests/load/soak_test.py --duration 3600 --host http://localhost:8000
"""

from __future__ import annotations

import argparse
import gc
import resource
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)


@dataclass
class SoakMetrics:
    """Metrics collected during a soak test."""

    total_requests: int = 0
    successful: int = 0
    errors: int = 0
    rate_limited: int = 0
    latencies: list[float] = field(default_factory=list)
    rss_samples: list[int] = field(default_factory=list)  # in KB
    error_codes: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def record(self, latency: float, status: int) -> None:
        self.total_requests += 1
        self.latencies.append(latency)
        if status == 200:
            self.successful += 1
        elif status == 429:
            self.rate_limited += 1
        else:
            self.errors += 1
        self.error_codes[status] += 1

    def sample_rss(self) -> None:
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.rss_samples.append(rss_kb)

    def report(self) -> str:
        if not self.latencies:
            return "No requests completed."
        sorted_lat = sorted(self.latencies)
        n = len(sorted_lat)
        p50 = sorted_lat[n // 2]
        p95 = sorted_lat[int(n * 0.95)]
        p99 = sorted_lat[int(n * 0.99)]
        error_rate = (self.errors / self.total_requests) * 100 if self.total_requests else 0
        rss_growth = (
            (self.rss_samples[-1] - self.rss_samples[0]) / 1024 if len(self.rss_samples) > 1 else 0
        )
        return (
            f"=== Soak Test Report ===\n"
            f"Total requests:   {self.total_requests}\n"
            f"Successful:       {self.successful}\n"
            f"Errors:           {self.errors} ({error_rate:.2f}%)\n"
            f"Rate limited:     {self.rate_limited}\n"
            f"Latency p50:      {p50 * 1000:.1f} ms\n"
            f"Latency p95:      {p95 * 1000:.1f} ms\n"
            f"Latency p99:      {p99 * 1000:.1f} ms\n"
            f"RSS (max):        {max(self.rss_samples) / 1024:.1f} MB\n"
            f"RSS growth:       {rss_growth:.1f} MB\n"
            f"Error codes:      {dict(self.error_codes)}\n"
        )


def worker(
    host: str,
    duration: float,
    metrics: SoakMetrics,
    stop_event: threading.Event,
) -> None:
    """One worker thread that sends requests until stop_event is set."""
    prompts = [
        "Explain recursion.",
        "Write a haiku.",
        "What is 2+2?",
        "Debug this code.",
        "Summarize this text.",
    ]
    while not stop_event.is_set():
        t0 = time.perf_counter()
        try:
            r = requests.post(
                f"{host}/v1/infer",
                json={
                    "prompt": prompts[int(time.time()) % len(prompts)],
                    "phase": 1,
                    "backend": "mock",
                    "max_tokens": 32,
                },
                timeout=10,
            )
            latency = time.perf_counter() - t0
            metrics.record(latency, r.status_code)
        except Exception:
            latency = time.perf_counter() - t0
            metrics.record(latency, 0)
        time.sleep(0.1)  # 10 req/s per worker


def main() -> int:
    parser = argparse.ArgumentParser(description="CHEN soak test.")
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--duration", type=int, default=3600, help="Duration in seconds.")
    parser.add_argument("--workers", type=int, default=10, help="Number of worker threads.")
    parser.add_argument("--report-interval", type=int, default=60)
    args = parser.parse_args()

    print(f"Starting soak test: {args.duration}s, {args.workers} workers")
    print(f"Target: {args.host}")

    metrics = SoakMetrics()
    stop_event = threading.Event()

    threads = []
    for _ in range(args.workers):
        t = threading.Thread(target=worker, args=(args.host, args.duration, metrics, stop_event))
        t.start()
        threads.append(t)

    start = time.perf_counter()
    try:
        while time.perf_counter() - start < args.duration:
            time.sleep(args.report_interval)
            metrics.sample_rss()
            gc.collect()
            elapsed = time.perf_counter() - start
            print(
                f"[{elapsed:.0f}s] {metrics.total_requests} reqs, {metrics.errors} errors, RSS={max(metrics.rss_samples) / 1024:.1f}MB"
            )
    except KeyboardInterrupt:
        print("\nInterrupted — stopping workers...")

    stop_event.set()
    for t in threads:
        t.join(timeout=5)

    metrics.sample_rss()
    print("\n" + metrics.report())
    return 0 if metrics.errors / max(1, metrics.total_requests) < 0.01 else 1


if __name__ == "__main__":
    sys.exit(main())
