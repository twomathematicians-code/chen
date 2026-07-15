"""Performance benchmarks (pytest-benchmark).

Run with:
    pytest tests/benchmarks/ --benchmark-only

These tests measure pipeline throughput on the MockBackend. They are
not measures of real model performance — they measure the overhead of
the orchestration layer (router, pipeline, metrics aggregation).

For real-model benchmarks, use `examples/run_benchmarks.py` with the
HuggingFaceBackend.
"""
