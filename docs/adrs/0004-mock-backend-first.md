# ADR 0004: Mock backend first — every test runs without a GPU

- Status: Accepted
- Date: 2025-01-15
- Deciders: CHEN core team

## Context

LLM projects commonly have a problem: their test suite requires
downloading multi-GB model weights and (often) a CUDA GPU. This makes
CI slow, expensive, and inaccessible to contributors on laptops.

CHEN has 141 tests. If each required real model inference, CI would
take ~30 minutes and cost real money per run. Worse, tests would be
non-deterministic (LLMs sample).

## Decision

Ship a `MockBackend` that:

1. Implements the full `InferenceBackend` protocol (including KV-cache encode/decode/transfer).
2. Is **deterministic** — same prompt → same output, byte-identical.
3. Requires **no dependencies** beyond numpy.
4. Runs in **<1 ms** per call.
5. Produces output that is **structured enough** for downstream experts to operate on (role hint, prompt hash, source-text snippet).

Every test in the suite uses `MockBackend` by default. Tests that
genuinely require a GPU are marked `@pytest.mark.gpu` and skipped in CI.

## Consequences

### Positive

- CI runs in ~1 second.
- Contributors without a GPU can run the full test suite.
- Determinism — failing tests fail every time, not flakily.
- The MockBackend serves as the protocol reference implementation.

### Negative

- The MockBackend's "output" is structured gibberish, not real language. Tests can verify *plumbing* but not *quality*.
- Risk of false confidence — a green test suite doesn't mean the real models work.
- The KV-cache produced by MockBackend is arbitrary (hash-based), so cross-family transfer tests don't validate real projection logic.

### Neutral

- The `mock_friendly_grader` in `benchmarks/tasks.py` acknowledges this — it gives partial credit for keyword overlap, since the mock output won't contain real answers.

## Mitigations for the negatives

- The `examples/run_real_models.py` script exists specifically to run the pipeline against real HF models. It is run manually before releases.
- The HuggingFace backend has its own test file (`tests/test_hf_backend.py` — when added) gated behind `@pytest.mark.integration` and skipped in fast CI.

## Alternatives considered

### Alternative A: Use a tiny real model (e.g. GPT-2 small) in tests

Download `gpt2` (500 MB) once per CI run.

**Why not:** still ~30s per test, non-deterministic, and 500 MB of
cache per CI run is wasteful.

### Alternative B: Mock at the function level, not the backend level

Patch `model.generate` to return canned strings.

**Why not:** this couples tests to the HF backend's internals. We want
to test the *protocol*, not one implementation.
