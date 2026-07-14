# Contributing to CHEN

First off, thank you for considering a contribution to **CHEN (Collaborative Heterogeneous Expert Network)**. This project exists to prove that a coordinated "garage" of small, specialized models can match the quality of a single monolithic model at a fraction of the compute cost. Every contribution — bug reports, new backends, new routing strategies, benchmark tasks, documentation fixes — moves that mission forward.

This document describes how to set up your environment, the conventions we follow, and the process for getting your change merged.

---

## Code of Conduct

Participation in this project is governed by the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to abide by its terms.

---

## Quick Start

```bash
# 1. Clone your fork
git clone https://github.com/<your-username>/chen.git
cd chen

# 2. Create a virtual environment (Python 3.9+)
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# 3. Install in editable mode with dev dependencies
make dev-install           # core + mock backend + pytest/ruff/mypy
# or, if you have a GPU and want the HF backend:
make all-install

# 4. Verify everything works
make test-fast
```

You should see all tests pass without needing a GPU or downloading any model weights. The `MockBackend` ships with the repo and is used by every test by default.

---

## Repository Layout

```
src/chen/
    backends/     # Pluggable inference backends (mock, hf, vllm, llama_cpp)
    core/         # Expert, Router, Memory, KV-cache, Pipeline, Config
    phases/       # Phase 1 (cascade), Phase 2 (KV-pass), Phase 3 (routing)
    benchmarks/   # KPIs (EPU, cost, latency) and benchmark tasks
tests/            # Pytest suite — mirrors src/ layout
examples/         # Runnable scripts for each phase + benchmarks
notebooks/        # Step-by-step walkthroughs
docs/             # ARCHITECTURE.md and related design docs
.github/workflows # CI pipeline
```

---

## Branch Naming

| Branch prefix | Use |
|---------------|-----|
| `feat/...`    | New feature or capability |
| `fix/...`     | Bug fix |
| `docs/...`    | Documentation only |
| `bench/...`   | New benchmark task or KPI |
| `refactor/...`| Internal refactor, no behavior change |

Examples: `feat/vllm-kv-pass`, `fix/router-determinism`, `docs/architecture-typo`.

---

## Commit Messages

We follow a lightweight version of [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

- **type**: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `bench`
- **scope** (optional): `router`, `memory`, `backends`, `kv-cache`, `benchmarks`, `ci`, etc.
- **subject**: imperative mood, lowercase, no period — e.g. `add cosine router similarity`
- **body**: explain *why*, not *what*. Wrap at ~72 chars.
- **footer**: reference issues or PRs, e.g. `Closes #42`.

Example:
```
feat(router): add prompt-embedding cosine router

Adds a deterministic cosine-similarity router that maps a prompt's
embedding to the closest expert prototype. Falls back to the
logistic router if embeddings are unavailable.

Closes #87
```

---

## Code Style

- **Formatter / linter**: `ruff`. Run `make format` to auto-format and `make lint` to check.
- **Line length**: 100 characters.
- **Type hints**: required on all public APIs in `src/chen/`. Internal helpers may omit them when obvious.
- **Imports**: `ruff` handles isort automatically — group stdlib, third-party, then local.
- **Naming**: `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_SNAKE` for module-level constants.
- **Docstrings**: Google-style. Every public class and function needs one.

```python
def route_prompt(prompt: str, experts: list[Expert]) -> list[str]:
    """Return the ordered list of expert names that should handle the prompt.

    Args:
        prompt: The user's raw input prompt.
        experts: The pool of available experts.

    Returns:
        A list of expert names, in the order they should be invoked.

    Raises:
        RouterError: If no expert can handle the prompt.
    """
```

---

## Tests

- Tests live in `tests/` and mirror the `src/chen/` structure.
- All new features **must** include tests that run under `make test-fast` (i.e. no GPU, no model downloads).
- If a test genuinely requires a GPU, mark it: `@pytest.mark.gpu`. It will be skipped in CI and in `make test-fast`.
- Integration tests (`@pytest.mark.integration`) may use real HF models but must be skipped by default.
- Aim for ≥85% coverage on the modules you touch. Run `make test-cov` to see the report.

---

## Adding a New Backend

If you want to add a new inference backend (e.g. TGI, Triton, ONNX Runtime):

1. Create `src/chen/backends/<your_backend>.py` implementing the `InferenceBackend` protocol from `backends/base.py`.
2. Register it in `backends/__init__.py` via the `BACKEND_REGISTRY`.
3. Add a config section in `src/chen/config.py`.
4. Add unit tests in `tests/test_backends.py` using a stubbed or mock model.
5. Update `.env.example` and the README "Backends" table.
6. Add a `[project.optional-dependencies]` entry in `pyproject.toml` if the backend needs extra packages.

---

## Adding a New Benchmark Task

1. Add the task definition to `src/chen/benchmarks/tasks.py` as a `BenchmarkTask` dataclass.
2. Include at least 5 sample prompts and a deterministic grader.
3. Register it in the `TASK_REGISTRY`.
4. Add a test in `tests/test_benchmarks.py` that runs the task against the MockBackend.
5. Document the task in `docs/benchmarks.md` (create the file if it doesn't exist).

---

## Pull Request Process

1. **Open an issue first** for non-trivial changes (new backends, new phases, breaking API changes). For typos and small fixes, just open the PR.
2. Fork the repo, create a branch from `main` using the naming convention above.
3. Write code + tests. Make sure `make test-fast`, `make lint`, and `make typecheck` all pass locally.
4. Update the CHANGELOG (or README if no CHANGELOG exists yet) if your change is user-facing.
5. Open the PR against `main`. Use a clear title following the commit-message convention.
6. Fill in the PR template (CI will remind you if you skip it).
7. Address review feedback by pushing new commits — do not force-push unless explicitly asked.

CI runs on every push: `ruff check`, `ruff format --check`, `mypy`, and `pytest -m "not slow and not gpu and not integration"`. All must pass before merge.

---

## Releasing

This project uses semantic versioning (`MAJOR.MINOR.PATCH`):

- **PATCH**: bug fixes, no API changes
- **MINOR**: new features, backward-compatible
- **MAJOR**: breaking API changes

Releases are tagged on `main` by maintainers. If you need a release for a specific fix, mention it in your PR.

---

## Questions?

- Open a [Discussion](https://github.com/your-org/chen/discussions) for questions about architecture or intended use.
- Open an [Issue](https://github.com/your-org/chen/issues) for bugs or feature requests.
- For private security disclosures, see `SECURITY.md` (if absent, email the maintainers directly).

Thank you for helping build a more decentralized future for LLM inference.
