.PHONY: help install dev-install server-install all-install \
        test test-fast test-cov test-integration test-property test-benchmarks \
        lint format typecheck clean \
        notebook serve-docs docs-build docs-strict \
        demo-phase1 demo-phase2 demo-phase3 demo-benchmarks \
        cli-info cli-run cli-bench cli-serve \
        docker-build docker-run docker-up docker-down \
        pre-commit-install pre-commit-run \
        smoke-test check-env

help: ## Show this help
        @grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package (core only, no backends)
        pip install -e .

dev-install: ## Install with dev dependencies + mock backend (no GPU needed)
        pip install -e ".[dev]"

server-install: ## Install with HTTP API server
        pip install -e ".[server]"

all-install: ## Install everything: HF + memory + server + notebooks + dev
        pip install -e ".[all]"

# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

test: ## Run the full test suite
        pytest

test-fast: ## Run only fast tests (skip slow / gpu / integration / benchmarks)
        pytest -m "not slow and not gpu and not integration" \
                --ignore=tests/integration \
                --ignore=tests/benchmarks

test-cov: ## Run tests with coverage report
        pytest --cov=chen --cov-report=term --cov-report=html

test-integration: ## Run integration tests (server)
        pytest tests/integration/ -m integration -v

test-property: ## Run property-based tests (Hypothesis)
        pytest tests/property/ -v

test-benchmarks: ## Run performance benchmarks
        pytest tests/benchmarks/ --benchmark-only

smoke-test: ## Run the smoke test
        python scripts/smoke_test.py

check-env: ## Check environment (deps, GPU, backends)
        python scripts/check_env.py

# --------------------------------------------------------------------------
# Code quality
# --------------------------------------------------------------------------

lint: ## Lint with ruff
        ruff check src tests

format: ## Auto-format with ruff
        ruff format src tests
        ruff check --fix src tests

typecheck: ## Type check with mypy
        mypy src/chen

pre-commit-install: ## Install pre-commit hooks
        pre-commit install

pre-commit-run: ## Run pre-commit on all files
        pre-commit run --all-files

# --------------------------------------------------------------------------
# Documentation
# --------------------------------------------------------------------------

serve-docs: ## Serve MkDocs site locally (http://127.0.0.1:8000)
        mkdocs serve

docs-build: ## Build MkDocs site (output: site/)
        mkdocs build

docs-strict: ## Build MkDocs site in strict mode (warnings = errors)
        mkdocs build --strict

# --------------------------------------------------------------------------
# Demos (Python examples)
# --------------------------------------------------------------------------

demo-phase1: ## Run Phase 1 demo (static cascade, mock backend)
        python examples/run_phase1.py --backend mock

demo-phase2: ## Run Phase 2 demo (KV-cache passing)
        python examples/run_phase2.py

demo-phase3: ## Run Phase 3 demo (dynamic routing)
        python examples/run_phase3.py --router logistic

demo-benchmarks: ## Run benchmark suite
        python examples/run_benchmarks.py --phase 1

# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

cli-info: ## CHEN CLI: print environment info
        chen info

cli-run: ## CHEN CLI: run a prompt
        chen run --prompt "Explain recursion." --phase 1 --backend mock

cli-bench: ## CHEN CLI: run benchmark suite
        chen bench --phase 1

cli-serve: ## CHEN CLI: start HTTP API server on :8000
        chen serve --host 0.0.0.0 --port 8000

# --------------------------------------------------------------------------
# Docker
# --------------------------------------------------------------------------

docker-build: ## Build the Docker image
        docker build -f docker/Dockerfile -t chen:latest .

docker-run: ## Run the Docker image (entrypoint: chen --help)
        docker run --rm chen:latest info

docker-up: ## Start CHEN API server + Prometheus via docker-compose
        cd docker && docker compose up -d

docker-down: ## Stop docker-compose services
        cd docker && docker compose down

# --------------------------------------------------------------------------
# Cleanup
# --------------------------------------------------------------------------

clean: ## Remove build artifacts and caches
        rm -rf build dist *.egg-info src/*.egg-info site
        rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov .benchmarks
        rm -rf chen_data
        find . -type d -name __pycache__ -exec rm -rf {} +
        find . -type d -name "*.egg-info" -exec rm -rf {} +

notebook: ## Start Jupyter Lab
        jupyter lab

