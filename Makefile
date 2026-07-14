.PHONY: help install dev-install test test-cov lint format typecheck clean notebook serve-docs

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package (core only, no backends)
	pip install -e .

dev-install: ## Install with dev dependencies + mock backend (no GPU needed)
	pip install -e ".[dev]"

all-install: ## Install everything: HF backend + memory + notebooks + dev
	pip install -e ".[all]"

test: ## Run the test suite
	pytest

test-fast: ## Run only fast tests (skip slow / gpu / integration)
	pytest -m "not slow and not gpu and not integration"

test-cov: ## Run tests with coverage report
	pytest --cov=chen --cov-report=term --cov-report=html

lint: ## Lint with ruff
	ruff check src tests

format: ## Auto-format with ruff
	ruff format src tests
	ruff check --fix src tests

typecheck: ## Type check with mypy
	mypy src/chen

clean: ## Remove build artifacts and caches
	rm -rf build dist *.egg-info src/*.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

notebook: ## Start Jupyter Lab
	jupyter lab

serve-docs: ## Serve docs locally
	@echo "Open README.md and ARCHITECTURE.md in your editor, or use mkdocs:"
	@echo "  pip install mkdocs mkdocs-material"
	@echo "  mkdocs serve"

# Run the 3 phases
demo-phase1: ## Run Phase 1 demo (static cascade)
	python examples/run_phase1.py

demo-phase2: ## Run Phase 2 demo (KV-cache passing)
	python examples/run_phase2.py

demo-phase3: ## Run Phase 3 demo (dynamic routing)
	python examples/run_phase3.py

demo-benchmarks: ## Run benchmark suite
	python examples/run_benchmarks.py
