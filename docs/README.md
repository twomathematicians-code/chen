# CHEN Documentation

This directory contains design and reference documentation for CHEN. For tutorials and walkthroughs, see the [`notebooks/`](../notebooks) directory.

## Index

| Document | Description |
|----------|-------------|
| [`ARCHITECTURE.md`](../ARCHITECTURE.md) | Canonical reference for the three pillars, data flow, KV-cache protocol, router design, backend abstraction, cost model, experimental phases, failure modes, and open questions. **Read this first** if you want to understand how CHEN works or contribute a new backend / router / benchmark task. |
| [`../README.md`](../README.md) | Quick start, install, usage examples, and high-level feature overview. Start here if you're new to the project. |
| [`../CHANGELOG.md`](../CHANGELOG.md) | Versioned history of notable changes. Follows [Keep a Changelog](https://keepachangelog.com/) format. |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | How to set up your dev environment, code style conventions, branch naming, commit messages, PR process, and how to add a new backend / benchmark task. |
| [`../SECURITY.md`](../SECURITY.md) | Supported versions, vulnerability reporting process, safe usage guidelines. |
| [`../CODE_OF_CONDUCT.md`](../CODE_OF_CONDUCT.md) | Contributor Covenant Code of Conduct. |
| [`../CITATION.cff`](../CITATION.cff) | Citation metadata for academic use. |

## API Reference

CHEN's public API is documented inline in the source via Google-style docstrings. To generate an HTML API reference:

```bash
pip install sphinx sphinx-rtd-theme
sphinx-quickstart docs/api
# Add 'chen' to the intersphinx mapping and autodoc extensions
sphinx-build docs/api docs/api/_build
```

(MkDocs is also supported — see `make serve-docs`.)

## Notebooks

| Notebook | Description |
|----------|-------------|
| [`01_phase1_cascade.ipynb`](../notebooks/01_phase1_cascade.ipynb) | Static cascade: text handoff between 3 experts. |
| [`02_phase2_kv_pass.ipynb`](../notebooks/02_phase2_kv_pass.ipynb) | KV-cache passing: latent state handoff. |
| [`03_phase3_routing.ipynb`](../notebooks/03_phase3_routing.ipynb) | Dynamic routing: router activates different expert subsets per prompt. |

## Examples

| Script | Description |
|--------|-------------|
| [`examples/run_phase1.py`](../examples/run_phase1.py) | Run Phase 1 on a sample prompt. |
| [`examples/run_phase2.py`](../examples/run_phase2.py) | Run Phase 2 and compare against Phase 1. |
| [`examples/run_phase3.py`](../examples/run_phase3.py) | Run Phase 3 with three sample prompts and three router variants. |
| [`examples/run_benchmarks.py`](../examples/run_benchmarks.py) | Run the full benchmark suite and print KPI reports. |

## Roadmap

See the [Roadmap section of the README](../README.md#roadmap) and the `[Unreleased]` section of the [Changelog](../CHANGELOG.md#unreleased) for planned work.

## Getting Help

- **Bug reports & feature requests:** [GitHub Issues](https://github.com/your-org/chen/issues)
- **Architecture & usage questions:** [GitHub Discussions](https://github.com/your-org/chen/discussions)
- **Security disclosures:** See [`SECURITY.md`](../SECURITY.md) — do **not** open a public issue.
