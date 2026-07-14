#!/usr/bin/env bash
# CHEN — one-shot setup script for contributors.
#
# Creates a virtual environment, installs CHEN with dev dependencies,
# runs the test suite, and verifies the example scripts work.
#
# Usage:
#   bash scripts/setup.sh          # full setup + verification
#   bash scripts/setup.sh --quick  # skip tests, just install

set -euo pipefail

QUICK=0
if [[ "${1-}" == "--quick" ]]; then
  QUICK=1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "================================================================"
echo "  CHEN setup"
echo "================================================================"
echo "  Project root: $ROOT_DIR"
echo "  Python:       $(python3 --version)"
echo "================================================================"

# 1. Create virtual environment (skip if .venv already exists).
if [[ ! -d ".venv" ]]; then
  echo ">>> Creating virtual environment (.venv)"
  python3 -m venv .venv
fi

# 2. Activate and upgrade pip.
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip --quiet

# 3. Install CHEN with dev dependencies.
echo ">>> Installing CHEN with dev dependencies"
pip install -e ".[dev]" --quiet

# 4. Optionally install the HF backend.
if [[ -n "${CHEN_INSTALL_HF:-}" ]]; then
  echo ">>> Installing HuggingFace backend (CHEN_INSTALL_HF is set)"
  pip install -e ".[hf]" --quiet
fi

# 5. Quick verification.
echo ">>> Verifying installation"
python -c "import chen; print(f'CHEN version: {chen.__version__}')"
python -c "from chen.backends import list_backends; print(f'Backends: {list_backends()}')"

if [[ "$QUICK" -eq 1 ]]; then
  echo ">>> --quick mode: skipping tests"
  echo "Done. Activate the venv with: source .venv/bin/activate"
  exit 0
fi

# 6. Run the test suite (fast tests only).
echo ">>> Running test suite (fast tests only)"
pytest -m "not slow and not gpu and not integration" --no-header -q

# 7. Run an example to verify the pipeline works end-to-end.
echo ">>> Running example: Phase 1"
python examples/run_phase1.py --backend mock > /dev/null
echo "  Phase 1: OK"

echo ">>> Running example: Phase 2"
python examples/run_phase2.py > /dev/null
echo "  Phase 2: OK"

echo ">>> Running example: Phase 3"
python examples/run_phase3.py > /dev/null
echo "  Phase 3: OK"

echo "================================================================"
echo "  Setup complete!"
echo "================================================================"
echo "  Activate the venv with:  source .venv/bin/activate"
echo "  Run tests with:          pytest"
echo "  Run examples with:       python examples/run_phase1.py"
echo "  Read the docs:           cat README.md ARCHITECTURE.md"
echo "================================================================"
