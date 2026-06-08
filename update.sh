#!/usr/bin/env bash
set -e

echo "========================================"
echo "  CFD Solver - Update"
echo "========================================"
echo

# Must be run from the repo root
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: Run this script from the CFD-Solver directory."
    exit 1
fi

# Check git is available
if ! command -v git &> /dev/null; then
    echo "ERROR: Git not found. Cannot pull updates."
    echo "Install it with:"
    echo "  macOS:  brew install git"
    echo "  Ubuntu: sudo apt install git"
    exit 1
fi

# Check we're inside a git repo
if ! git rev-parse --is-inside-work-tree &> /dev/null; then
    echo "ERROR: Not inside a git repository."
    echo "Make sure you cloned the repo with 'git clone'."
    exit 1
fi

echo "[..] Fetching latest changes from GitHub..."
git pull origin main
echo "[OK] Code updated"

# Show the current version
VERSION=$(git describe --tags --always 2>/dev/null || echo "unknown")
echo "[OK] Current version: ${VERSION}"

# Re-install the package in case dependencies or entry points changed
if [ -d "venv" ]; then
    echo "[..] Updating dependencies..."
    . venv/bin/activate
    pip install -r requirements.txt -q
    pip install -e . -q
    echo "[OK] Dependencies up to date"
    PKG_VERSION=$(python -c "from cfd_solver import __version__; print(__version__)" 2>/dev/null || echo "unknown")
    echo "[OK] Package version: ${PKG_VERSION}"
else
    echo "[SKIP] No virtual environment found (run ./setup.sh first)"
fi

echo
echo "========================================"
echo "  Update complete!"
echo "========================================"
echo
echo "Run the solver with:"
echo "    ./run.sh"
echo
