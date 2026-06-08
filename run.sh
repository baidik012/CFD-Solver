#!/usr/bin/env bash
set -e

echo "========================================"
echo "  CFD Solver"
echo "========================================"
echo

# Create venv if missing
if [ ! -d "venv" ]; then
    echo "[..] Creating virtual environment..."
    python3 -m venv venv
    echo "[OK] Virtual environment created"
fi

. venv/bin/activate

# Install dependencies if numpy is not available
if ! python3 -c "import numpy" 2>/dev/null; then
    echo "[..] Installing dependencies..."
    pip install -r requirements.txt -q
    echo "[OK] Dependencies installed"
fi

# Ensure the solver package is installed
if ! python3 -c "import cfd_solver" 2>/dev/null; then
    echo "[..] Installing solver package..."
    pip install -e . -q
    echo "[OK] Solver package installed"
fi

python3 run_interactive.py
