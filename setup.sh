#!/usr/bin/env bash
set -e

echo "========================================"
echo "  CFD Solver - Setup"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found."
    echo "Install it with:"
    echo "  macOS:  brew install python"
    echo "  Ubuntu: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi
echo "[OK] Python found"
python3 --version

# Check if Git is installed
if ! command -v git &> /dev/null; then
    echo "WARNING: Git not found."
    echo "Install it with:"
    echo "  macOS:  brew install git"
    echo "  Ubuntu: sudo apt install git"
    echo "You can still run the solver, but won't be able to pull updates."
fi

# Create virtual environment
if [ -d "venv" ]; then
    echo "[SKIP] Virtual environment already exists"
else
    echo "[..] Creating virtual environment..."
    python3 -m venv venv
    echo "[OK] Virtual environment created"
fi

# Activate and install
echo "[..] Installing dependencies..."
. venv/bin/activate
pip install -r requirements.txt
pip install .
echo "[OK] Dependencies installed"

echo
echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo
echo "To run the solver:"
echo "    ./run.sh"
echo
echo "Or manually:"
echo "    source venv/bin/activate"
echo "    python3 run_interactive.py"
echo
