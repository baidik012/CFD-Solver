#!/usr/bin/env bash
set -e

echo "========================================"
echo "  CFD Solver"
echo "========================================"
echo

if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found."
    echo "Run ./setup.sh first."
    exit 1
fi

. venv/bin/activate
python3 run_interactive.py
