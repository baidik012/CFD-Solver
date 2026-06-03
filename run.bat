@echo off
title CFD Solver
echo ========================================
echo   CFD Solver
echo ========================================
echo.

if not exist venv\ (
    echo ERROR: Virtual environment not found.
    echo Run setup.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python run_interactive.py
pause
