@echo off
title CFD Solver
echo ========================================
echo   CFD Solver
echo ========================================
echo.

REM Create venv if missing
if not exist venv\ (
    echo [..] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment.
        echo Make sure Python is installed and added to PATH.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

call venv\Scripts\activate.bat

REM Install dependencies if numpy is not available
python -c "import numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo [..] Installing dependencies...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo.
        echo ERROR: Failed to install dependencies.
        echo Try running: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed
)

python run_interactive.py
pause
