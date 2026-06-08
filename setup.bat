@echo off
title CFD Solver Setup
echo ========================================
echo   CFD Solver - Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found.
    echo Download it from https://python.org/downloads
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)
echo [OK] Python found
python --version

REM Check if Git is installed
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Git not found. Install from https://git-scm.com/download/win
    echo You can still run the solver, but won't be able to pull updates.
)

REM Create virtual environment
if exist venv\ (
    echo [SKIP] Virtual environment already exists
) else (
    echo [..] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

REM Install dependencies and the solver package using venv Python directly
echo [..] Installing dependencies...
venv\Scripts\python.exe -m pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo.
    echo WARNING: Some packages failed to install.
    echo Try running: venv\Scripts\pip.exe install -r requirements.txt
)
echo [OK] Dependencies installed

echo [..] Installing solver package...
venv\Scripts\python.exe -m pip install -e . -q
if %errorlevel% neq 0 (
    echo WARNING: Failed to install solver package in editable mode.
    echo The solver will still run from source.
)
echo [OK] Solver package installed

echo.
echo ========================================
echo   Setup complete!
echo ========================================
echo.
echo To run the solver:
echo     run.bat
echo.
echo Or manually:
echo     venv\Scripts\activate
echo     python examples\lid_cavity.py
echo.
pause
