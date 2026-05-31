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

REM Activate and install
echo [..] Installing dependencies...
call venv\Scripts\activate.bat
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo WARNING: Some packages failed to install.
    echo Try running: pip install -r requirements.txt
)
echo [OK] Dependencies installed

echo.
echo ========================================
echo   Setup complete!
echo ========================================
echo.
echo To run the solver:
echo     python examples\staggered_cavity.py
echo.
echo To activate the environment next time:
echo     venv\Scripts\activate
echo.
pause
