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

REM Use venv Python directly — no activation needed
set VENV_PYTHON=venv\Scripts\python.exe

REM Install dependencies if numpy is not available
%VENV_PYTHON% -c "import numpy, scipy, matplotlib, yaml, cfd_solver" >nul 2>&1
if %errorlevel% neq 0 (
    echo [..] Installing dependencies...
    %VENV_PYTHON% -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo.
        echo ERROR: Failed to install dependencies.
        echo Try running manually: venv\Scripts\pip.exe install -r requirements.txt
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed
)

REM Ensure the solver package is installed
%VENV_PYTHON% -c "import cfd_solver" >nul 2>&1
if %errorlevel% neq 0 (
    echo [..] Installing solver package...
    %VENV_PYTHON% -m pip install -e . -q
    if %errorlevel% neq 0 (
        echo WARNING: Failed to install solver package. Running from source.
    ) else (
        echo [OK] Solver package installed
    )
)

%VENV_PYTHON% run_interactive.py
pause
