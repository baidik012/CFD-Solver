@echo off
title CFD Solver Update
echo ========================================
echo   CFD Solver - Update
echo ========================================
echo.

REM Must be run from the repo root
if not exist pyproject.toml (
    echo ERROR: Run this script from the CFD-Solver directory.
    pause
    exit /b 1
)

REM Check git is available
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Git not found. Cannot pull updates.
    echo Install from https://git-scm.com/download/win
    pause
    exit /b 1
)

echo [..] Fetching latest changes from GitHub...
git pull origin main
if %errorlevel% neq 0 (
    echo ERROR: Failed to pull updates. Check your internet connection.
    pause
    exit /b 1
)
echo [OK] Code updated

REM Re-install the package in case dependencies or entry points changed
if exist venv\ (
    echo [..] Updating dependencies...
    call venv\Scripts\activate.bat
    pip install -r requirements.txt -q
    pip install -e . -q
    echo [OK] Dependencies up to date
) else (
    echo [SKIP] No virtual environment found. Run setup.bat first.
)

echo.
echo ========================================
echo   Update complete!
echo ========================================
echo.
echo Run the solver with:
echo     run.bat
echo.
pause
