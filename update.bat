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

REM Show the current version
for /f "delims=" %%i in ('git describe --tags --always 2^>nul') do set VERSION=%%i
echo [OK] Current version: %VERSION%

REM Re-install the package in case dependencies or entry points changed
if exist venv\ (
    echo [..] Updating dependencies...
    venv\Scripts\python.exe -m pip install -r requirements.txt -q
    venv\Scripts\python.exe -m pip install -e . -q
    echo [OK] Dependencies up to date
    for /f "delims=" %%i in ('venv\Scripts\python.exe -c "import importlib.metadata; print(importlib.metadata.version('cfd-solver'))" 2^>nul') do set PKG_VERSION=%%i
        echo [OK] Package version: %PKG_VERSION%
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
