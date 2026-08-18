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

REM Check the repository is clean before pulling
for /f "delims=" %%i in ('git status --porcelain') do set DIRTY=1
if defined DIRTY (
    echo ERROR: Working tree has uncommitted changes.
    echo Commit or stash them before updating.
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

REM Reinstall dependencies and the editable package using the project venv.
if not exist venv\Scripts\python.exe (
    echo ERROR: Python virtual environment not found.
    echo Run setup.bat first.
    pause
    exit /b 1
)
echo [..] Updating dependencies...
venv\Scripts\python.exe -m pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)
echo [OK] Dependencies updated

echo [..] Updating solver package...
venv\Scripts\python.exe -m pip install -e . -q
if %errorlevel% neq 0 (
    echo ERROR: Failed to install the solver package.
    pause
    exit /b 1
)
echo [OK] Solver package updated

REM Show the current version
for /f "delims=" %%i in ('git describe --tags --always 2^>nul') do set VERSION=%%i
echo [OK] Current version: %VERSION%

REM Get package version from git tag
for /f "delims=" %%i in ('git tag --points-at HEAD 2^>nul') do set PKG_VERSION=%%i
if "%PKG_VERSION%"=="" (
    for /f "delims=" %%i in ('git describe --tags --always 2^>nul') do set PKG_VERSION=%%i
)
echo [OK] Package version: %PKG_VERSION%

echo.
echo ========================================
echo   Update complete!
echo ========================================
echo.
echo Run the solver with:
echo     run.bat
echo.
pause
