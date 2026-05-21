@echo off
setlocal
cd /d "%~dp0"
title ImagePlot-CLIP v5.2.2 - Windows 11 CPU

echo.
echo ImagePlot-CLIP v5.2.2 - Windows 11 CPU launcher
echo.
echo The first launch can take several minutes because libraries and models are prepared.
echo This script installs Python libraries inside .venv only.
echo.

set PYTHON_CMD=
where py >nul 2>nul
if not errorlevel 1 set PYTHON_CMD=py -3
if "%PYTHON_CMD%"=="" (
  where python >nul 2>nul
  if not errorlevel 1 set PYTHON_CMD=python
)

if "%PYTHON_CMD%"=="" (
  echo Python 3.10 or newer was not found.
  echo.
  echo Recommended Windows 11 step:
  echo 1. Install Python from https://www.python.org/downloads/windows/
  echo 2. During installation, enable "Add python.exe to PATH".
  echo 3. Run this file again.
  echo.
  echo Optional winget command:
  echo winget install Python.Python.3.12
  echo.
  pause
  exit /b 1
)

%PYTHON_CMD% launcher.py --torch cpu
if errorlevel 1 (
  echo.
  echo ImagePlot-CLIP could not start. Review the message above.
  echo Use this CPU launcher first if CUDA fails.
)
pause
