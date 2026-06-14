@echo off
setlocal
pushd "%~dp0..\.."
title ImagePlot-CLIP - Windows CPU launcher

echo.
echo ImagePlot-CLIP - Windows CPU launcher
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
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_windows.ps1"
  echo.
  echo Please run this launcher again after Python has been installed.
  popd
  pause
  exit /b 1
)

if not exist "output" mkdir "output"
> "output\.startup_variant" echo cpu

%PYTHON_CMD% launcher.py --torch cpu
if errorlevel 1 (
  echo.
  echo ImagePlot-CLIP could not start. Review the message above.
  echo Use this CPU launcher first if CUDA fails.
)
popd
pause
