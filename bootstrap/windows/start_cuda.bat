@echo off
setlocal
pushd "%~dp0..\.."
title ImagePlot-CLIP - Windows CUDA launcher

echo.
echo ImagePlot-CLIP - Windows CUDA launcher
echo.
echo Use this only with a compatible NVIDIA GPU and driver.
echo If you are unsure, close this window and use Start Windows.bat.
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
> "output\.startup_variant" echo cuda

%PYTHON_CMD% launcher.py --torch cuda
if errorlevel 1 (
  echo.
  echo CUDA startup failed or ImagePlot-CLIP could not start.
  echo Try Start Windows.bat as the safest alternative.
)
popd
pause
