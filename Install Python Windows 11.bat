@echo off
setlocal
title Install Python for ImagePlot-CLIP

echo.
echo This helper tries to install Python 3 with winget.
echo It is optional. If it fails, install Python manually from python.org.
echo.

where winget >nul 2>nul
if errorlevel 1 (
  echo winget was not found on this Windows installation.
  echo Please install Python 3.10 or newer from:
  echo https://www.python.org/downloads/windows/
  pause
  exit /b 1
)

winget install Python.Python.3.12
echo.
echo After installation, close this window and run Start Windows 11 CPU.bat.
pause
