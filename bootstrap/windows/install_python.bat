@echo off
setlocal
title Install Python for ImagePlot-CLIP

echo.
echo This helper tries to install Python 3 with winget.
echo It is optional. If it fails, install Python manually from python.org.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_windows.ps1"
echo.
echo After installation, close this window and run the Windows launcher again.
pause
