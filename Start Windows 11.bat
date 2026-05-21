@echo off
setlocal
cd /d "%~dp0"
title ImagePlot-CLIP v5.2.2 - Windows 11

echo.
echo ImagePlot-CLIP v5.2.2 - Windows 11
echo.
echo This launcher prepares the local Python environment automatically.
echo It installs the required libraries inside the project folder in .venv.
echo.
echo Choose PyTorch mode:
echo   1^) CPU  - safest option, works on most Windows 11 computers
echo   2^) CUDA - NVIDIA GPU only, requires compatible NVIDIA driver
echo.
set /p choice="Type 1 or 2 and press Enter [1]: "
if "%choice%"=="" set choice=1

if "%choice%"=="2" (
  call "Start Windows 11 CUDA.bat"
) else (
  call "Start Windows 11 CPU.bat"
)
