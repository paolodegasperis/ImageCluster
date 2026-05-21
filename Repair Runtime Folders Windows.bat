@echo off
setlocal
cd /d "%~dp0"
echo Creating ImagePlot-CLIP runtime folders...
if not exist "img" mkdir "img"
if not exist "output" mkdir "output"
if not exist "output\embeddings" mkdir "output\embeddings"
if not exist "output\projections" mkdir "output\projections"
if not exist "output\logs" mkdir "output\logs"
echo.
echo Runtime folders are ready.
echo You can now run Start Windows 11 CPU.bat.
echo.
pause
