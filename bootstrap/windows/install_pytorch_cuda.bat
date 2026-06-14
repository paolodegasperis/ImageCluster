@echo off
setlocal
pushd "%~dp0..\.."
title ImagePlot-CLIP - Install Windows CUDA dependencies
echo Installing ImagePlot-CLIP dependencies for Windows CUDA mode.
echo This prepares .venv but does not start the web app.
where py >nul 2>nul
if not errorlevel 1 (
  py -3 launcher.py --torch cuda --no-start --reinstall
) else (
  python launcher.py --torch cuda --no-start --reinstall
)
popd
pause
