@echo off
setlocal
pushd "%~dp0..\.."
title ImagePlot-CLIP - Install Windows CPU dependencies
echo Installing ImagePlot-CLIP dependencies for Windows CPU mode.
echo This prepares .venv but does not start the web app.
where py >nul 2>nul
if not errorlevel 1 (
  py -3 launcher.py --torch cpu --no-start --reinstall
) else (
  python launcher.py --torch cpu --no-start --reinstall
)
popd
pause
