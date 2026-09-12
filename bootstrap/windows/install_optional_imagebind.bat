@echo off
setlocal
pushd "%~dp0..\.."
title ImagePlot-CLIP - Optional ImageBind installer

echo.
echo Optional ImageBind installer.
echo ImageBind is large and may not work on every Windows/macOS setup.
echo The main app works without this optional component.
echo.

if not exist ".venv" (
  echo .venv was not found. Run Start Windows.bat first.
  pause
  popd
  exit /b 1
)
.venv\Scripts\python.exe -m pip install imagebind-packaged soundfile
popd
pause
