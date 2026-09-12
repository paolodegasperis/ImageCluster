@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap\windows\start_windows.ps1"
if errorlevel 1 pause
