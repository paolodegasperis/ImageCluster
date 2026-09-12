@echo off
rem Convenience shortcut: installs the optional multimodal models
rem (Qwen3-VL Embedding 2B, Jina v5 Omni Small) into the local .venv.
rem Run this after the first normal start, then restart the app.
call "%~dp0bootstrap\windows\install_optional_multimodal.bat"
