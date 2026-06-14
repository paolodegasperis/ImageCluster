@echo off
setlocal
pushd "%~dp0..\.."
title ImageCluster - Optional multimodal models installer

echo.
echo Optional multimodal embedding models installer.
echo This enables: Qwen3-VL Embedding (2B) and Jina v5 Omni (Small).
echo These models are large and need a GPU; the main app works without them.
echo Note: Jina v5 Omni is licensed CC BY-NC 4.0 (non-commercial use).
echo.

if not exist ".venv" (
  echo .venv was not found. Run Start Windows.bat first.
  pause
  popd
  exit /b 1
)
.venv\Scripts\python.exe -m pip install sentence-transformers "transformers>=4.57" qwen-vl-utils peft
echo.
echo Done. Restart the app (Start Windows.bat) so the new models become selectable.
popd
pause
