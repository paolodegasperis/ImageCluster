#!/bin/bash
# Convenience shortcut: installs the optional multimodal models
# (Qwen3-VL Embedding 2B, Jina v5 Omni Small) into the local .venv.
# Run this after the first normal start, then restart the app.
exec "$(cd "$(dirname "$0")" && pwd)/bootstrap/macos/install_optional_multimodal.command"
