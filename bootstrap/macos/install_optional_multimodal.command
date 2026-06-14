#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"
echo ""
echo "Optional multimodal embedding models installer."
echo "This enables: Qwen3-VL Embedding (2B) and Jina v5 Omni (Small)."
echo "These models are large and need a GPU; the main app works without them."
echo "Note: Jina v5 Omni is licensed CC BY-NC 4.0 (non-commercial use)."
echo ""
if [ ! -d ".venv" ]; then
  echo ".venv was not found. Run Start macOS.command first."
  read -p "Press Enter to exit..."
  exit 1
fi
.venv/bin/python -m pip install sentence-transformers "transformers>=4.57" qwen-vl-utils peft
echo ""
echo "Done. Restart the app (Start macOS.command) so the new models become selectable."
read -p "Press Enter to exit..."
