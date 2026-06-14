#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"
echo ""
echo "Optional ImageBind installer."
echo "ImageBind is large and may not work on every setup."
echo "The main app works without this optional component."
echo ""
if [ ! -d ".venv" ]; then
  echo ".venv was not found. Run Start macOS.command first."
  read -p "Press Enter to exit..."
  exit 1
fi
.venv/bin/python -m pip install imagebind-packaged soundfile
read -p "Done. Press Enter to exit..."
