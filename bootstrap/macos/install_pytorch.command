#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"
echo "Installing ImagePlot-CLIP dependencies for macOS."
echo "This prepares .venv but does not start the web app."
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3.10 or newer was not found."
  echo "Install Python from https://www.python.org/downloads/macos/"
  read -p "Press Enter to exit..."
  exit 1
fi
python3 launcher.py --torch macos --no-start --reinstall
read -p "Done. Press Enter to exit..."
