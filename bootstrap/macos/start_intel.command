#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo ""
echo "ImagePlot-CLIP - macOS Intel launcher"
echo ""

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3.10 or newer was not found."
  echo "Install Python from https://www.python.org/downloads/macos/"
  read -p "Press Enter to exit..."
  exit 1
fi

python3 launcher.py --torch macos
if [ $? -ne 0 ]; then
  echo ""
  echo "ImagePlot-CLIP could not start. Review the message above."
fi
read -p "Press Enter to exit..."
