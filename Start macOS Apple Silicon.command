#!/bin/bash
cd "$(dirname "$0")"

echo ""
echo "ImagePlot-CLIP v5.2.2 - macOS launcher"
echo ""
echo "The first launch can take several minutes because libraries and models are prepared."
echo "This script installs Python libraries inside .venv only."
echo ""

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3.10 or newer was not found."
  echo ""
  echo "Install Python from https://www.python.org/downloads/macos/"
  echo "or, if you use Homebrew, run:"
  echo "  brew install python"
  echo ""
  read -p "Press Enter to exit..."
  exit 1
fi

python3 launcher.py --torch macos
if [ $? -ne 0 ]; then
  echo ""
  echo "ImagePlot-CLIP could not start. Review the message above."
fi
read -p "Press Enter to exit..."
