#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_CMD=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
fi

if [ -z "$PYTHON_CMD" ]; then
  choice=$(osascript <<'APPLESCRIPT' 2>/dev/null
display dialog "Python 3.10 or newer is required to start ImagePlot-CLIP.

Choose Open Official Site to download Python, Use Homebrew if you already use it, or Cancel." buttons {"Open Official Site", "Use Homebrew", "Cancel"} default button "Open Official Site" with icon caution
APPLESCRIPT
)
  case "$choice" in
    *"Open Official Site"*)
      open "https://www.python.org/downloads/macos/"
      ;;
    *"Use Homebrew"*)
      if command -v brew >/dev/null 2>&1; then
        brew install python
      else
        open "https://www.python.org/downloads/macos/"
      fi
      ;;
  esac
  exit 1
fi

"$PYTHON_CMD" launcher.py --torch macos
