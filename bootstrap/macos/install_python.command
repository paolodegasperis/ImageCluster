#!/bin/bash
echo ""
echo "This helper installs Python with Homebrew if Homebrew is available."
echo "It is optional. If it fails, install Python from https://www.python.org/downloads/macos/"
echo ""

if ! command -v brew >/dev/null 2>&1; then
  osascript -e 'display dialog "Homebrew was not found. Open the official Python download page?" buttons {"Open Site", "Cancel"} default button "Open Site" with icon caution' 2>/dev/null
  open "https://www.python.org/downloads/macos/"
  read -p "Press Enter to exit..."
  exit 1
fi

choice=$(osascript <<'APPLESCRIPT' 2>/dev/null
display dialog "Homebrew is available. Do you want to install Python now with Homebrew?" buttons {"Install", "Open Official Site", "Cancel"} default button "Install" with icon caution
APPLESCRIPT
)
case "$choice" in
  *"Install"*)
    brew install python
    ;;
  *"Open Official Site"*)
    open "https://www.python.org/downloads/macos/"
    ;;
  *)
    ;;
esac
echo ""
echo "After installation, run Start macOS.command."
read -p "Press Enter to exit..."
