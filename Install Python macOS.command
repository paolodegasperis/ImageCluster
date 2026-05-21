#!/bin/bash
echo ""
echo "This helper installs Python with Homebrew if Homebrew is available."
echo "It is optional. If it fails, install Python from https://www.python.org/downloads/macos/"
echo ""

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew was not found."
  echo "Install Python from https://www.python.org/downloads/macos/"
  read -p "Press Enter to exit..."
  exit 1
fi

brew install python
echo ""
echo "After installation, run Start macOS.command."
read -p "Press Enter to exit..."
