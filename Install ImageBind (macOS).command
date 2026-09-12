#!/bin/bash
# Convenience shortcut: installs the optional ImageBind (Huge) model into the
# local .venv. Run this after the first normal start, then restart the app.
exec "$(cd "$(dirname "$0")" && pwd)/bootstrap/macos/install_optional_imagebind.command"
