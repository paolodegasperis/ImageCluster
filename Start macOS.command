#!/bin/bash
set -e
cd "$(dirname "$0")"
exec bash "./bootstrap/macos/start_macos.command"
