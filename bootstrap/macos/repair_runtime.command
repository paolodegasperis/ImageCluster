#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"
echo "Creating ImagePlot-CLIP runtime folders..."
mkdir -p img output/embeddings output/projections output/logs
echo
echo "Runtime folders are ready."
echo "You can now run Start macOS.command."
echo
read -p "Press Enter to close..."
