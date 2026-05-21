#!/bin/bash
cd "$(dirname "$0")"
echo "Creating ImagePlot-CLIP runtime folders..."
mkdir -p img output/embeddings output/projections output/logs
echo
echo "Runtime folders are ready."
echo "You can now run Start macOS.command."
echo
read -p "Press Enter to close..."
