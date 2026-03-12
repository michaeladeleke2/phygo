#!/bin/bash
set -e

echo "🚀 Building PhyGO for macOS..."

# Activate virtual environment
source ../venv/bin/activate

# Install PyInstaller if not already installed
pip install pyinstaller

# Clean previous builds
rm -rf build dist

# Build the app
pyinstaller radar_gui.spec

echo "✅ Build complete!"
echo "📦 App bundle: dist/PhyGO.app"
echo ""
echo "To test: open dist/PhyGO.app"
echo "To create DMG for distribution:"
echo "  hdiutil create -volname PhyGO -srcfolder dist/PhyGO.app -ov -format UDZO PhyGO-macOS.dmg"