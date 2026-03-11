#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

source .venv/bin/activate
python -m pip install -q pyinstaller

VERSION=$(cat VERSION | tr -d '\n')

pyinstaller \
  --noconfirm \
  --windowed \
  --name "KokoroTTS" \
  --icon "icons/kokorotts.icns" \
  --add-data "app.py:." \
  --add-data "VERSION:." \
  --add-data "kokorotts.png:." \
  --add-data "icons/kokorotts.png:icons" \
  --add-data "icons/kokorotts-menubar.png:icons" \
  --add-data "locales:locales" \
  --collect-data safehttpx \
  --collect-data groovy \
  --collect-all kokoro_onnx \
  --collect-all phonemizer \
  --collect-all segments \
  --collect-all csvw \
  --collect-all language_tags \
  --collect-all espeakng_loader \
  --collect-all gradio \
  --collect-all rumps \
  launcher.py

PLIST="$(pwd)/dist/KokoroTTS.app/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString ${VERSION}" "$PLIST" || \
  /usr/libexec/PlistBuddy -c "Add :CFBundleShortVersionString string ${VERSION}" "$PLIST"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion ${VERSION}" "$PLIST" || \
  /usr/libexec/PlistBuddy -c "Add :CFBundleVersion string ${VERSION}" "$PLIST"

# Re-sign after plist edits to avoid macOS "app is damaged" warnings
codesign --force --deep --sign - "$(pwd)/dist/KokoroTTS.app"

echo "Built app: $(pwd)/dist/KokoroTTS.app (version ${VERSION})"
