#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

APP_PATH="dist/KokoroTTS.app"
DMG_PATH="dist/KokoroTTS-mac-arm64.dmg"
VOL_NAME="KokoroTTS"
STAGE_DIR="dist/dmg-stage"

if [[ ! -d "$APP_PATH" ]]; then
  echo "App mangler. Bygger først..."
  ./build-mac-app.sh
fi

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"
cp -R "$APP_PATH" "$STAGE_DIR/"
ln -s /Applications "$STAGE_DIR/Applications"

rm -f "$DMG_PATH"
hdiutil create -volname "$VOL_NAME" -srcfolder "$STAGE_DIR" -ov -format UDZO "$DMG_PATH"

echo "DMG klar: $(pwd)/$DMG_PATH"
