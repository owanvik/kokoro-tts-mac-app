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

# Create a proper Finder alias (shows Applications icon more reliably than symlink)
osascript <<OSA
set stagePosix to POSIX file "${STAGE_DIR}" as alias
tell application "Finder"
  make new alias file at stagePosix to POSIX file "/Applications"
end tell
OSA

rm -f "$DMG_PATH"
hdiutil create -volname "$VOL_NAME" -srcfolder "$STAGE_DIR" -ov -format UDZO "$DMG_PATH"

echo "DMG klar: $(pwd)/$DMG_PATH"
