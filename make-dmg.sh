#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

APP_PATH="dist/KokoroTTS.app"
DMG_PATH="dist/KokoroTTS-mac-arm64.dmg"
VOL_NAME="KokoroTTS"
STAGE_DIR="dist/dmg-stage"
TMP_DMG="dist/KokoroTTS-tmp.dmg"
MOUNT_DIR="/Volumes/${VOL_NAME}"
BG_SRC="assets/dmg-background.png"

if [[ ! -d "$APP_PATH" ]]; then
  echo "App mangler. Bygger først..."
  ./build-mac-app.sh
fi

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/.background"
cp -R "$APP_PATH" "$STAGE_DIR/"
cp "$BG_SRC" "$STAGE_DIR/.background/background.png"

# Proper Finder alias so Applications icon appears reliably
osascript <<OSA
set stagePosix to POSIX file "${STAGE_DIR}" as alias
tell application "Finder"
  make new alias file at stagePosix to POSIX file "/Applications"
end tell
OSA

rm -f "$TMP_DMG" "$DMG_PATH"

# 1) Create writable DMG (size based on app + headroom)
APP_MB=$(du -sm "$APP_PATH" | awk '{print $1}')
DMG_MB=$((APP_MB + 250))
hdiutil create -size "${DMG_MB}m" -fs HFS+ -volname "$VOL_NAME" -ov "$TMP_DMG" >/dev/null

# 2) Mount and copy payload
ATTACH_OUT=$(hdiutil attach "$TMP_DMG" -mountpoint "$MOUNT_DIR" -nobrowse)
DEVICE=$(echo "$ATTACH_OUT" | awk '/Apple_HFS/ {print $1; exit}')
cp -R "$STAGE_DIR"/* "$MOUNT_DIR"/
cp -R "$STAGE_DIR"/.background "$MOUNT_DIR"/

# 3) Finder cosmetics (background + bigger icons + layout)
osascript <<OSA
 set bgFile to POSIX file "${MOUNT_DIR}/.background/background.png"
 tell application "Finder"
   tell disk "${VOL_NAME}"
     open
     set current view of container window to icon view
     set toolbar visible of container window to false
     set statusbar visible of container window to false
     set bounds of container window to {120, 120, 980, 660}

     tell icon view options of container window
       set arrangement to not arranged
       set icon size to 144
       set text size to 14
       set background picture to bgFile
     end tell

     set position of item "KokoroTTS.app" of container window to {220, 300}
     try
       set position of item "Applications" of container window to {660, 300}
     end try

     close
     open
     update without registering applications
     delay 1
   end tell
 end tell
OSA

# 4) Unmount and convert to compressed read-only DMG
hdiutil detach "$DEVICE" -quiet || hdiutil detach "$DEVICE" -force -quiet
hdiutil convert "$TMP_DMG" -format UDZO -imagekey zlib-level=9 -o "$DMG_PATH" >/dev/null
rm -f "$TMP_DMG"

echo "DMG klar: $(pwd)/$DMG_PATH"
