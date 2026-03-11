#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

source .venv/bin/activate
python -m pip install -q pyinstaller

pyinstaller \
  --noconfirm \
  --windowed \
  --name "KokoroTTS" \
  --icon "icons/kokorotts.icns" \
  --add-data "app.py:." \
  --collect-data safehttpx \
  --collect-data groovy \
  --collect-all kokoro_onnx \
  --collect-all phonemizer \
  --collect-all segments \
  --collect-all csvw \
  --collect-all language_tags \
  --collect-all espeakng_loader \
  --collect-all gradio \
  launcher.py

echo "Built app: $(pwd)/dist/KokoroTTS.app"
