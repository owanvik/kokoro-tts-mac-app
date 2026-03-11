#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

source .venv/bin/activate
python -m pip install -q pyinstaller

pyinstaller \
  --noconfirm \
  --windowed \
  --name "KokoroTTS" \
  --add-data "app.py:." \
  launcher.py

echo "Built app: $(pwd)/dist/KokoroTTS.app"
