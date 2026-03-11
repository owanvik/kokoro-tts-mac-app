#!/usr/bin/env python3
"""Kokoro TTS – core engine, settings, model management (no GUI code)."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
if getattr(sys, "frozen", False):
    APP_DIR = Path.home() / "Library" / "Application Support" / "KokoroTTS"
else:
    APP_DIR = BASE_DIR

MODELS_DIR = APP_DIR / "models"
OUT_DIR = APP_DIR / "exports"

for d in [MODELS_DIR, OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Model registry ──────────────────────────────────────────────────
MODEL_REGISTRY: dict[str, dict] = {
    "v1.0": {
        "model": "kokoro-v1.0.onnx",
        "voices": "voices-v1.0.bin",
        "urls": [
            ("https://github.com/owanvik/kokoro-tts-mac-app/releases/download/models-v1.0/kokoro-v1.0.onnx",
             "https://github.com/owanvik/kokoro-tts-mac-app/releases/download/models-v1.0/voices-v1.0.bin"),
            ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
             "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"),
        ],
    },
}
DEFAULT_MODEL_VERSION = "v1.0"

APP_VERSION = f"v{(BASE_DIR / 'VERSION').read_text(encoding='utf-8').strip()}"
APP_DISPLAY_VERSION = f"{APP_VERSION} — Eiriik Edition"
GITHUB_REPO = "owanvik/kokoro-tts-mac-app"
LATEST_RELEASE_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
LATEST_RELEASE_URL = f"https://github.com/{GITHUB_REPO}/releases/latest"
FAVORITES_FILE = APP_DIR / "favorites.json"
SETTINGS_FILE = APP_DIR / "settings.json"
LOCALES_DIR = BASE_DIR / "locales"

# Migrate old flat model files into versioned subfolder
def _model_dir(version: str) -> Path:
    d = MODELS_DIR / version
    d.mkdir(parents=True, exist_ok=True)
    return d

def _model_paths(version: str) -> tuple[Path, Path]:
    info = MODEL_REGISTRY[version]
    d = _model_dir(version)
    return d / info["model"], d / info["voices"]

_old_model = MODELS_DIR / "kokoro-v1.0.onnx"
_old_voices = MODELS_DIR / "voices-v1.0.bin"
if _old_model.exists() and not (_model_dir("v1.0") / "kokoro-v1.0.onnx").exists():
    _old_model.rename(_model_dir("v1.0") / "kokoro-v1.0.onnx")
if _old_voices.exists() and not (_model_dir("v1.0") / "voices-v1.0.bin").exists():
    _old_voices.rename(_model_dir("v1.0") / "voices-v1.0.bin")

# ── Voice language mapping ───────────────────────────────────────────
VOICE_LANG_MAP = {
    "af": "en-us", "am": "en-us",
    "bf": "en-gb", "bm": "en-gb",
    "ef": "es",    "em": "es",
    "ff": "fr-fr",
    "hf": "hi",    "hm": "hi",
    "if": "it",    "im": "it",
    "jf": "ja",    "jm": "ja",
    "pf": "pt-br", "pm": "pt-br",
    "zf": "cmn",   "zm": "cmn",
}

LANGUAGE_CHOICES = [
    ("English (US)", "en-us"),
    ("English (UK)", "en-gb"),
    ("Norsk", "nb"),
    ("Svenska", "sv"),
    ("Dansk", "da"),
    ("Deutsch", "de"),
    ("Français", "fr-fr"),
    ("Español", "es"),
    ("Italiano", "it"),
    ("Português (BR)", "pt-br"),
    ("日本語", "ja"),
    ("中文", "cmn"),
]

STYLES = ["Neutral", "Direct", "Angry", "Calm", "Warm", "Sad", "Cheerful", "Narrator", "Whisper-ish", "Urgent"]

def voices_for_lang(lang: str, all_voices: list[str], show_all: bool = False) -> list[str]:
    if show_all:
        return all_voices
    matching = [v for v in all_voices if VOICE_LANG_MAP.get(v.split("_")[0]) == lang]
    return matching if matching else all_voices

# ── Settings / i18n ──────────────────────────────────────────────────
def load_settings() -> dict:
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}

def save_settings(data: dict) -> None:
    SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def get_ui_language() -> str:
    lang = load_settings().get("ui_language", "nb")
    return lang if lang in {"nb", "en"} else "nb"

def get_model_version() -> str:
    v = load_settings().get("model_version", DEFAULT_MODEL_VERSION)
    return v if v in MODEL_REGISTRY else DEFAULT_MODEL_VERSION

_locale_cache: dict[str, dict[str, str]] = {}

def _load_locale(lang: str) -> dict[str, str]:
    if lang not in _locale_cache:
        path = LOCALES_DIR / f"{lang}.json"
        try:
            _locale_cache[lang] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            _locale_cache[lang] = {}
    return _locale_cache[lang]

def tr(key: str, lang: str | None = None, **kwargs) -> str:
    lang = lang or get_ui_language()
    text = _load_locale(lang).get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, IndexError):
            pass
    return text

# ── Favorites ────────────────────────────────────────────────────────
def load_favorites() -> list[str]:
    try:
        data = json.loads(FAVORITES_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [x for x in data if isinstance(x, str)]
    except Exception:
        pass
    return []

def save_favorites(favorites: list[str]) -> None:
    FAVORITES_FILE.write_text(json.dumps(favorites, ensure_ascii=False, indent=2), encoding="utf-8")

def toggle_favorite(voice: str) -> tuple[list[str], str]:
    favorites = load_favorites()
    if voice in favorites:
        favorites.remove(voice)
        status = tr("fav_removed", voice=voice)
    else:
        favorites.append(voice)
        status = tr("fav_added", voice=voice)
    save_favorites(favorites)
    return favorites, status

# ── Update logic ─────────────────────────────────────────────────────
def _parse_version(tag: str) -> tuple[int, ...]:
    nums = re.findall(r"\d+", tag or "")
    return tuple(int(n) for n in nums[:3]) if nums else (0,)

def get_latest_release() -> tuple[str, str, str]:
    req = urllib.request.Request(
        LATEST_RELEASE_API,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "KokoroTTS"},
    )
    with urllib.request.urlopen(req, timeout=6) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    dmg_url = ""
    for asset in data.get("assets", []):
        name = (asset.get("name") or "").lower()
        if name.endswith(".dmg"):
            dmg_url = asset.get("browser_download_url", "")
            break
    return data.get("tag_name", ""), data.get("html_url", LATEST_RELEASE_URL), dmg_url

def check_updates_message() -> str:
    try:
        latest_tag, _, _ = get_latest_release()
    except Exception:
        return tr("update_check_failed", version=APP_VERSION)
    if _parse_version(latest_tag) > _parse_version(APP_VERSION):
        return tr("update_available", tag=latest_tag, version=APP_VERSION)
    return tr("up_to_date", version=APP_VERSION)

def _get_app_bundle_path() -> Path | None:
    if not getattr(sys, "frozen", False):
        return None
    exe = Path(sys.executable).resolve()
    for parent in exe.parents:
        if parent.suffix == ".app":
            return parent
    return None

def auto_update() -> str:
    try:
        latest_tag, latest_url, dmg_url = get_latest_release()
    except Exception as e:
        return tr("update_check_error", error=str(e))
    if _parse_version(latest_tag) <= _parse_version(APP_VERSION):
        return tr("already_latest", version=APP_VERSION)
    if not dmg_url:
        return tr("dmg_not_found", tag=latest_tag, url=latest_url)
    current_app = _get_app_bundle_path()
    if current_app is None:
        import webbrowser
        webbrowser.open(latest_url)
        return tr("update_dev_mode", url=latest_url)

    tmp_dir = tempfile.mkdtemp(prefix="kokoro-update-")
    dmg_path = Path(tmp_dir) / "update.dmg"
    mount_point = Path(tmp_dir) / "mount"
    mount_point.mkdir()
    try:
        urllib.request.urlretrieve(dmg_url, dmg_path)
    except Exception as e:
        return tr("download_failed", error=str(e))
    try:
        subprocess.run(
            ["hdiutil", "attach", str(dmg_path), "-mountpoint", str(mount_point), "-nobrowse", "-quiet"],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        return tr("update_mount_failed", error=e.stderr.decode(errors="replace").strip())

    new_app = None
    for item in mount_point.iterdir():
        if item.suffix == ".app" and item.is_dir():
            new_app = item
            break
    if new_app is None:
        subprocess.run(["hdiutil", "detach", str(mount_point), "-quiet"], capture_output=True)
        return tr("update_no_app_in_dmg")

    install_dest = current_app.parent
    app_name = current_app.name
    updater_script = Path(tmp_dir) / "updater.sh"
    updater_script.write_text(f"""#!/bin/bash
PID={os.getpid()}
for i in {{1..30}}; do
    kill -0 "$PID" 2>/dev/null || break
    sleep 0.5
done
rm -rf "{install_dest / app_name}"
cp -pR "{new_app}" "{install_dest / app_name}"
hdiutil detach "{mount_point}" -quiet 2>/dev/null
rm -rf "{tmp_dir}"
open "{install_dest / app_name}"
""", encoding="utf-8")
    updater_script.chmod(0o755)
    subprocess.Popen(
        ["/bin/bash", str(updater_script)],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    import threading, time
    def _delayed_exit():
        time.sleep(1.5)
        os._exit(0)
    threading.Thread(target=_delayed_exit, daemon=True).start()
    return tr("update_restarting", tag=latest_tag)

# ── Model download / engine ──────────────────────────────────────────
def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    urllib.request.urlretrieve(url, dest)

def _download_model(version: str) -> tuple[Path, Path]:
    info = MODEL_REGISTRY[version]
    model_path, voices_path = _model_paths(version)
    for model_url, voices_url in info["urls"]:
        try:
            _download(model_url, model_path)
            _download(voices_url, voices_path)
            return model_path, voices_path
        except Exception:
            for p in (model_path, voices_path):
                if p.exists() and p.stat().st_size == 0:
                    p.unlink()
            continue
    raise RuntimeError(f"Could not download model {version} from any mirror")

engine: Kokoro | None = None
voices: list[str] = []
_current_model_version: str | None = None

def ensure_engine(version: str | None = None) -> tuple[Kokoro, list[str]]:
    global engine, voices, _current_model_version
    version = version or get_model_version()
    if engine is None or _current_model_version != version:
        model_path, voices_path = _download_model(version)
        engine = Kokoro(model_path=str(model_path), voices_path=str(voices_path))
        voices = engine.get_voices()
        _current_model_version = version
    return engine, voices

# ── Style / synthesis ────────────────────────────────────────────────
def apply_style(text: str, style: str, speed: float) -> tuple[str, float]:
    style_speed = {
        "Neutral": 1.00, "Direct": 1.07, "Angry": 1.15, "Calm": 0.92,
        "Warm": 0.96, "Sad": 0.90, "Cheerful": 1.10, "Narrator": 0.94,
        "Whisper-ish": 0.88, "Urgent": 1.20,
    }
    target = style_speed.get(style, 1.00)
    merged_speed = max(0.5, min(2.0, (float(speed) + target) / 2.0))
    styled_text = text.strip()
    if style == "Direct":
        styled_text = styled_text.replace("...", ".")
    elif style == "Angry":
        if not styled_text.endswith(("!", ".", "?")):
            styled_text += "!"
    elif style in ("Calm", "Whisper-ish"):
        styled_text = styled_text.replace("!", ".")
    elif style == "Urgent":
        if not styled_text.endswith("!"):
            styled_text += "!"
    return styled_text, merged_speed

def _db_to_gain(db: float) -> float:
    return float(10 ** (db / 20.0))

def apply_preset(preset: str) -> tuple[str, float, float]:
    presets = {
        "neutral": ("Neutral", 1.00, 0.0),
        "alert": ("Urgent", 1.18, 2.0),
        "narration": ("Narrator", 0.92, -1.0),
        "direct": ("Direct", 1.08, 1.0),
    }
    return presets.get(preset, presets["neutral"])

def synthesize(text: str, voice: str, speed: float, lang: str,
               style: str, gain_db: float, output_format: str) -> tuple[str, str]:
    """Run TTS. Returns (output_path, info_message)."""
    if not text or not text.strip():
        raise ValueError(tr("error_empty_text"))

    styled_text, styled_speed = apply_style(text, style, speed)
    tts, _ = ensure_engine()
    audio, sample_rate = tts.create(
        text=styled_text, voice=voice, speed=styled_speed, lang=lang,
    )
    audio = np.clip(audio * _db_to_gain(float(gain_db)), -1.0, 1.0)

    voice_safe = re.sub(r'[^\w\s-]', '', voice).strip()
    voice_safe = re.sub(r'\s+', '_', voice_safe) or "voice"
    snippet = text.strip()[:40].rstrip()
    snippet_safe = re.sub(r'[^\w\s-]', '', snippet).strip()
    snippet_safe = re.sub(r'\s+', '_', snippet_safe) or "clip"
    timestamp = datetime.now().strftime('%H%M%S')
    ext = "mp3" if output_format.lower() == "mp3" else "wav"
    filename = f"{voice_safe}_{snippet_safe}_{timestamp}.{ext}"
    out_path = OUT_DIR / filename
    sf.write(out_path, audio, sample_rate)

    info = tr("synth_done", voice=voice, lang=lang, style=style,
              speed=f"{styled_speed:.2f}", gain=f"{gain_db:+.1f}", format=ext.upper())
    return str(out_path), info
