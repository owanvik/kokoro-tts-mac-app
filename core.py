#!/usr/bin/env python3
"""Kokoro TTS – core engine, settings, model management (no GUI code)."""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
from urllib.error import URLError, HTTPError
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

try:
    from piper.config import SynthesisConfig
    from piper.voice import PiperVoice
except Exception:
    SynthesisConfig = None
    PiperVoice = None

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
DEFAULT_TTS_ENGINE = "kokoro"
DEFAULT_PIPER_MODEL = "no_NO-talesyntese-medium"

PIPER_MODEL_REGISTRY: dict[str, dict] = {
    "no_NO-talesyntese-medium": {
        "lang": "nb",
        "model": "no_NO-talesyntese-medium.onnx",
        "config": "no_NO-talesyntese-medium.onnx.json",
        "urls": [
            (
                "https://huggingface.co/rhasspy/piper-voices/resolve/main/no/no_NO/talesyntese/medium/no_NO-talesyntese-medium.onnx",
                "https://huggingface.co/rhasspy/piper-voices/resolve/main/no/no_NO/talesyntese/medium/no_NO-talesyntese-medium.onnx.json",
            ),
        ],
    },
    "no_NO-nvcc-medium": {
        "lang": "nb",
        "model": "no_NO-nvcc-medium.onnx",
        "config": "no_NO-nvcc-medium.onnx.json",
        "urls": [
            (
                "https://huggingface.co/rhasspy/piper-voices/resolve/main/no/no_NO/nvcc/medium/no_NO-nvcc-medium.onnx",
                "https://huggingface.co/rhasspy/piper-voices/resolve/main/no/no_NO/nvcc/medium/no_NO-nvcc-medium.onnx.json",
            ),
        ],
    },
}

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

def get_tts_engine() -> str:
    engine = str(load_settings().get("tts_engine", DEFAULT_TTS_ENGINE)).lower()
    return engine if engine in {"kokoro", "piper"} else DEFAULT_TTS_ENGINE

_piper_model_availability_cache: dict[str, bool] = {}

def _url_exists(url: str) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": "KokoroTTS"}, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            return 200 <= getattr(resp, "status", 200) < 400
    except Exception:
        try:
            req_get = urllib.request.Request(url, headers={"User-Agent": "KokoroTTS"})
            with urllib.request.urlopen(req_get, timeout=6) as resp:
                return 200 <= getattr(resp, "status", 200) < 400
        except Exception:
            return False

def _is_piper_model_available(model_id: str) -> bool:
    if model_id in _piper_model_availability_cache:
        return _piper_model_availability_cache[model_id]
    info = PIPER_MODEL_REGISTRY.get(model_id, {})
    urls = info.get("urls") or []
    available = False
    for model_url, config_url in urls:
        if _url_exists(model_url) and _url_exists(config_url):
            available = True
            break
    _piper_model_availability_cache[model_id] = available
    return available

def get_available_piper_models() -> list[str]:
    available = [model_id for model_id in PIPER_MODEL_REGISTRY.keys() if _is_piper_model_available(model_id)]
    if available:
        return available
    return [DEFAULT_PIPER_MODEL] if DEFAULT_PIPER_MODEL in PIPER_MODEL_REGISTRY else list(PIPER_MODEL_REGISTRY.keys())

def get_piper_model() -> str:
    selected = str(load_settings().get("piper_model", DEFAULT_PIPER_MODEL))
    available = get_available_piper_models()
    if selected in available:
        return selected
    if DEFAULT_PIPER_MODEL in available:
        return DEFAULT_PIPER_MODEL
    return available[0] if available else DEFAULT_PIPER_MODEL

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

def _normalize_release_notes(body: str) -> str:
    notes = (body or "").strip()
    if not notes:
        return ""
    max_len = 4000
    if len(notes) > max_len:
        return notes[:max_len].rstrip() + "\n\n…"
    return notes

def _get_latest_release_via_redirect() -> tuple[str, str, str]:
    req = urllib.request.Request(
        LATEST_RELEASE_URL,
        headers={"User-Agent": "KokoroTTS"},
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        final_url = resp.geturl()

    match = re.search(r"/releases/tag/([^/?#]+)", final_url)
    if not match:
        raise RuntimeError("Could not resolve latest release tag")

    tag = urllib.parse.unquote(match.group(1))
    dmg_url = f"https://github.com/{GITHUB_REPO}/releases/download/{tag}/KokoroTTS-mac-arm64.dmg"
    return tag, final_url, dmg_url

def get_recent_releases(limit: int = 20) -> list[dict[str, str]]:
    safe_limit = max(1, min(int(limit), 20))
    try:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{GITHUB_REPO}/releases?per_page={safe_limit}",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "KokoroTTS"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        releases: list[dict[str, str]] = []
        for release in data if isinstance(data, list) else []:
            tag = str(release.get("tag_name") or "").strip()
            if not tag:
                continue
            html_url = str(release.get("html_url") or "").strip() or f"https://github.com/{GITHUB_REPO}/releases/tag/{tag}"
            dmg_url = ""
            for asset in release.get("assets", []) or []:
                name = (asset.get("name") or "").lower()
                if name.endswith(".dmg"):
                    dmg_url = str(asset.get("browser_download_url") or "")
                    break
            if not dmg_url:
                dmg_url = f"https://github.com/{GITHUB_REPO}/releases/download/{tag}/KokoroTTS-mac-arm64.dmg"
            releases.append({
                "tag": tag,
                "url": html_url,
                "dmg_url": dmg_url,
            })
        return releases[:safe_limit]
    except Exception:
        req = urllib.request.Request(
            f"https://github.com/{GITHUB_REPO}/releases",
            headers={"User-Agent": "KokoroTTS"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        tags: list[str] = []
        for raw_tag in re.findall(rf"/{re.escape(GITHUB_REPO)}/releases/tag/([^\"#?]+)", html):
            tag = urllib.parse.unquote(raw_tag).strip()
            if tag and tag not in tags:
                tags.append(tag)
            if len(tags) >= safe_limit:
                break

        return [
            {
                "tag": tag,
                "url": f"https://github.com/{GITHUB_REPO}/releases/tag/{tag}",
                "dmg_url": f"https://github.com/{GITHUB_REPO}/releases/download/{tag}/KokoroTTS-mac-arm64.dmg",
            }
            for tag in tags
        ]

def get_latest_release_details() -> tuple[str, str, str, str]:
    try:
        req = urllib.request.Request(
            LATEST_RELEASE_API,
            headers={"Accept": "application/vnd.github+json", "User-Agent": "KokoroTTS"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        dmg_url = ""
        for asset in data.get("assets", []):
            name = (asset.get("name") or "").lower()
            if name.endswith(".dmg"):
                dmg_url = asset.get("browser_download_url", "")
                break
        release_notes = _normalize_release_notes(str(data.get("body") or ""))
        return (
            data.get("tag_name", ""),
            data.get("html_url", LATEST_RELEASE_URL),
            dmg_url,
            release_notes,
        )
    except Exception:
        tag, html_url, dmg_url = _get_latest_release_via_redirect()
        return tag, html_url, dmg_url, ""

def get_latest_release() -> tuple[str, str, str]:
    tag, url, dmg_url, _ = get_latest_release_details()
    return tag, url, dmg_url

def check_updates_details() -> dict[str, str | bool]:
    try:
        latest_tag, _, _, release_notes = get_latest_release_details()
    except Exception:
        return {
            "update_available": False,
            "message": tr("update_check_failed", version=APP_VERSION),
            "tag": "",
            "release_notes": "",
        }
    if _parse_version(latest_tag) > _parse_version(APP_VERSION):
        return {
            "update_available": True,
            "message": tr("update_available", tag=latest_tag, version=APP_VERSION),
            "tag": latest_tag,
            "release_notes": release_notes,
        }
    return {
        "update_available": False,
        "message": tr("up_to_date", version=APP_VERSION),
        "tag": latest_tag,
        "release_notes": "",
    }

def check_updates_message() -> str:
    details = check_updates_details()
    return str(details["message"])

def _get_app_bundle_path() -> Path | None:
    if not getattr(sys, "frozen", False):
        return None
    exe = Path(sys.executable).resolve()
    for parent in exe.parents:
        if parent.suffix == ".app":
            return parent
    return None

def _install_release_from_dmg(tag: str, dmg_url: str, release_url: str) -> str:
    if not dmg_url:
        return tr("dmg_not_found", tag=tag, url=release_url)

    current_app = _get_app_bundle_path()
    if current_app is None:
        return tr("update_dev_mode")

    tmp_dir = tempfile.mkdtemp(prefix="kokoro-update-")
    dmg_path = Path(tmp_dir) / "update.dmg"
    mount_point = Path(tmp_dir) / "mount"
    mount_point.mkdir()
    try:
        req = urllib.request.Request(dmg_url, headers={"User-Agent": "KokoroTTS"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(dmg_path, "wb") as out_f:
            shutil.copyfileobj(resp, out_f)
    except Exception as e:
        return tr("download_failed", error=str(e))
    try:
        subprocess.run(
            ["hdiutil", "attach", str(dmg_path), "-mountpoint", str(mount_point), "-nobrowse", "-quiet"],
            check=True, capture_output=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return tr("update_mount_failed", error="timeout")
    except subprocess.CalledProcessError as e:
        return tr("update_mount_failed", error=e.stderr.decode(errors="replace").strip())

    new_app = None
    for item in mount_point.iterdir():
        if item.suffix == ".app" and item.is_dir():
            new_app = item
            break
    if new_app is None:
        subprocess.run(["hdiutil", "detach", str(mount_point), "-quiet"], capture_output=True, timeout=20)
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
    return tr("update_restarting", tag=tag)

def auto_update() -> str:
    try:
        latest_tag, latest_url, dmg_url = get_latest_release()
    except Exception as e:
        return tr("update_check_error", error=str(e))
    if _parse_version(latest_tag) <= _parse_version(APP_VERSION):
        return tr("already_latest", version=APP_VERSION)
    return _install_release_from_dmg(latest_tag, dmg_url, latest_url)

def rollback_to_release(tag: str, dmg_url: str, release_url: str) -> str:
    clean_tag = (tag or "").strip()
    if not clean_tag:
        return tr("rollback_invalid_selection")
    return _install_release_from_dmg(clean_tag, dmg_url, release_url)

# ── Model download / engine ──────────────────────────────────────────
def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "KokoroTTS"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as out_f:
                shutil.copyfileobj(resp, out_f)
            if dest.exists() and dest.stat().st_size > 0:
                return
            raise RuntimeError("Downloaded file is empty")
        except (URLError, HTTPError, TimeoutError, OSError, RuntimeError) as exc:
            last_error = exc
            if dest.exists() and dest.stat().st_size == 0:
                dest.unlink(missing_ok=True)
            continue
    raise RuntimeError(f"Download failed for {url}: {last_error}")

def _download_model(version: str) -> tuple[Path, Path]:
    info = MODEL_REGISTRY[version]
    model_path, voices_path = _model_paths(version)
    mirror_errors: list[str] = []
    for model_url, voices_url in info["urls"]:
        try:
            _download(model_url, model_path)
            _download(voices_url, voices_path)
            return model_path, voices_path
        except Exception as exc:
            mirror_errors.append(f"model={model_url} | voices={voices_url} | error={exc}")
            for p in (model_path, voices_path):
                if p.exists() and p.stat().st_size == 0:
                    p.unlink()
            continue
    details = "\n".join(mirror_errors[-3:]) if mirror_errors else "Unknown network error"
    raise RuntimeError(
        f"Could not download model {version} from any mirror.\n"
        f"Please check network/proxy/firewall settings and try again.\n"
        f"Details:\n{details}"
    )

def _piper_dir() -> Path:
    d = MODELS_DIR / "piper"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _piper_paths(voice_id: str) -> tuple[Path, Path]:
    info = PIPER_MODEL_REGISTRY[voice_id]
    base = _piper_dir()
    return base / info["model"], base / info["config"]

def _download_piper_voice(voice_id: str) -> tuple[Path, Path]:
    if voice_id not in PIPER_MODEL_REGISTRY:
        raise RuntimeError(f"Unknown Piper voice: {voice_id}")
    info = PIPER_MODEL_REGISTRY[voice_id]
    model_path, config_path = _piper_paths(voice_id)
    mirror_errors: list[str] = []
    for model_url, config_url in info["urls"]:
        try:
            _download(model_url, model_path)
            _download(config_url, config_path)
            return model_path, config_path
        except Exception as exc:
            mirror_errors.append(f"model={model_url} | config={config_url} | error={exc}")
            for p in (model_path, config_path):
                if p.exists() and p.stat().st_size == 0:
                    p.unlink(missing_ok=True)
            continue
    details = "\n".join(mirror_errors[-3:]) if mirror_errors else "Unknown network error"
    raise RuntimeError(
        f"Could not download Piper voice '{voice_id}' from any mirror.\n"
        f"Please check network/proxy/firewall settings and try again.\n"
        f"Details:\n{details}"
    )

engine: Kokoro | None = None
voices: list[str] = []
_current_model_version: str | None = None
_piper_voice_cache: dict[str, object] = {}

def _piper_runtime_dir() -> Path:
    d = APP_DIR / "runtime" / "piper"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _find_piper_binary(search_dir: Path) -> Path | None:
    candidates = [search_dir / "piper", search_dir / "bin" / "piper"]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    for candidate in search_dir.rglob("piper"):
        if candidate.is_file() and ".app/" not in str(candidate):
            return candidate
    return None

def _platform_piper_runtime_urls() -> list[str]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        asset = "piper_macos_aarch64.tar.gz"
    elif system == "darwin":
        asset = "piper_macos_x64.tar.gz"
    else:
        return []
    return [
        f"https://github.com/rhasspy/piper/releases/latest/download/{asset}",
        f"https://github.com/rhasspy/piper/releases/download/2023.11.14-2/{asset}",
    ]

def _safe_extract_tar_gz(archive_path: Path, dest_dir: Path) -> None:
    dest_root = dest_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            target = (dest_dir / member.name).resolve()
            if not str(target).startswith(str(dest_root)):
                raise RuntimeError("Unsafe archive content")
        tar.extractall(dest_dir)

def _ensure_piper_runtime_binary() -> Path | None:
    runtime_dir = _piper_runtime_dir()
    existing = _find_piper_binary(runtime_dir)
    if existing is not None:
        existing.chmod(existing.stat().st_mode | 0o111)
        return existing

    runtime_urls = _platform_piper_runtime_urls()
    if not runtime_urls:
        return None

    archive_path = runtime_dir / "piper_runtime.tar.gz"
    last_error: Exception | None = None
    for url in runtime_urls:
        try:
            _download(url, archive_path)
            break
        except Exception as exc:
            last_error = exc
            archive_path.unlink(missing_ok=True)
            continue
    if not archive_path.exists():
        if last_error:
            raise RuntimeError(f"Could not download Piper runtime: {last_error}")
        return None

    try:
        _safe_extract_tar_gz(archive_path, runtime_dir)
    finally:
        archive_path.unlink(missing_ok=True)

    binary = _find_piper_binary(runtime_dir)
    if binary is None:
        raise RuntimeError("Piper runtime downloaded, but no piper binary was found")
    binary.chmod(binary.stat().st_mode | 0o111)
    return binary

def _resolve_piper_command() -> list[str] | None:
    piper_bin = shutil.which("piper")
    if piper_bin:
        return [piper_bin]
    try:
        probe = subprocess.run(
            [sys.executable, "-m", "piper", "--help"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=8,
            check=False,
        )
        if probe.returncode == 0:
            return [sys.executable, "-m", "piper"]
    except Exception:
        pass
    runtime_binary = _ensure_piper_runtime_binary()
    if runtime_binary is not None:
        return [str(runtime_binary)]
    return None

def _synthesize_with_piper_python(text: str, voice_id: str, speed: float) -> tuple[np.ndarray, int]:
    if PiperVoice is None or SynthesisConfig is None:
        raise RuntimeError("Piper Python runtime is unavailable")

    model_path, config_path = _download_piper_voice(voice_id)
    cached = _piper_voice_cache.get(voice_id)
    if cached is None:
        cached = PiperVoice.load(model_path=model_path, config_path=config_path)
        _piper_voice_cache[voice_id] = cached

    length_scale = max(0.35, min(2.5, 1.0 / max(0.35, float(speed))))
    syn_config = SynthesisConfig(length_scale=length_scale)
    chunks = list(cached.synthesize(text, syn_config=syn_config))
    if not chunks:
        raise RuntimeError("Piper returned no audio chunks")

    parts: list[np.ndarray] = []
    sample_rate = int(chunks[0].sample_rate)
    for chunk in chunks:
        arr = getattr(chunk, "audio_float_array", None)
        if arr is None:
            continue
        part = np.asarray(arr, dtype=np.float32).reshape(-1)
        if part.size > 0:
            parts.append(part)
    if not parts:
        raise RuntimeError("Piper returned empty audio")
    audio = np.concatenate(parts)
    return np.ascontiguousarray(audio, dtype=np.float32), sample_rate

def _synthesize_with_piper(text: str, voice: str, speed: float) -> tuple[np.ndarray, int]:
    voice_id = voice if voice in PIPER_MODEL_REGISTRY else get_piper_model()
    python_error: str | None = None

    if PiperVoice is not None and SynthesisConfig is not None:
        try:
            return _synthesize_with_piper_python(text=text, voice_id=voice_id, speed=speed)
        except Exception as exc:
            python_error = str(exc)

    piper_cmd = _resolve_piper_command()
    if piper_cmd is None:
        if python_error:
            raise RuntimeError(tr("piper_synthesis_failed", error=python_error))
        raise RuntimeError(tr("piper_not_available"))

    model_path, config_path = _download_piper_voice(voice_id)

    tmp_wav = OUT_DIR / f"_tmp_piper_{datetime.now().strftime('%H%M%S%f')}.wav"
    length_scale = max(0.35, min(2.5, 1.0 / max(0.35, float(speed))))

    cmd = [
        *piper_cmd,
        "--model", str(model_path),
        "--config", str(config_path),
        "--output_file", str(tmp_wav),
        "--length_scale", f"{length_scale:.3f}",
    ]
    try:
        subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
            timeout=120,
        )
        audio, sample_rate = sf.read(tmp_wav, dtype="float32")
        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            audio = audio.mean(axis=1)
        return np.ascontiguousarray(audio, dtype=np.float32), int(sample_rate)
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.decode(errors="replace").strip()
        if python_error:
            err = f"python={python_error} | binary={err}"
        raise RuntimeError(tr("piper_synthesis_failed", error=err or "unknown"))
    finally:
        tmp_wav.unlink(missing_ok=True)

def ensure_engine(version: str | None = None) -> tuple[Kokoro, list[str]]:
    global engine, voices, _current_model_version
    if get_tts_engine() == "piper":
        return None, [get_piper_model()]
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

    engine_name = get_tts_engine()
    styled_text, styled_speed = apply_style(text, style, speed)
    if engine_name == "piper":
        audio, sample_rate = _synthesize_with_piper(styled_text, voice, styled_speed)
    else:
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
    if engine_name == "piper":
        info = f"{info} | engine=Piper"
    return str(out_path), info
