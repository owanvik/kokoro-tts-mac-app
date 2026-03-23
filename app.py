#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
import shutil
import signal
import ssl
import subprocess
import re
import sys
import tempfile
import urllib.request
from datetime import datetime
from pathlib import Path

try:
    import certifi
    _CERTIFI_CA = certifi.where()
except Exception:
    _CERTIFI_CA = None

def _ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=_CERTIFI_CA)

import gradio as gr
import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

BASE_DIR = Path(__file__).resolve().parent
if getattr(sys, "frozen", False):
    APP_DIR = Path.home() / "Library" / "Application Support" / "KokoroTTS"
else:
    APP_DIR = BASE_DIR

MODELS_DIR = APP_DIR / "models"
OUT_DIR = APP_DIR / "exports"
TMP_DIR = APP_DIR / "tmp"
CERTS_DIR = APP_DIR / "certs"

# ── Model registry ──────────────────────────────────────────────────
# Each version maps to mirror URL (our repo) + fallback (upstream).
# Files stored per-version: models/v1.0/kokoro-v1.0.onnx, etc.
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

def _model_dir(version: str) -> Path:
    d = MODELS_DIR / version
    d.mkdir(parents=True, exist_ok=True)
    return d

def _model_paths(version: str) -> tuple[Path, Path]:
    info = MODEL_REGISTRY[version]
    d = _model_dir(version)
    return d / info["model"], d / info["voices"]

def get_model_version() -> str:
    v = load_settings().get("model_version", DEFAULT_MODEL_VERSION)
    return v if v in MODEL_REGISTRY else DEFAULT_MODEL_VERSION

APP_VERSION = f"v{(BASE_DIR / 'VERSION').read_text(encoding='utf-8').strip()}"
GITHUB_REPO = "owanvik/kokoro-tts-mac-app"
LATEST_RELEASE_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
LATEST_RELEASE_URL = f"https://github.com/{GITHUB_REPO}/releases/latest"
FAVORITES_FILE = APP_DIR / "favorites.json"
SETTINGS_FILE = APP_DIR / "settings.json"
LOCALES_DIR = BASE_DIR / "locales"

for d in [MODELS_DIR, OUT_DIR, TMP_DIR, CERTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Migrate old flat model files into versioned subfolder
_old_model = MODELS_DIR / "kokoro-v1.0.onnx"
_old_voices = MODELS_DIR / "voices-v1.0.bin"
if _old_model.exists() and not (_model_dir("v1.0") / "kokoro-v1.0.onnx").exists():
    _old_model.rename(_model_dir("v1.0") / "kokoro-v1.0.onnx")
if _old_voices.exists() and not (_model_dir("v1.0") / "voices-v1.0.bin").exists():
    _old_voices.rename(_model_dir("v1.0") / "voices-v1.0.bin")

engine: Kokoro | None = None
voices: list[str] = []

# Map voice prefix to training language
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


def voices_for_lang(lang: str, all_voices: list[str], show_all: bool = False) -> list[str]:
    if show_all:
        return all_voices
    matching = [v for v in all_voices if VOICE_LANG_MAP.get(v.split("_")[0]) == lang]
    return matching if matching else all_voices


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


def save_ui_language(lang: str) -> str:
    settings = load_settings()
    settings["ui_language"] = lang if lang in {"nb", "en"} else "nb"
    save_settings(settings)
    return tr("language_saved")


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


def _parse_version(tag: str) -> tuple[int, ...]:
    nums = re.findall(r"\d+", tag or "")
    return tuple(int(n) for n in nums[:3]) if nums else (0,)


def get_latest_release() -> tuple[str, str, str]:
    req = urllib.request.Request(
        LATEST_RELEASE_API,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "KokoroTTS"},
    )
    with urllib.request.urlopen(req, timeout=6, context=_ssl_context()) as resp:
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
        latest_tag, latest_url, _ = get_latest_release()
    except Exception:
        return tr("update_check_failed", version=APP_VERSION)

    if _parse_version(latest_tag) > _parse_version(APP_VERSION):
        return tr("update_available", tag=latest_tag, version=APP_VERSION)

    return tr("up_to_date", version=APP_VERSION)


def _get_app_bundle_path() -> Path | None:
    """Return the path to the running .app bundle, or None if not frozen."""
    if not getattr(sys, "frozen", False):
        return None
    exe = Path(sys.executable).resolve()
    # PyInstaller: .app/Contents/MacOS/<name>  →  walk up to .app
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
        # Dev mode — fall back to opening the release page
        import webbrowser
        webbrowser.open(latest_url)
        return tr("update_dev_mode", url=latest_url)

    # --- Real in-place update ---
    tmp_dir = tempfile.mkdtemp(prefix="kokoro-update-")
    dmg_path = Path(tmp_dir) / "update.dmg"
    mount_point = Path(tmp_dir) / "mount"
    mount_point.mkdir()

    try:
        req = urllib.request.Request(dmg_url, headers={"User-Agent": "KokoroTTS"})
        with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as resp, open(dmg_path, "wb") as out_f:
            shutil.copyfileobj(resp, out_f)
    except Exception as e:
        return tr("download_failed", error=str(e))

    try:
        subprocess.run(
            ["hdiutil", "attach", str(dmg_path), "-mountpoint", str(mount_point), "-nobrowse", "-quiet"],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        return tr("update_mount_failed", error=e.stderr.decode(errors="replace").strip())

    # Find the .app inside the mounted DMG
    new_app = None
    for item in mount_point.iterdir():
        if item.suffix == ".app" and item.is_dir():
            new_app = item
            break

    if new_app is None:
        subprocess.run(["hdiutil", "detach", str(mount_point), "-quiet"], capture_output=True)
        return tr("update_no_app_in_dmg")

    install_dest = current_app.parent  # e.g. /Applications
    app_name = current_app.name        # e.g. KokoroTTS.app

    # Write an updater script that runs after we quit
    updater_script = Path(tmp_dir) / "updater.sh"
    updater_script.write_text(f"""#!/bin/bash
# Wait for the app to quit
PID={os.getpid()}
for i in {{1..30}}; do
    kill -0 "$PID" 2>/dev/null || break
    sleep 0.5
done

# Replace the old app
rm -rf "{install_dest / app_name}"
cp -pR "{new_app}" "{install_dest / app_name}"

# Unmount and clean up
hdiutil detach "{mount_point}" -quiet 2>/dev/null
rm -rf "{tmp_dir}"

# Relaunch
open "{install_dest / app_name}"
""", encoding="utf-8")
    updater_script.chmod(0o755)

    # Launch the updater in the background and quit
    subprocess.Popen(
        ["/bin/bash", str(updater_script)],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Give Gradio a moment to send the response, then exit
    def _delayed_exit():
        import time
        time.sleep(1.5)
        os._exit(0)

    import threading
    threading.Thread(target=_delayed_exit, daemon=True).start()

    return tr("update_restarting", tag=latest_tag)


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "KokoroTTS"})
    with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as resp, open(dest, "wb") as out_f:
        shutil.copyfileobj(resp, out_f)


def _download_model(version: str) -> tuple[Path, Path]:
    """Download model files, trying mirror first then fallback."""
    info = MODEL_REGISTRY[version]
    model_path, voices_path = _model_paths(version)
    for model_url, voices_url in info["urls"]:
        try:
            _download(model_url, model_path)
            _download(voices_url, voices_path)
            return model_path, voices_path
        except Exception:
            # Remove partial downloads before trying next mirror
            for p in (model_path, voices_path):
                if p.exists() and p.stat().st_size == 0:
                    p.unlink()
            continue
    raise RuntimeError(f"Could not download model {version} from any mirror")


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


def apply_style(text: str, style: str, speed: float) -> tuple[str, float]:
    style_speed = {
        "Neutral": 1.00,
        "Direct": 1.07,
        "Angry": 1.15,
        "Calm": 0.92,
        "Warm": 0.96,
        "Sad": 0.90,
        "Cheerful": 1.10,
        "Narrator": 0.94,
        "Whisper-ish": 0.88,
        "Urgent": 1.20,
    }
    target = style_speed.get(style, 1.00)
    merged_speed = max(0.5, min(2.0, (float(speed) + target) / 2.0))

    styled_text = text.strip()
    if style == "Direct":
        styled_text = styled_text.replace("...", ".")
    elif style == "Angry":
        if not styled_text.endswith(("!", ".", "?")):
            styled_text += "!"
    elif style == "Calm":
        styled_text = styled_text.replace("!", ".")
    elif style == "Whisper-ish":
        styled_text = styled_text.replace("!", ".")
    elif style == "Urgent":
        if not styled_text.endswith("!"):
            styled_text += "!"

    return styled_text, merged_speed


def _db_to_gain(db: float) -> float:
    return float(10 ** (db / 20.0))


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


def toggle_favorite(voice: str):
    favorites = load_favorites()
    if voice in favorites:
        favorites.remove(voice)
        status = tr("fav_removed", voice=voice)
    else:
        favorites.append(voice)
        status = tr("fav_added", voice=voice)
    save_favorites(favorites)
    return gr.update(choices=favorites, value=(favorites[0] if favorites else None)), status


def apply_preset(preset: str):
    presets = {
        "neutral": ("Neutral", 1.00, 0.0),
        "alert": ("Urgent", 1.18, 2.0),
        "narration": ("Narrator", 0.92, -1.0),
        "direct": ("Direct", 1.08, 1.0),
    }
    style, speed, gain = presets.get(preset, presets["neutral"])
    return style, speed, gain


def synthesize(text: str, voice: str, speed: float, lang: str, style: str, gain_db: float, output_format: str, history: list):
    if not text or not text.strip():
        raise gr.Error(tr("error_empty_text"))

    styled_text, styled_speed = apply_style(text, style, speed)

    tts, _ = ensure_engine()
    try:
        audio, sample_rate = tts.create(
            text=styled_text,
            voice=voice,
            speed=styled_speed,
            lang=lang,
        )
    except RuntimeError as e:
        raise gr.Error(tr("error_lang_unsupported", lang=lang)) from e

    # Volume gain
    audio = np.clip(audio * _db_to_gain(float(gain_db)), -1.0, 1.0)

    # Build filename from text (max 40 chars), sanitised
    snippet = text.strip()[:40].rstrip()
    snippet_safe = re.sub(r'[^\w\s-]', '', snippet).strip()
    snippet_safe = re.sub(r'\s+', '_', snippet_safe) or "clip"
    timestamp = datetime.now().strftime('%H%M%S')
    ext = "mp3" if output_format.lower() == "mp3" else "wav"
    filename = f"{snippet_safe}_{timestamp}.{ext}"
    out_path = OUT_DIR / filename
    sf.write(out_path, audio, sample_rate)

    # Label for history list: truncated text + time
    label = f"{snippet[:30]}{'…' if len(text.strip()) > 30 else ''} ({timestamp[:2]}:{timestamp[2:4]})"
    new_history = [(label, str(out_path))] + list(history or [])
    history_choices = [(lbl, path) for lbl, path in new_history]

    return (
        str(out_path),
        new_history,
        gr.update(choices=history_choices, value=str(out_path)),
        str(out_path),
        tr("synth_done", voice=voice, lang=lang, style=style,
           speed=f"{styled_speed:.2f}", gain=f"{gain_db:+.1f}", format=ext.upper()),
    )


def build_ui() -> gr.Blocks:
    L = get_ui_language()
    t = lambda key, **kw: tr(key, L, **kw)
    _, v = ensure_engine()
    show_all = load_settings().get("show_all_voices", False)
    default_lang = "en-us"
    filtered = voices_for_lang(default_lang, v, show_all)
    default_voice = "af_heart" if "af_heart" in filtered else filtered[0]
    favorites = load_favorites()

    logo_file = BASE_DIR / "kokorotts.png"
    logo_b64 = ""
    try:
        logo_b64 = base64.b64encode(logo_file.read_bytes()).decode()
    except Exception:
        pass
    css = """
    .main-btn { min-height: 46px !important; font-size: 1.1em !important; }
    .header-wrap { text-align: center; padding: 12px 0 4px; }
    .header-wrap img { height: 72px; border-radius: 14px; vertical-align: middle; margin-right: 10px; }
    .header-wrap .ver { color: #888; font-size: 0.85em; margin-top: 2px; }
    """

    logo_html = f'<img src="data:image/png;base64,{logo_b64}" alt="">' if logo_b64 else ""

    with gr.Blocks(title="Kokoro TTS", css=css) as demo:
        gr.HTML(f"""
            <div class="header-wrap">
                {logo_html}
                <span style="font-size:1.8em; font-weight:700; vertical-align:middle;">{t('title')}</span>
                <div class="ver">{APP_VERSION}</div>
            </div>
        """)

        with gr.Tabs():
            with gr.Tab(t("tab_generate")):
                text = gr.Textbox(
                    lines=5, max_lines=12,
                    label=t("text"),
                    placeholder=t("text_placeholder"),
                )

                with gr.Group():
                    gr.Markdown(f"#### {t('voice_group')}")
                    with gr.Row():
                        voice = gr.Dropdown(choices=filtered, value=default_voice, label=t("voice"), scale=3)
                        favorite_voice = gr.Dropdown(
                            choices=favorites,
                            value=(favorites[0] if favorites else None),
                            label=t("favorites"),
                            scale=2,
                        )
                        fav_btn = gr.Button(t("toggle_favorite"), scale=1)

                with gr.Group():
                    gr.Markdown(f"#### {t('audio_settings')}")
                    with gr.Row():
                        lang = gr.Dropdown(
                            choices=[
                                ("English (US)", "en-us"),
                                ("English (UK)", "en-gb"),
                                ("Norsk", "nb"),
                                ("Svenska", "sv"),
                                ("Dansk", "da"),
                                ("Deutsch", "de"),
                                ("Fran\u00e7ais", "fr-fr"),
                                ("Espa\u00f1ol", "es"),
                                ("Italiano", "it"),
                                ("Portugu\u00eas (BR)", "pt-br"),
                                ("\u65e5\u672c\u8a9e", "ja"),
                                ("\u4e2d\u6587", "cmn"),
                            ],
                            value="en-us",
                            label=t("language_code"),
                        )
                        style = gr.Dropdown(
                            choices=["Neutral", "Direct", "Angry", "Calm", "Warm", "Sad", "Cheerful", "Narrator", "Whisper-ish", "Urgent"],
                            value="Neutral",
                            label=t("style"),
                        )
                        preset = gr.Dropdown(
                            choices=[
                                (t("preset_neutral"), "neutral"),
                                (t("preset_alert"), "alert"),
                                (t("preset_narration"), "narration"),
                                (t("preset_direct"), "direct"),
                            ],
                            value="neutral",
                            label=t("preset"),
                        )
                    with gr.Row():
                        speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label=t("base_speed"))
                        gain_db = gr.Slider(-12.0, 12.0, value=0.0, step=0.5, label=t("volume_db"))
                        output_format = gr.Dropdown(choices=["wav", "mp3"], value="wav", label=t("format"))

                btn = gr.Button(t("generate"), variant="primary", elem_classes=["main-btn"])

                audio = gr.Audio(type="filepath", label=t("preview"))
                file_history = gr.State(value=[])
                history_radio = gr.Radio(
                    choices=[], value=None,
                    label=t("history"), interactive=True,
                )
                with gr.Row():
                    download_btn = gr.Button("⬇ " + t("download"))
                download_file = gr.File(label=t("download"), visible=False)
                info = gr.Textbox(label=t("info"), interactive=False)

            with gr.Tab(t("tab_settings")):
                with gr.Group():
                    gr.Markdown(f"#### {t('ui_language')}")
                    ui_language = gr.Dropdown(
                        choices=[("Norsk", "nb"), ("English", "en")],
                        value=L,
                        label=t("ui_language"),
                    )
                    lang_info = gr.Textbox(label=t("info"), interactive=False)

                with gr.Group():
                    gr.Markdown(f"#### {t('voice_settings')}")
                    show_all_voices = gr.Checkbox(
                        value=show_all,
                        label=t("show_all_voices"),
                    )

                with gr.Group():
                    gr.Markdown(f"#### {t('model_settings')}")
                    current_model = get_model_version()
                    model_choices = [(f"Kokoro {k}", k) for k in MODEL_REGISTRY]
                    model_selector = gr.Dropdown(
                        choices=model_choices,
                        value=current_model,
                        label=t("model_version"),
                    )
                    model_info = gr.Textbox(label=t("info"), interactive=False)

                with gr.Group():
                    gr.Markdown(f"#### {t('security_settings')}")
                    ssl_enabled = gr.Checkbox(
                        value=load_settings().get("ssl_enabled", False),
                        label=t("enable_ssl"),
                    )
                    ssl_info = gr.Textbox(label=t("info"), interactive=False)

                with gr.Group():
                    gr.Markdown(f"#### {t('app_status')}")
                    update_status = gr.Textbox(
                        label=t("app_status"),
                        value=check_updates_message(),
                        interactive=False,
                    )
                    with gr.Row():
                        check_update_btn = gr.Button(t("check_update"))
                        auto_update_btn = gr.Button(t("update_now"))

        # Events
        def _update_voices(selected_lang, all_voices_on):
            settings = load_settings()
            settings["show_all_voices"] = all_voices_on
            save_settings(settings)
            _, current_voices = ensure_engine()
            filtered_v = voices_for_lang(selected_lang, current_voices, all_voices_on)
            current = voice.value
            new_val = current if current in filtered_v else (filtered_v[0] if filtered_v else current_voices[0])
            return gr.update(choices=filtered_v, value=new_val)

        lang.change(fn=_update_voices, inputs=[lang, show_all_voices], outputs=[voice])
        show_all_voices.change(fn=_update_voices, inputs=[lang, show_all_voices], outputs=[voice])

        def _play_selected(path):
            return path if path else None

        def _download_selected(path):
            if path and Path(path).exists():
                return gr.update(value=path, visible=True)
            return gr.update(visible=False)

        btn.click(
            fn=synthesize,
            inputs=[text, voice, speed, lang, style, gain_db, output_format, file_history],
            outputs=[audio, file_history, history_radio, download_file, info],
        )
        history_radio.change(fn=_play_selected, inputs=[history_radio], outputs=[audio])
        download_btn.click(fn=_download_selected, inputs=[history_radio], outputs=[download_file])
        def _switch_model(version, selected_lang, all_voices_on):
            settings = load_settings()
            settings["model_version"] = version
            save_settings(settings)
            try:
                _, new_voices = ensure_engine(version)
            except Exception as e:
                return gr.update(), gr.update(), tr("model_switch_error", error=str(e))
            filtered_v = voices_for_lang(selected_lang, new_voices, all_voices_on)
            default_v = filtered_v[0] if filtered_v else new_voices[0]
            return (
                gr.update(choices=filtered_v, value=default_v),
                gr.update(choices=new_voices, value=default_v),
                tr("model_switched", version=version),
            )

        model_selector.change(
            fn=_switch_model,
            inputs=[model_selector, lang, show_all_voices],
            outputs=[voice, voice, model_info],
        )
        check_update_btn.click(fn=check_updates_message, outputs=[update_status])
        auto_update_btn.click(fn=auto_update, outputs=[update_status])
        ui_language.change(fn=save_ui_language, inputs=[ui_language], outputs=[lang_info])
        fav_btn.click(fn=toggle_favorite, inputs=[voice], outputs=[favorite_voice, info])
        favorite_voice.change(fn=lambda x: x, inputs=[favorite_voice], outputs=[voice])
        preset.change(fn=apply_preset, inputs=[preset], outputs=[style, speed, gain_db])

        def _toggle_ssl(enabled):
            settings = load_settings()
            settings["ssl_enabled"] = enabled
            save_settings(settings)
            return tr("ssl_restart_required")

        ssl_enabled.change(fn=_toggle_ssl, inputs=[ssl_enabled], outputs=[ssl_info])

    return demo


def _ensure_ssl() -> tuple[str, str] | None:
    """Generate a self-signed cert for localhost on first run, trust it in macOS keychain."""
    cert_file = CERTS_DIR / "localhost.pem"
    key_file = CERTS_DIR / "localhost-key.pem"
    if cert_file.exists() and key_file.exists():
        return str(cert_file), str(key_file)
    try:
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(key_file), "-out", str(cert_file),
            "-days", "3650", "-nodes",
            "-subj", "/CN=localhost",
            "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
        ], check=True, capture_output=True)
        # Trust the cert in macOS keychain (may prompt for password)
        subprocess.run([
            "security", "add-trusted-cert", "-p", "ssl",
            "-k", str(Path.home() / "Library/Keychains/login.keychain-db"),
            str(cert_file),
        ], capture_output=True)
        return str(cert_file), str(key_file)
    except Exception:
        return None


if __name__ == "__main__":
    ui = build_ui()
    launch_kwargs = dict(
        server_name="0.0.0.0",
        server_port=7861,
        allowed_paths=[str(TMP_DIR), str(OUT_DIR), str(BASE_DIR)],
    )
    if load_settings().get("ssl_enabled", False):
        ssl = _ensure_ssl()
        if ssl:
            launch_kwargs["ssl_certfile"], launch_kwargs["ssl_keyfile"] = ssl
    ui.launch(**launch_kwargs)
