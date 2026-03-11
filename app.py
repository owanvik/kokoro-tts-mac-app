#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import re
import sys
import urllib.request
from pathlib import Path

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

MODEL_PATH = MODELS_DIR / "kokoro-v1.0.onnx"
VOICES_PATH = MODELS_DIR / "voices-v1.0.bin"

MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

APP_VERSION = f"v{(BASE_DIR / 'VERSION').read_text(encoding='utf-8').strip()}"
GITHUB_REPO = "owanvik/kokoro-tts-mac-app"
LATEST_RELEASE_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
LATEST_RELEASE_URL = f"https://github.com/{GITHUB_REPO}/releases/latest"
FAVORITES_FILE = APP_DIR / "favorites.json"
SETTINGS_FILE = APP_DIR / "settings.json"
LOCALES_DIR = BASE_DIR / "locales"

for d in [MODELS_DIR, OUT_DIR, TMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

engine: Kokoro | None = None
voices: list[str] = []


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


def tr(key: str, lang: str | None = None) -> str:
    lang = lang or get_ui_language()
    path = LOCALES_DIR / f"{lang}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get(key, key)
    except Exception:
        return key


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
        latest_tag, latest_url, _ = get_latest_release()
    except Exception:
        return f"Versjon: {APP_VERSION} | Oppdateringssjekk utilgjengelig nå."

    if _parse_version(latest_tag) > _parse_version(APP_VERSION):
        return f"Oppdatering tilgjengelig: {latest_tag} (du har {APP_VERSION}). Klar for automatisk oppdatering."

    return f"Du har nyeste versjon ({APP_VERSION})."


def auto_update() -> str:
    try:
        latest_tag, latest_url, dmg_url = get_latest_release()
    except Exception as e:
        return f"Kunne ikke sjekke oppdatering nå: {e}"

    if _parse_version(latest_tag) <= _parse_version(APP_VERSION):
        return f"Du har allerede nyeste versjon ({APP_VERSION})."

    if not dmg_url:
        return f"Fant ikke DMG i release {latest_tag}. Åpne manuelt: {latest_url}"

    downloads = Path.home() / "Downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    dmg_path = downloads / f"KokoroTTS-{latest_tag}-mac-arm64.dmg"

    try:
        urllib.request.urlretrieve(dmg_url, dmg_path)
        subprocess.Popen(["open", str(dmg_path)])
    except Exception as e:
        return f"Nedlasting/åpning feilet: {e}"

    return (
        f"Ny versjon {latest_tag} lastet ned. Installasjonsvindu åpnes nå. "
        "Dra appen til Applications for å fullføre oppdatering."
    )


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    urllib.request.urlretrieve(url, dest)


def ensure_engine() -> tuple[Kokoro, list[str]]:
    global engine, voices
    if engine is None:
        _download(MODEL_URL, MODEL_PATH)
        _download(VOICES_URL, VOICES_PATH)
        engine = Kokoro(model_path=str(MODEL_PATH), voices_path=str(VOICES_PATH))
        voices = engine.get_voices()
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
        status = f"Fjernet favoritt: {voice}"
    else:
        favorites.append(voice)
        status = f"La til favoritt: {voice}"
    save_favorites(favorites)
    return gr.update(choices=favorites, value=(favorites[0] if favorites else None)), status


def apply_preset(preset: str):
    presets = {
        "Nøytral": ("Neutral", 1.00, 0.0),
        "Varsel": ("Urgent", 1.18, 2.0),
        "Fortelling": ("Narrator", 0.92, -1.0),
        "Direkte": ("Direct", 1.08, 1.0),
    }
    style, speed, gain = presets.get(preset, presets["Nøytral"])
    return style, speed, gain


def synthesize(text: str, voice: str, speed: float, lang: str, style: str, gain_db: float, output_format: str):
    if not text or not text.strip():
        raise gr.Error("Skriv inn tekst først")

    # Normalize a few user-friendly aliases to eSpeak-supported language codes.
    lang_alias = {
        "no": "nb",
        "zh": "cmn",
    }
    resolved_lang = lang_alias.get(lang, lang)

    styled_text, styled_speed = apply_style(text, style, speed)

    tts, _ = ensure_engine()
    try:
        audio, sample_rate = tts.create(
            text=styled_text,
            voice=voice,
            speed=styled_speed,
            lang=resolved_lang,
        )
    except RuntimeError as e:
        raise gr.Error(f"Språkkoden '{lang}' støttes ikke av motoren. Prøv f.eks. 'nb' for norsk.") from e

    # Volume gain
    audio = np.clip(audio * _db_to_gain(float(gain_db)), -1.0, 1.0)

    stem = voice.replace("/", "_")
    ext = "mp3" if output_format.lower() == "mp3" else "wav"
    out_path = TMP_DIR / f"kokoro-{stem}.{ext}"
    sf.write(out_path, audio, sample_rate)

    return str(out_path), str(out_path), (
        f"Klar: {voice} | lang={lang} | style={style} | speed={styled_speed:.2f} | gain={gain_db:+.1f} dB | format={ext.upper()}"
    )


def build_ui() -> gr.Blocks:
    ui_lang = get_ui_language()
    _, v = ensure_engine()
    default_voice = "af_heart" if "af_heart" in v else v[0]
    favorites = load_favorites()

    with gr.Blocks(title="Kokoro TTS") as demo:
        gr.Markdown(f"## {tr('title', ui_lang)}")
        with gr.Row():
            ui_language = gr.Dropdown(choices=["nb", "en"], value=ui_lang, label=tr("ui_language", ui_lang))
            save_language_btn = gr.Button(tr("save_language", ui_lang))
        update_status = gr.Textbox(label=tr("app_status", ui_lang), value=check_updates_message(), interactive=False)
        with gr.Row():
            check_update_btn = gr.Button(tr("check_update", ui_lang))
            auto_update_btn = gr.Button(tr("update_now", ui_lang))

        text = gr.Textbox(lines=6, label=tr("text", ui_lang), placeholder=tr("text_placeholder", ui_lang))
        with gr.Row():
            voice = gr.Dropdown(choices=v, value=default_voice, label=tr("voice", ui_lang))
            favorite_voice = gr.Dropdown(choices=favorites, value=(favorites[0] if favorites else None), label=tr("favorites", ui_lang))
            fav_btn = gr.Button(tr("toggle_favorite", ui_lang))
        with gr.Row():
            lang = gr.Dropdown(
                choices=["en-us", "en-gb", "nb", "no", "sv", "da", "de", "fr-fr", "es", "it", "pt-br", "ja", "cmn", "zh"],
                value="en-us",
                label=tr("language_code", ui_lang),
            )
            style = gr.Dropdown(
                choices=["Neutral", "Direct", "Angry", "Calm", "Warm", "Sad", "Cheerful", "Narrator", "Whisper-ish", "Urgent"],
                value="Neutral",
                label=tr("style", ui_lang),
            )
            speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label=tr("base_speed", ui_lang))
            gain_db = gr.Slider(-12.0, 12.0, value=0.0, step=0.5, label=tr("volume_db", ui_lang))
            output_format = gr.Dropdown(choices=["wav", "mp3"], value="wav", label=tr("format", ui_lang))
        with gr.Row():
            preset = gr.Dropdown(choices=["Nøytral", "Varsel", "Fortelling", "Direkte"], value="Nøytral", label=tr("preset", ui_lang))
            apply_preset_btn = gr.Button(tr("apply_preset", ui_lang))
            btn = gr.Button(tr("generate", ui_lang), variant="primary")

        audio = gr.Audio(type="filepath", label=tr("preview", ui_lang))
        download = gr.File(label=tr("download", ui_lang))
        info = gr.Textbox(label=tr("info", ui_lang))

        btn.click(fn=synthesize, inputs=[text, voice, speed, lang, style, gain_db, output_format], outputs=[audio, download, info])
        check_update_btn.click(fn=check_updates_message, outputs=[update_status])
        auto_update_btn.click(fn=auto_update, outputs=[update_status])
        save_language_btn.click(fn=save_ui_language, inputs=[ui_language], outputs=[info])
        fav_btn.click(fn=toggle_favorite, inputs=[voice], outputs=[favorite_voice, info])
        favorite_voice.change(fn=lambda x: x, inputs=[favorite_voice], outputs=[voice])
        apply_preset_btn.click(fn=apply_preset, inputs=[preset], outputs=[style, speed, gain_db])

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7861,
        allowed_paths=[str(TMP_DIR), str(OUT_DIR)],
    )
