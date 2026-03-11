#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

import gradio as gr
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

APP_VERSION = "v0.3.0"
GITHUB_REPO = "owanvik/kokoro-tts-mac-app"
LATEST_RELEASE_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
LATEST_RELEASE_URL = f"https://github.com/{GITHUB_REPO}/releases/latest"

for d in [MODELS_DIR, OUT_DIR, TMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

engine: Kokoro | None = None
voices: list[str] = []


def _parse_version(tag: str) -> tuple[int, ...]:
    nums = re.findall(r"\d+", tag or "")
    return tuple(int(n) for n in nums[:3]) if nums else (0,)


def get_latest_release() -> tuple[str, str]:
    req = urllib.request.Request(
        LATEST_RELEASE_API,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "KokoroTTS"},
    )
    with urllib.request.urlopen(req, timeout=6) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("tag_name", ""), data.get("html_url", LATEST_RELEASE_URL)


def check_updates_message() -> str:
    try:
        latest_tag, latest_url = get_latest_release()
    except Exception:
        return f"Versjon: {APP_VERSION} | Oppdateringssjekk utilgjengelig nå."

    if _parse_version(latest_tag) > _parse_version(APP_VERSION):
        return f"Oppdatering tilgjengelig: {latest_tag} (du har {APP_VERSION}). Last ned: {latest_url}"

    return f"Du har nyeste versjon ({APP_VERSION})."


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


def synthesize(text: str, voice: str, speed: float, lang: str, style: str):
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

    stem = voice.replace("/", "_")
    out_path = TMP_DIR / f"kokoro-{stem}.wav"
    sf.write(out_path, audio, sample_rate)

    return str(out_path), str(out_path), f"Klar: {voice} | lang={lang} | style={style} | speed={styled_speed:.2f}"


def build_ui() -> gr.Blocks:
    _, v = ensure_engine()
    default_voice = "af_heart" if "af_heart" in v else v[0]

    with gr.Blocks(title="Kokoro TTS") as demo:
        gr.Markdown("## Kokoro TTS WebUI")
        update_status = gr.Textbox(label="App-status", value=check_updates_message(), interactive=False)
        check_update_btn = gr.Button("Sjekk oppdatering")

        text = gr.Textbox(lines=6, label="Tekst", placeholder="Skriv tekst her...")
        with gr.Row():
            voice = gr.Dropdown(choices=v, value=default_voice, label="Stemme")
            lang = gr.Dropdown(
                choices=["en-us", "en-gb", "nb", "no", "sv", "da", "de", "fr-fr", "es", "it", "pt-br", "ja", "cmn", "zh"],
                value="en-us",
                label="Språkkode",
            )
            style = gr.Dropdown(
                choices=["Neutral", "Direct", "Angry", "Calm", "Warm", "Sad", "Cheerful", "Narrator", "Whisper-ish", "Urgent"],
                value="Neutral",
                label="Style",
            )
            speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Base hastighet")

        with gr.Row():
            btn = gr.Button("Generer", variant="primary")

        audio = gr.Audio(type="filepath", label="Forhåndslytt")
        download = gr.File(label="Nedlasting")
        info = gr.Textbox(label="Info")

        btn.click(fn=synthesize, inputs=[text, voice, speed, lang, style], outputs=[audio, download, info])
        check_update_btn.click(fn=check_updates_message, outputs=[update_status])

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7861)
