#!/usr/bin/env python3
"""Mac menubar launcher for Kokoro TTS WebUI."""

from __future__ import annotations

import os
import threading
import time
import webbrowser

import rumps

from app import build_ui

URL = "http://127.0.0.1:7861"


def start_webui() -> None:
    ui = build_ui()
    ui.launch(
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=False,
        share=False,
        prevent_thread_lock=True,
    )


class KokoroMenuBar(rumps.App):
    def __init__(self):
        super().__init__("KokoroTTS", quit_button=None)
        self.menu = ["Åpne KokoroTTS", "Avslutt KokoroTTS"]

    @rumps.clicked("Åpne KokoroTTS")
    def open_ui(self, _):
        webbrowser.open(URL)

    @rumps.clicked("Avslutt KokoroTTS")
    def quit_app(self, _):
        rumps.quit_application()
        os._exit(0)


if __name__ == "__main__":
    threading.Thread(target=start_webui, daemon=True).start()
    time.sleep(1.5)
    webbrowser.open(URL)
    KokoroMenuBar().run()
