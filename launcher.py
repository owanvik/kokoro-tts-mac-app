#!/usr/bin/env python3
"""Mac app launcher for Kokoro TTS WebUI."""

import threading
import time
import webbrowser

from app import build_ui

URL = "http://127.0.0.1:7861"


def open_browser_delayed():
    time.sleep(2)
    webbrowser.open(URL)


if __name__ == "__main__":
    threading.Thread(target=open_browser_delayed, daemon=True).start()
    ui = build_ui()
    ui.launch(server_name="127.0.0.1", server_port=7861, inbrowser=False, share=False)
