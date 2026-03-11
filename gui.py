#!/usr/bin/env python3
"""Kokoro TTS – modern native macOS GUI with waveform player."""
from __future__ import annotations

import shutil
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
import numpy as np
import sounddevice as sd
import soundfile as sf

from core import (
    APP_VERSION, APP_DISPLAY_VERSION, BASE_DIR, OUT_DIR,
    MODEL_REGISTRY, LANGUAGE_CHOICES, STYLES,
    load_settings, save_settings, get_ui_language, get_model_version,
    load_favorites, toggle_favorite, voices_for_lang, ensure_engine,
    apply_preset, synthesize, check_updates_message, auto_update, tr,
)

# ── Colors ───────────────────────────────────────────────────────────
BG_DARK    = "#0c0c0e"
BG_CARD    = "#16161a"
BG_INPUT   = "#1e1e24"
ACCENT     = "#f97316"
ACCENT_DIM = "#c2410c"
TEXT_PRI   = "#ebebf0"
TEXT_SEC   = "#9898a4"
TEXT_DIM   = "#5c5c6a"
BORDER     = "#2a2a34"
WAVE_BG    = "#111114"
WAVE_FG    = "#a84e10"
WAVE_PLAY  = "#f97316"
PLAYHEAD   = "#ffffff"

PAD  = 12
HALF = 6
RADIUS = 12


# ── Patch: CTkOptionMenu dropdown opens at selected item ────────────
def _patch_option_menu_dropdown():
    """Override CTkOptionMenu._open_dropdown_menu so the native menu opens
    with the current value pre-selected, enabling arrow-key navigation."""
    _orig = ctk.CTkOptionMenu._open_dropdown_menu

    def _open_at_selected(self):
        menu = self._dropdown_menu
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height()
        current = (self._variable.get() if self._variable else
                   getattr(self, "_current_value", ""))
        idx = 0
        for i, v in enumerate(getattr(menu, "_values", []) or []):
            if v == current:
                idx = i
                break
        menu.tk_popup(int(x), int(y), idx)

    ctk.CTkOptionMenu._open_dropdown_menu = _open_at_selected

_patch_option_menu_dropdown()


# ── Waveform Player ─────────────────────────────────────────────────
class WaveformPlayer(ctk.CTkFrame):
    """Audio waveform visualiser with integrated playback controls."""

    def __init__(self, master, **kw):
        super().__init__(master, fg_color=BG_CARD, corner_radius=RADIUS, **kw)
        self._audio: np.ndarray | None = None
        self._sr: int = 24000
        self._duration: float = 0.0
        self._playing = False
        self._paused = False
        self._play_start: float = 0.0
        self._pause_offset: float = 0.0
        self._peaks: list[float] = []
        self._file_path: str | None = None

        # Canvas
        self._canvas = tk.Canvas(
            self, bg=WAVE_BG, highlightthickness=0, height=64, cursor="hand2",
        )
        self._canvas.pack(fill="x", padx=PAD, pady=(HALF, 4))
        self._canvas.bind("<Configure>", lambda e: self._draw_waveform())
        self._canvas.bind("<Button-1>", self._on_canvas_click)

        # Controls row
        ctrl = ctk.CTkFrame(self, fg_color="transparent")
        ctrl.pack(fill="x", padx=PAD, pady=(0, HALF))

        self._time_label = ctk.CTkLabel(
            ctrl, text="0:00 / 0:00", text_color=TEXT_SEC,
            font=ctk.CTkFont(size=12, family="SF Mono"),
        )
        self._time_label.pack(side="left")

        btn_frame = ctk.CTkFrame(ctrl, fg_color="transparent")
        btn_frame.pack(side="right")

        btn_kw = dict(
            width=32, height=32, corner_radius=16, fg_color=BG_INPUT,
            hover_color=BORDER, text_color=TEXT_PRI, font=ctk.CTkFont(size=14),
        )

        self._prev_btn = ctk.CTkButton(
            btn_frame, text="⏮", command=self._rewind, **btn_kw,
        )
        self._prev_btn.pack(side="left", padx=2)

        self._play_btn = ctk.CTkButton(
            btn_frame, text="▶", command=self._toggle_play,
            width=38, height=38, corner_radius=19,
            fg_color=ACCENT, hover_color=ACCENT_DIM,
            text_color="#fff", font=ctk.CTkFont(size=18),
        )
        self._play_btn.pack(side="left", padx=6)

        self._next_btn = ctk.CTkButton(
            btn_frame, text="⏭", command=self._forward, **btn_kw,
        )
        self._next_btn.pack(side="left", padx=2)

    # ── Audio loading ────────────────────────────────────────────────
    def load_file(self, path: str) -> None:
        self._stop_playback()
        self._file_path = path
        try:
            data, sr = sf.read(path)
            if data.ndim > 1:
                data = data.mean(axis=1)
            self._audio = data.astype(np.float32)
            self._sr = sr
            self._duration = len(data) / sr
            self._compute_peaks()
            self._draw_waveform()
            self._update_time(0.0)
        except Exception:
            self._audio = None
            self._duration = 0.0

    def _compute_peaks(self):
        if self._audio is None:
            self._peaks = []
            return
        n = len(self._audio)
        num_bars = 400
        chunk = max(1, n // num_bars)
        peaks = []
        for i in range(0, n, chunk):
            seg = self._audio[i : i + chunk]
            peaks.append(float(np.max(np.abs(seg))))
        mx = max(peaks) if peaks else 1.0
        self._peaks = [p / mx if mx > 0 else 0.0 for p in peaks]

    # ── Drawing ──────────────────────────────────────────────────────
    def _draw_waveform(self, position: float = 0.0):
        c = self._canvas
        c.delete("all")
        w = c.winfo_width() or 600
        h = c.winfo_height() or 100

        if not self._peaks:
            # Empty placeholder
            mid = h // 2
            for x in range(0, w, 6):
                c.create_line(x + 1, mid - 4, x + 1, mid + 4, fill=TEXT_DIM, width=2)
            return

        n = len(self._peaks)
        bar_w = max(2, w / n)
        gap = max(1, bar_w * 0.25)
        mid = h / 2
        play_x = (position / self._duration * w) if self._duration > 0 else 0

        for i, p in enumerate(self._peaks):
            x = i * bar_w
            bh = max(2, p * (mid - 4))
            color = WAVE_PLAY if x < play_x else WAVE_FG
            c.create_line(
                x + gap, mid - bh, x + gap, mid + bh,
                fill=color, width=max(1.5, bar_w - gap),
            )

        # Playhead
        if self._duration > 0 and position > 0:
            c.create_line(play_x, 0, play_x, h, fill=PLAYHEAD, width=2)

    def _update_time(self, pos: float):
        total = self._duration

        def fmt(s):
            m, sec = divmod(max(0, int(s)), 60)
            return f"{m}:{sec:02d}"

        self._time_label.configure(text=f"{fmt(pos)} / {fmt(total)}")

    # ── Transport ────────────────────────────────────────────────────
    def _toggle_play(self):
        if self._playing and not self._paused:
            self._pause()
        elif self._paused:
            self._resume()
        else:
            self._play()

    def _play(self, start_offset: float = 0.0):
        if self._audio is None:
            return
        self._stop_playback()
        self._playing = True
        self._paused = False
        self._pause_offset = start_offset
        self._play_btn.configure(text="⏸")

        start_frame = int(start_offset * self._sr)
        audio_slice = self._audio[start_frame:]
        sd.play(audio_slice, self._sr)
        self._play_start = time.time() - start_offset
        self._tick()

    def _pause(self):
        self._paused = True
        self._pause_offset = time.time() - self._play_start
        sd.stop()
        self._play_btn.configure(text="▶")

    def _resume(self):
        self._play(self._pause_offset)

    def _stop_playback(self):
        self._playing = False
        self._paused = False
        self._pause_offset = 0.0
        sd.stop()
        self._play_btn.configure(text="▶")

    def _rewind(self):
        self._play(0.0)

    def _forward(self):
        if self._duration > 0:
            if self._playing:
                pos = min(
                    self._duration - 0.5,
                    (time.time() - self._play_start) + 5.0,
                )
            else:
                pos = 5.0
            self._play(max(0, pos))

    def _on_canvas_click(self, event):
        if self._duration <= 0:
            return
        w = self._canvas.winfo_width()
        frac = max(0.0, min(1.0, event.x / w))
        self._play(frac * self._duration)

    def _tick(self):
        if not self._playing or self._paused:
            return
        pos = time.time() - self._play_start
        if pos >= self._duration:
            self._stop_playback()
            self._draw_waveform(self._duration)
            self._update_time(self._duration)
            return
        self._draw_waveform(pos)
        self._update_time(pos)
        self.after(50, self._tick)


# ── Main Application ─────────────────────────────────────────────────
class KokoroApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self._L = get_ui_language()
        self.title(f"Kokoro TTS  {APP_DISPLAY_VERSION}")
        self.geometry("860x820")
        self.minsize(700, 650)
        self.configure(fg_color=BG_DARK)

        self._history: list[tuple[str, str]] = []
        self._selected_history: int = 0

        self._logo = None
        logo_path = BASE_DIR / "kokorotts.png"
        if logo_path.exists():
            try:
                from PIL import Image

                img = Image.open(logo_path).resize((64, 64))
                self._logo = ctk.CTkImage(
                    light_image=img, dark_image=img, size=(36, 36),
                )
            except Exception:
                pass

        self._build_ui()
        self._set_status(self._t("loading_model"))
        threading.Thread(target=self._init_engine, daemon=True).start()

    # ── Helpers ──────────────────────────────────────────────────────
    def _t(self, key, **kw):
        return tr(key, self._L, **kw)

    def _set_status(self, msg):
        self._info_var.set(msg)

    def _card(self, parent):
        f = ctk.CTkFrame(
            parent, fg_color=BG_CARD, corner_radius=RADIUS,
            border_width=1, border_color=BORDER,
        )
        f.pack(fill="x", padx=0, pady=(0, 6))
        return f

    def _section_heading(self, parent, text):
        ctk.CTkLabel(
            parent, text=text, text_color=TEXT_PRI,
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=PAD, pady=(HALF, 4))

    def _setup_global_scroll(self):
        """Bind trackpad/mousewheel scroll to the active tab's canvas.

        Tk 9.0 (TIP 684) uses <TouchpadScroll> natively on macOS.
        We bind on 'all' in Tcl and route to the active canvas via a Tcl variable.
        """
        # Detect macOS "natural scrolling" setting (default=1, reversed=0)
        import subprocess
        try:
            r = subprocess.run(
                ["defaults", "read", "NSGlobalDomain",
                 "com.apple.swipescrolldirection"],
                capture_output=True, text=True, timeout=2,
            )
            natural = r.stdout.strip() != "0"
        except Exception:
            natural = True
        scroll_sign = -1 if natural else 1

        # Collect canvas widgets from CTkScrollableFrame internals
        canvases = []
        for sf in (self._scroll, self._settings_scroll):
            c = getattr(sf, "_parent_canvas", None)
            if c is None:
                for child in sf.winfo_children():
                    if isinstance(child, tk.Canvas):
                        c = child
                        break
            if c is not None:
                c.configure(yscrollincrement=2)
                canvases.append(c)

        self._scroll_canvases = canvases

        # Store the active canvas path in a Tcl variable
        active = str(canvases[0]) if canvases else ""
        self.tk.eval(f'set ::_active_canvas {active}')

        # Update Tcl variable when tab changes
        def _on_tab_change(*args):
            tab = self._tabs.get()
            idx = 0 if tab == self._t("tab_generate") else 1
            if idx < len(canvases):
                self.tk.eval(f'set ::_active_canvas {str(canvases[idx])}')
        self._tabs.configure(command=_on_tab_change)

        # Tk 9.0+: bind <TouchpadScroll> on all widgets, scroll active canvas
        # - Multiply delta for smoother feel (yscrollincrement=2 × 3 = ~6px/tick)
        # - Respect system scroll direction
        # - Suppress macOS rubber-band bounce at edges
        try:
            self.tk.eval(f'''
                set ::_scroll_sign {scroll_sign}
                set ::_edge_lock_top 0
                set ::_edge_lock_bot 0
                bind all <TouchpadScroll> {{
                    set dy [expr {{(%D >> 16) & 0xFFFF}}]
                    if {{$dy > 32767}} {{ set dy [expr {{$dy - 65536}}] }}
                    if {{$dy == 0 || $::_active_canvas eq ""}} return
                    set pos [$::_active_canvas yview]
                    set top [lindex $pos 0]
                    set bot [lindex $pos 1]
                    set amt [expr {{$::_scroll_sign * $dy * 3}}]
                    if {{$top <= 0.0 && $amt < 0}} {{
                        set ::_edge_lock_top 1
                        after cancel {{set ::_edge_lock_top 0}}
                        after 200 {{set ::_edge_lock_top 0}}
                    }}
                    if {{$bot >= 1.0 && $amt > 0}} {{
                        set ::_edge_lock_bot 1
                        after cancel {{set ::_edge_lock_bot 0}}
                        after 200 {{set ::_edge_lock_bot 0}}
                    }}
                    if {{$::_edge_lock_top && $amt > 0}} return
                    if {{$::_edge_lock_bot && $amt < 0}} return
                    $::_active_canvas yview scroll $amt units
                }}
            ''')
        except tk.TclError:
            pass

        # Fallback: <MouseWheel> for older Tk / external mouse
        sign = scroll_sign
        def _on_mousewheel(event):
            tab = self._tabs.get()
            idx = 0 if tab == self._t("tab_generate") else 1
            if idx < len(canvases):
                canvases[idx].yview_scroll(sign * event.delta * 3, "units")

        self.bind_all("<MouseWheel>", _on_mousewheel)

    # ── Build UI ─────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=PAD + 4, pady=(HALF, 2))
        if self._logo:
            ctk.CTkLabel(header, image=self._logo, text="").pack(side="left", padx=(0, 10))
        ctk.CTkLabel(
            header, text="Kokoro TTS",
            font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT_PRI,
        ).pack(side="left")
        ctk.CTkLabel(
            header, text=APP_DISPLAY_VERSION, text_color=TEXT_DIM,
            font=ctk.CTkFont(size=13),
        ).pack(side="left", padx=10)

        # Tabs
        self._tabs = ctk.CTkTabview(
            self, fg_color=BG_DARK,
            segmented_button_fg_color=BG_DARK,
            segmented_button_selected_color=ACCENT,
            segmented_button_unselected_color=BG_INPUT,
            segmented_button_selected_hover_color=ACCENT_DIM,
        )
        self._tabs.pack(fill="both", expand=True, padx=PAD, pady=(0, HALF))

        tab_gen = self._tabs.add(self._t("tab_generate"))
        tab_set = self._tabs.add(self._t("tab_settings"))
        tab_gen.configure(fg_color=BG_DARK)
        tab_set.configure(fg_color=BG_DARK)

        self._scroll = ctk.CTkScrollableFrame(tab_gen, fg_color=BG_DARK)
        self._scroll.pack(fill="both", expand=True)

        self._build_generate(self._scroll)
        self._build_settings(tab_set)

        self._setup_global_scroll()

    # ── Generate tab ─────────────────────────────────────────────────
    def _build_generate(self, parent):
        opt_kw = dict(
            fg_color=BG_INPUT, button_color=BORDER,
            button_hover_color=ACCENT_DIM,
            dropdown_fg_color=BG_INPUT, dropdown_hover_color=ACCENT,
        )

        # ── Text input ───────────────────────────────────────────────
        card = self._card(parent)
        ctk.CTkLabel(
            card, text=self._t("text"), text_color=TEXT_SEC,
            font=ctk.CTkFont(size=12),
        ).pack(anchor="w", padx=PAD, pady=(PAD, 2))
        self._text = ctk.CTkTextbox(
            card, height=72, fg_color=BG_INPUT, text_color=TEXT_PRI,
            border_width=1, border_color=BORDER, corner_radius=8,
            font=ctk.CTkFont(size=14),
        )
        self._text.pack(fill="x", padx=PAD, pady=(0, PAD))

        # ── Voice ────────────────────────────────────────────────────
        card = self._card(parent)
        ctk.CTkLabel(
            card, text=self._t("voice_group"), text_color=TEXT_PRI,
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=PAD, pady=(HALF, 4))

        vrow = ctk.CTkFrame(card, fg_color="transparent")
        vrow.pack(fill="x", padx=PAD, pady=(0, PAD))
        vrow.columnconfigure(0, weight=3, uniform="voice")
        vrow.columnconfigure(1, weight=2, uniform="voice")
        vrow.columnconfigure(2, weight=0)

        self._voice_var = ctk.StringVar(value="…")
        ctk.CTkLabel(
            vrow, text=self._t("voice"), text_color=TEXT_SEC,
            font=ctk.CTkFont(size=11),
        ).grid(row=0, column=0, sticky="w")
        self._voice_menu = ctk.CTkOptionMenu(
            vrow, variable=self._voice_var, values=["…"], **opt_kw,
        )
        self._voice_menu.grid(row=1, column=0, sticky="ew", padx=(0, HALF))

        self._fav_var = ctk.StringVar()
        favs = load_favorites()
        ctk.CTkLabel(
            vrow, text=self._t("favorites"), text_color=TEXT_SEC,
            font=ctk.CTkFont(size=11),
        ).grid(row=0, column=1, sticky="w")
        self._fav_menu = ctk.CTkOptionMenu(
            vrow, variable=self._fav_var,
            values=favs if favs else ["\u2014"],
            command=self._on_fav_select, **opt_kw,
        )
        self._fav_menu.grid(row=1, column=1, sticky="ew", padx=HALF)

        self._fav_btn = ctk.CTkButton(
            vrow, text="⭐", width=44, height=32, corner_radius=8,
            fg_color=ACCENT, hover_color=ACCENT_DIM, text_color="#fff",
            font=ctk.CTkFont(size=16), command=self._on_toggle_fav,
        )
        self._fav_btn.grid(row=1, column=2, padx=(HALF, 0))

        # ── Audio settings ───────────────────────────────────────────
        card = self._card(parent)
        ctk.CTkLabel(
            card, text=self._t("audio_settings"), text_color=TEXT_PRI,
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=PAD, pady=(HALF, 4))

        r1 = ctk.CTkFrame(card, fg_color="transparent")
        r1.pack(fill="x", padx=PAD, pady=(0, 8))
        for i in range(3):
            r1.columnconfigure(i, weight=1, uniform="audio1")

        lang_display = [name for name, _ in LANGUAGE_CHOICES]
        self._lang_var = ctk.StringVar(value=lang_display[0])
        ctk.CTkLabel(
            r1, text=self._t("language_code"), text_color=TEXT_SEC,
            font=ctk.CTkFont(size=11),
        ).grid(row=0, column=0, sticky="w")
        self._lang_menu = ctk.CTkOptionMenu(
            r1, variable=self._lang_var, values=lang_display,
            command=self._on_lang_change, **opt_kw,
        )
        self._lang_menu.grid(row=1, column=0, sticky="ew", padx=(0, HALF))

        self._style_var = ctk.StringVar(value="Neutral")
        ctk.CTkLabel(
            r1, text=self._t("style"), text_color=TEXT_SEC,
            font=ctk.CTkFont(size=11),
        ).grid(row=0, column=1, sticky="w")
        ctk.CTkOptionMenu(
            r1, variable=self._style_var, values=STYLES, **opt_kw,
        ).grid(row=1, column=1, sticky="ew", padx=HALF)

        preset_names = [
            self._t(f"preset_{k}") for k in ("neutral", "alert", "narration", "direct")
        ]
        self._preset_keys = ["neutral", "alert", "narration", "direct"]
        self._preset_var = ctk.StringVar(value=preset_names[0])
        ctk.CTkLabel(
            r1, text=self._t("preset"), text_color=TEXT_SEC,
            font=ctk.CTkFont(size=11),
        ).grid(row=0, column=2, sticky="w")
        ctk.CTkOptionMenu(
            r1, variable=self._preset_var, values=preset_names,
            command=self._on_preset, **opt_kw,
        ).grid(row=1, column=2, sticky="ew", padx=(HALF, 0))

        # Sliders row
        r2 = ctk.CTkFrame(card, fg_color="transparent")
        r2.pack(fill="x", padx=PAD, pady=(0, PAD))
        r2.columnconfigure(0, weight=2, uniform="audio2")
        r2.columnconfigure(1, weight=2, uniform="audio2")
        r2.columnconfigure(2, weight=1)

        self._speed_var = tk.DoubleVar(value=1.0)
        ctk.CTkLabel(
            r2, text=self._t("base_speed"), text_color=TEXT_SEC,
            font=ctk.CTkFont(size=11),
        ).grid(row=0, column=0, sticky="w")
        sf1 = ctk.CTkFrame(r2, fg_color="transparent")
        sf1.grid(row=1, column=0, sticky="ew", padx=(0, HALF))
        self._speed_slider = ctk.CTkSlider(
            sf1, from_=0.5, to=2.0, variable=self._speed_var,
            progress_color=ACCENT, button_color=ACCENT,
            button_hover_color=ACCENT_DIM, fg_color=BG_INPUT,
            command=lambda v: self._speed_lbl.configure(text=f"{v:.2f}"),
        )
        self._speed_slider.pack(side="left", fill="x", expand=True)
        self._speed_lbl = ctk.CTkLabel(
            sf1, text="1.00", text_color=TEXT_SEC, width=40,
            font=ctk.CTkFont(size=12, family="SF Mono"),
        )
        self._speed_lbl.pack(side="left", padx=(6, 0))

        self._gain_var = tk.DoubleVar(value=0.0)
        ctk.CTkLabel(
            r2, text=self._t("volume_db"), text_color=TEXT_SEC,
            font=ctk.CTkFont(size=11),
        ).grid(row=0, column=1, sticky="w")
        sf2 = ctk.CTkFrame(r2, fg_color="transparent")
        sf2.grid(row=1, column=1, sticky="ew", padx=HALF)
        self._gain_slider = ctk.CTkSlider(
            sf2, from_=-12.0, to=12.0, variable=self._gain_var,
            progress_color=ACCENT, button_color=ACCENT,
            button_hover_color=ACCENT_DIM, fg_color=BG_INPUT,
            command=lambda v: self._gain_lbl.configure(text=f"{v:+.1f}"),
        )
        self._gain_slider.pack(side="left", fill="x", expand=True)
        self._gain_lbl = ctk.CTkLabel(
            sf2, text="+0.0", text_color=TEXT_SEC, width=44,
            font=ctk.CTkFont(size=12, family="SF Mono"),
        )
        self._gain_lbl.pack(side="left", padx=(6, 0))

        ctk.CTkLabel(
            r2, text=self._t("format"), text_color=TEXT_SEC,
            font=ctk.CTkFont(size=11),
        ).grid(row=0, column=2, sticky="w")
        self._format_var = ctk.StringVar(value="wav")
        ctk.CTkOptionMenu(
            r2, variable=self._format_var, values=["wav", "mp3"], **opt_kw,
        ).grid(row=1, column=2, sticky="ew", padx=(HALF, 0))

        # ── Generate button ──────────────────────────────────────────
        self._gen_btn = ctk.CTkButton(
            parent, text=self._t("generate"), height=40,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=ACCENT, hover_color=ACCENT_DIM, text_color="#fff",
            corner_radius=RADIUS, command=self._on_generate,
        )
        self._gen_btn.pack(fill="x", padx=0, pady=(6, 6))

        # ── Waveform player ──────────────────────────────────────────
        self._player = WaveformPlayer(parent)
        self._player.pack(fill="x", padx=0, pady=(0, 6))

        # ── History ──────────────────────────────────────────────────
        card = self._card(parent)
        ctk.CTkLabel(
            card, text=self._t("history"), text_color=TEXT_PRI,
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=PAD, pady=(HALF, 4))

        self._history_frame = ctk.CTkFrame(card, fg_color="transparent")
        self._history_frame.pack(fill="x", padx=PAD, pady=(0, 4))
        self._history_items: list[ctk.CTkButton] = []

        self._dl_btn = ctk.CTkButton(
            card, text="\u2B07  " + self._t("download"), height=32,
            fg_color=BG_INPUT, hover_color=BORDER, text_color=TEXT_PRI,
            corner_radius=8, command=self._on_save,
        )
        self._dl_btn.pack(fill="x", padx=PAD, pady=(0, HALF))

        # ── Status ───────────────────────────────────────────────────
        self._info_var = ctk.StringVar()
        ctk.CTkLabel(
            parent, textvariable=self._info_var, text_color=TEXT_DIM,
            font=ctk.CTkFont(size=11), anchor="w",
        ).pack(fill="x", padx=PAD, pady=(0, HALF))

    # ── Settings tab ─────────────────────────────────────────────────
    def _build_settings(self, parent):
        opt_kw = dict(
            fg_color=BG_INPUT, button_color=BORDER,
            button_hover_color=ACCENT_DIM,
            dropdown_fg_color=BG_INPUT, dropdown_hover_color=ACCENT,
        )

        self._settings_scroll = ctk.CTkScrollableFrame(parent, fg_color=BG_DARK)
        self._settings_scroll.pack(fill="both", expand=True)
        scroll = self._settings_scroll

        # Language
        card = self._card(scroll)
        self._section_heading(card, self._t("ui_language"))
        self._uilang_var = ctk.StringVar(
            value="Norsk" if self._L == "nb" else "English",
        )
        ctk.CTkOptionMenu(
            card, variable=self._uilang_var, values=["Norsk", "English"],
            command=self._on_uilang, **opt_kw,
        ).pack(fill="x", padx=PAD, pady=(0, PAD))

        # Voice settings
        card = self._card(scroll)
        self._section_heading(card, self._t("voice_settings"))
        self._show_all_var = ctk.BooleanVar(
            value=load_settings().get("show_all_voices", False),
        )
        ctk.CTkCheckBox(
            card, text=self._t("show_all_voices"),
            variable=self._show_all_var, command=self._on_show_all,
            fg_color=ACCENT, hover_color=ACCENT_DIM, text_color=TEXT_PRI,
        ).pack(fill="x", padx=PAD, pady=(0, PAD))

        # Model
        card = self._card(scroll)
        self._section_heading(card, self._t("model_settings"))
        model_names = [f"Kokoro {k}" for k in MODEL_REGISTRY]
        self._model_var = ctk.StringVar(
            value=f"Kokoro {get_model_version()}",
        )
        ctk.CTkOptionMenu(
            card, variable=self._model_var, values=model_names,
            command=self._on_model, **opt_kw,
        ).pack(fill="x", padx=PAD, pady=(0, 4))
        self._model_info = ctk.StringVar()
        ctk.CTkLabel(
            card, textvariable=self._model_info, text_color=TEXT_DIM,
            font=ctk.CTkFont(size=12),
        ).pack(fill="x", padx=PAD, pady=(0, PAD))

        # Updates
        card = self._card(scroll)
        self._section_heading(card, self._t("app_status"))
        self._update_var = ctk.StringVar(value=APP_DISPLAY_VERSION)
        ctk.CTkLabel(
            card, textvariable=self._update_var, text_color=TEXT_SEC,
            wraplength=600,
        ).pack(fill="x", padx=PAD, pady=(0, 8))
        brow = ctk.CTkFrame(card, fg_color="transparent")
        brow.pack(fill="x", padx=PAD, pady=(0, PAD))
        ctk.CTkButton(
            brow, text=self._t("check_update"), width=140, height=34,
            fg_color=BG_INPUT, hover_color=BORDER, text_color=TEXT_PRI,
            corner_radius=8, command=self._on_check_update,
        ).pack(side="left", padx=(0, HALF))
        ctk.CTkButton(
            brow, text=self._t("update_now"), width=140, height=34,
            fg_color=ACCENT, hover_color=ACCENT_DIM, text_color="#fff",
            corner_radius=8, command=self._on_auto_update,
        ).pack(side="left")

    # ── Engine init ──────────────────────────────────────────────────
    def _init_engine(self):
        try:
            _, vl = ensure_engine()
            show_all = self._show_all_var.get()
            lc = self._current_lang_code()
            filtered = voices_for_lang(lc, vl, show_all)
            default = "af_heart" if "af_heart" in filtered else filtered[0]
            self.after(0, self._populate_voices, filtered, default)
            self.after(0, self._set_status, self._t("up_to_date", version=APP_VERSION))
        except Exception as e:
            self.after(0, self._set_status, f"Error: {e}")

    def _populate_voices(self, vlist, default):
        self._voice_menu.configure(values=vlist)
        self._voice_var.set(default)

    def _current_lang_code(self):
        display = self._lang_var.get()
        for name, code in LANGUAGE_CHOICES:
            if name == display:
                return code
        return "en-us"

    # ── Event handlers ───────────────────────────────────────────────
    def _on_lang_change(self, _=None):
        _, vl = ensure_engine()
        filtered = voices_for_lang(
            self._current_lang_code(), vl, self._show_all_var.get(),
        )
        cur = self._voice_var.get()
        d = cur if cur in filtered else (filtered[0] if filtered else "")
        self._voice_menu.configure(values=filtered)
        self._voice_var.set(d)

    def _on_show_all(self):
        s = load_settings()
        s["show_all_voices"] = self._show_all_var.get()
        save_settings(s)
        self._on_lang_change()

    def _on_fav_select(self, val):
        if val and val != "\u2014":
            self._voice_var.set(val)

    def _on_toggle_fav(self):
        v = self._voice_var.get()
        if not v or v == "…":
            return
        favs, status = toggle_favorite(v)
        self._fav_menu.configure(values=favs if favs else ["\u2014"])
        if favs:
            self._fav_var.set(favs[0])
        self._set_status(status)

    def _on_preset(self, display_name):
        pnames = [self._t(f"preset_{k}") for k in self._preset_keys]
        idx = pnames.index(display_name) if display_name in pnames else 0
        style, speed, gain = apply_preset(self._preset_keys[idx])
        self._style_var.set(style)
        self._speed_var.set(speed)
        self._speed_lbl.configure(text=f"{speed:.2f}")
        self._gain_var.set(gain)
        self._gain_lbl.configure(text=f"{gain:+.1f}")

    def _on_generate(self):
        text = self._text.get("1.0", "end").strip()
        if not text:
            self._set_status(self._t("error_empty_text"))
            return
        self._gen_btn.configure(
            state="disabled",
            text="\u23F3  " + self._t("generate") + "\u2026",
        )
        self._set_status("Generating\u2026")

        def _run():
            try:
                path, info = synthesize(
                    text=text,
                    voice=self._voice_var.get(),
                    speed=self._speed_var.get(),
                    lang=self._current_lang_code(),
                    style=self._style_var.get(),
                    gain_db=self._gain_var.get(),
                    output_format=self._format_var.get(),
                )
                from datetime import datetime

                ts = datetime.now().strftime("%H:%M")
                clean = " ".join(text.split())  # collapse newlines/whitespace
                snippet = clean[:30] + ("\u2026" if len(clean) > 30 else "")
                label = f"{snippet}  ({ts})"
                self._history.insert(0, (label, path))
                self._selected_history = 0
                self.after(0, self._refresh_history)
                self.after(0, lambda: self._player.load_file(path))
                self.after(0, self._set_status, info)
                self.after(100, self._player._play)
            except Exception as e:
                self.after(0, self._set_status, str(e))
            finally:
                self.after(
                    0,
                    lambda: self._gen_btn.configure(
                        state="normal", text=self._t("generate"),
                    ),
                )

        threading.Thread(target=_run, daemon=True).start()

    def _refresh_history(self):
        for w in self._history_items:
            w.destroy()
        self._history_items.clear()
        for i, (label, path) in enumerate(self._history[:10]):
            is_sel = i == self._selected_history
            btn = ctk.CTkButton(
                self._history_frame, text=label, anchor="w", height=30,
                fg_color=ACCENT if is_sel else BG_INPUT,
                hover_color=ACCENT_DIM if is_sel else BORDER,
                text_color="#fff" if is_sel else TEXT_SEC,
                corner_radius=6, font=ctk.CTkFont(size=12),
                command=lambda idx=i: self._select_history(idx),
            )
            btn.pack(fill="x", pady=1)
            self._history_items.append(btn)

    def _select_history(self, idx):
        self._selected_history = idx
        _, path = self._history[idx]
        self._player.load_file(path)
        self._player._play()
        self._refresh_history()

    def _on_save(self):
        if not self._history:
            return
        _, path = self._history[self._selected_history]
        src = Path(path)
        if not src.exists():
            return
        dest = filedialog.asksaveasfilename(
            initialfile=src.name,
            defaultextension=src.suffix,
            filetypes=[("Audio", "*.wav *.mp3"), ("All", "*.*")],
        )
        if dest:
            shutil.copy2(src, dest)
            self._set_status(f"Saved \u2192 {Path(dest).name}")

    def _on_uilang(self, val):
        code = "nb" if val == "Norsk" else "en"
        s = load_settings()
        s["ui_language"] = code
        save_settings(s)
        self._set_status(self._t("language_saved"))

    def _on_model(self, val):
        version = val.replace("Kokoro ", "")
        s = load_settings()
        s["model_version"] = version
        save_settings(s)
        self._model_info.set("Loading\u2026")

        def _run():
            try:
                _, nv = ensure_engine(version)
                filtered = voices_for_lang(
                    self._current_lang_code(), nv, self._show_all_var.get(),
                )
                d = filtered[0] if filtered else nv[0]
                self.after(0, self._populate_voices, filtered, d)
                self.after(
                    0,
                    lambda: self._model_info.set(
                        self._t("model_switched", version=version),
                    ),
                )
            except Exception as e:
                self.after(
                    0,
                    lambda: self._model_info.set(
                        self._t("model_switch_error", error=str(e)),
                    ),
                )

        threading.Thread(target=_run, daemon=True).start()

    def _on_check_update(self):
        self._update_var.set("\u2026")

        def _run():
            msg = check_updates_message()
            self.after(0, lambda: self._update_var.set(msg))

        threading.Thread(target=_run, daemon=True).start()

    def _on_auto_update(self):
        self._update_var.set("\u2026")

        def _run():
            msg = auto_update()
            self.after(0, lambda: self._update_var.set(msg))

        threading.Thread(target=_run, daemon=True).start()


def main():
    ctk.set_appearance_mode("dark")
    app = KokoroApp()
    app.mainloop()


if __name__ == "__main__":
    main()
