#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import queue
import threading
import sys
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from PySide6.QtCore import Qt, QTimer, QLibraryInfo, QUrl
from PySide6.QtGui import QAction, QColor, QPainter, QPen, QPixmap, QKeySequence, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QToolButton,
    QSlider,
    QStackedWidget,
    QTabWidget,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QTextEdit,
)

from core import (
    APP_DISPLAY_VERSION,
    APP_VERSION,
    BASE_DIR,
    DEFAULT_MODEL_VERSION,
    DEFAULT_PIPER_MODEL,
    LANGUAGE_CHOICES,
    MODEL_REGISTRY,
    STYLES,
    apply_preset,
    auto_update,
    check_updates_details,
    check_updates_message,
    ensure_engine,
    download_piper_model,
    get_available_piper_models,
    get_model_version,
    get_recent_releases,
    get_piper_model,
    get_tts_engine,
    get_ui_language,
    is_piper_model_downloaded,
    rollback_to_release,
    load_favorites,
    load_settings,
    save_settings,
    synthesize,
    toggle_favorite,
    tr,
    voices_for_lang,
)


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelSlider(QSlider):
    def wheelEvent(self, event) -> None:
        event.ignore()


class ToggleSwitch(QCheckBox):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setText("")
        self.setFixedSize(46, 26)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(1, 1, -1, -1)
        track_color = QColor("#f97316") if self.isChecked() else QColor("#1e1e24")
        border_color = QColor("#c2410c") if self.isChecked() else QColor("#2a2a34")
        knob_color = QColor("#ffffff") if self.isChecked() else QColor("#8f8f9e")

        painter.setPen(QPen(border_color, 1))
        painter.setBrush(track_color)
        painter.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)

        knob_d = rect.height() - 6
        knob_y = rect.y() + (rect.height() - knob_d) / 2
        knob_x = rect.right() - knob_d - 2 if self.isChecked() else rect.x() + 2
        painter.setPen(Qt.NoPen)
        painter.setBrush(knob_color)
        painter.drawEllipse(int(knob_x), int(knob_y), int(knob_d), int(knob_d))

        if self.hasFocus():
            focus_rect = self.rect().adjusted(0, 0, -1, -1)
            painter.setPen(QPen(QColor("#f97316"), 1, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(focus_rect, focus_rect.height() / 2, focus_rect.height() / 2)


class AudioPlayerWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._audio: np.ndarray | None = None
        self._sr: int = 24000
        self._duration: float = 0.0
        self._playing = False
        self._paused = False
        self._play_start: float = 0.0
        self._pause_offset: float = 0.0
        self._updating_slider = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.waveform = WaveformWidget()
        layout.addWidget(self.waveform)

        self.position_slider = NoWheelSlider(Qt.Horizontal)
        self.position_slider.setObjectName("timeline")
        self.position_slider.setRange(0, 1000)
        self.position_slider.setValue(0)
        self.position_slider.sliderPressed.connect(self._on_seek_press)
        self.position_slider.sliderReleased.connect(self._on_seek_release)
        layout.addWidget(self.position_slider)

        row = QHBoxLayout()
        self.time_label = QLabel("0:00 / 0:00")
        row.addWidget(self.time_label)
        row.addStretch(1)

        self.rewind_btn = QPushButton("⏮")
        self.rewind_btn.setFixedWidth(72)
        self.rewind_btn.clicked.connect(self._rewind)
        row.addWidget(self.rewind_btn)

        self.play_btn = QPushButton("▶")
        self.play_btn.setObjectName("primary")
        self.play_btn.setFixedWidth(88)
        self.play_btn.clicked.connect(self._toggle_play)
        row.addWidget(self.play_btn)

        self.forward_btn = QPushButton("⏭")
        self.forward_btn.setFixedWidth(72)
        self.forward_btn.clicked.connect(self._forward)
        row.addWidget(self.forward_btn)

        layout.addLayout(row)

        self._timer = self.startTimer(80)

    def _pick_output_device(self) -> int | None:
        try:
            default_dev = sd.default.device
            if isinstance(default_dev, (tuple, list)) and len(default_dev) >= 2:
                out_dev = default_dev[1]
                if isinstance(out_dev, int) and out_dev >= 0:
                    info = sd.query_devices(out_dev)
                    if info and int(info.get("max_output_channels", 0)) > 0:
                        return out_dev
        except Exception:
            pass

        try:
            for idx, info in enumerate(sd.query_devices()):
                if int(info.get("max_output_channels", 0)) > 0:
                    return idx
        except Exception:
            return None
        return None

    def load_file(self, path: str) -> None:
        self.stop()
        try:
            data, sr = sf.read(path)
        except Exception:
            self._audio = None
            self._duration = 0.0
            self._update_time(0.0)
            return

        if isinstance(data, np.ndarray) and data.ndim > 1:
            data = data.mean(axis=1)

        self._audio = np.ascontiguousarray(data, dtype=np.float32)
        self._sr = int(sr)
        self._duration = float(len(self._audio) / self._sr) if self._sr > 0 else 0.0
        self.position_slider.setValue(0)
        self.waveform.set_audio(self._audio)
        self.waveform.set_position(0.0, self._duration)
        self._update_time(0.0)

    def stop(self) -> None:
        self._playing = False
        self._paused = False
        self._pause_offset = 0.0
        sd.stop()
        self.play_btn.setText("▶")

    def _toggle_play(self) -> None:
        if self._playing and not self._paused:
            self._pause()
        elif self._paused:
            self._resume()
        else:
            self._play(self._pause_offset)

    def _play(self, offset: float = 0.0) -> None:
        if self._audio is None or self._duration <= 0:
            return
        self.stop()
        self._playing = True
        self._paused = False
        self._pause_offset = max(0.0, min(offset, max(0.0, self._duration - 0.05)))

        start_frame = int(self._pause_offset * self._sr)
        out_device = self._pick_output_device()
        play_kwargs = {
            "blocking": False,
            "latency": "high",
            "blocksize": 2048,
        }
        if out_device is not None:
            play_kwargs["device"] = out_device

        try:
            sd.play(self._audio[start_frame:], self._sr, **play_kwargs)
        except Exception:
            return
        self._play_start = time.time() - self._pause_offset
        self.play_btn.setText("⏸")

    def _pause(self) -> None:
        if not self._playing:
            return
        self._paused = True
        self._pause_offset = self._current_position()
        sd.stop()
        self.play_btn.setText("▶")

    def _resume(self) -> None:
        self._play(self._pause_offset)

    def _rewind(self) -> None:
        if self._audio is None:
            return
        target = max(0.0, self._current_position() - 5.0)
        if self._playing or self._paused:
            self._play(target)
        else:
            self._pause_offset = target
            self._set_slider_from_pos(target)
            self._update_time(target)

    def _forward(self) -> None:
        if self._audio is None:
            return
        target = min(self._duration, self._current_position() + 5.0)
        if self._playing or self._paused:
            self._play(target)
        else:
            self._pause_offset = target
            self._set_slider_from_pos(target)
            self._update_time(target)

    def _on_seek_press(self) -> None:
        if self._playing and not self._paused:
            self._pause()

    def _on_seek_release(self) -> None:
        if self._duration <= 0:
            return
        frac = self.position_slider.value() / 1000.0
        target = frac * self._duration
        self._pause_offset = target
        self.waveform.set_position(target, self._duration)
        self._update_time(target)

    def _current_position(self) -> float:
        if self._playing and not self._paused:
            return max(0.0, min(self._duration, time.time() - self._play_start))
        return max(0.0, min(self._duration, self._pause_offset))

    def _set_slider_from_pos(self, pos: float) -> None:
        if self._duration <= 0:
            self.position_slider.setValue(0)
            return
        self._updating_slider = True
        self.position_slider.setValue(int((pos / self._duration) * 1000))
        self._updating_slider = False
        self.waveform.set_position(pos, self._duration)

    def _fmt(self, sec: float) -> str:
        s = max(0, int(sec))
        m, r = divmod(s, 60)
        return f"{m}:{r:02d}"

    def _update_time(self, pos: float) -> None:
        self.time_label.setText(f"{self._fmt(pos)} / {self._fmt(self._duration)}")

    def timerEvent(self, event) -> None:
        if event.timerId() != self._timer:
            return
        if self._duration <= 0 or self._updating_slider:
            return
        pos = self._current_position()
        if self._playing and not self._paused and pos >= self._duration:
            self.stop()
            self._set_slider_from_pos(self._duration)
            self._update_time(self._duration)
            return
        self._set_slider_from_pos(pos)
        self._update_time(pos)


class WaveformWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(72)
        self._peaks: list[float] = []
        self._progress: float = 0.0

    def set_audio(self, audio: np.ndarray | None) -> None:
        if audio is None or len(audio) == 0:
            self._peaks = []
            self.update()
            return
        bars = 220
        chunk = max(1, len(audio) // bars)
        peaks = []
        for i in range(0, len(audio), chunk):
            segment = audio[i : i + chunk]
            peaks.append(float(np.max(np.abs(segment))) if len(segment) else 0.0)
        m = max(peaks) if peaks else 1.0
        self._peaks = [p / m if m > 0 else 0.0 for p in peaks]
        self._progress = 0.0
        self.update()

    def set_position(self, pos: float, duration: float) -> None:
        self._progress = max(0.0, min(1.0, (pos / duration) if duration > 0 else 0.0))
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, QColor("#111114"))

        if not self._peaks:
            painter.setPen(QPen(QColor("#5c5c6a"), 2))
            y_mid = rect.center().y()
            for x in range(4, rect.width(), 6):
                painter.drawLine(x, y_mid - 3, x, y_mid + 3)
            return

        n = len(self._peaks)
        if n <= 0:
            return
        w = rect.width()
        h = rect.height()
        mid = h / 2
        bar_step = max(1.0, w / n)
        progress_x = w * self._progress

        played_pen = QPen(QColor("#f97316"), max(1.2, bar_step * 0.7), Qt.SolidLine, Qt.RoundCap)
        rest_pen = QPen(QColor("#2a2a34"), max(1.2, bar_step * 0.7), Qt.SolidLine, Qt.RoundCap)

        for i, peak in enumerate(self._peaks):
            x = i * bar_step
            amp = max(2.0, peak * (mid - 4))
            painter.setPen(played_pen if x <= progress_x else rest_pen)
            painter.drawLine(int(x), int(mid - amp), int(x), int(mid + amp))


class KokoroQtWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._L = get_ui_language()
        self.setWindowTitle(f"Kokoro TTS {APP_DISPLAY_VERSION}")
        self.resize(900, 840)
        self.setMinimumSize(720, 640)

        self._voices: list[str] = []
        self._history: list[tuple[str, str]] = []
        self._selected_history: int = 0
        self._lang_display_to_code = {name: code for name, code in LANGUAGE_CHOICES}
        self._preset_keys = ["neutral", "alert", "narration", "direct"]

        self._status_text = self._t("loading_model")
        self._status_labels: list[QLabel] = []

        self._build_ui()
        self._setup_app_menu()
        self._configure_tab_navigation()
        self._apply_styles()
        self._init_engine()

    def _t(self, key: str, **kw) -> str:
        return tr(key, self._L, **kw)

    def _setup_app_menu(self) -> None:
        app_menu = self.menuBar().addMenu(self._t("app_menu_title"))

        self.settings_action = QAction(self._t("app_menu_settings"), self)
        self.settings_action.setShortcut(QKeySequence("Ctrl+,"))
        self.settings_action.setMenuRole(QAction.PreferencesRole)
        self.settings_action.triggered.connect(self._show_settings_page)
        app_menu.addAction(self.settings_action)

        about_menu = self.menuBar().addMenu(self._t("about_menu_title"))

        self.about_action = QAction(self._t("app_menu_about"), self)
        self.about_action.setMenuRole(QAction.NoRole)
        self.about_action.triggered.connect(self._show_about_dialog)
        about_menu.addAction(self.about_action)

        self.github_action = QAction(self._t("github_menu_link"), self)
        self.github_action.setMenuRole(QAction.NoRole)
        self.github_action.triggered.connect(self._open_github)
        about_menu.addAction(self.github_action)

    def _open_github(self) -> None:
        QDesktopServices.openUrl(QUrl("https://github.com/owanvik/kokoro-tts-mac-app"))

    def _show_about_dialog(self) -> None:
        QMessageBox.about(
            self,
            self._t("app_menu_about"),
            self._t("about_dialog_text", version=APP_DISPLAY_VERSION),
        )

    def _set_status(self, msg: str) -> None:
        self._status_text = msg
        for label in self._status_labels:
            label.setText(msg)

    def _card(self, title: str) -> tuple[QWidget, QVBoxLayout]:
        card = QWidget()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 10, 12, 10)
        card_layout.setSpacing(8)
        label = QLabel(title)
        label.setObjectName("sectionTitle")
        card_layout.addWidget(label)
        return card, card_layout

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 10, 0, 12)
        root_layout.setSpacing(8)

        root_layout.addWidget(self._build_top_section())

        self.pages = QStackedWidget()
        self.main_page = self._build_generate_tab()
        self.settings_page = self._build_settings_page()
        self.pages.addWidget(self.main_page)
        self.pages.addWidget(self.settings_page)
        root_layout.addWidget(self.pages, 1)

        status = QLabel(self._status_text)
        status.setWordWrap(True)
        status.setObjectName("status")
        self._status_labels.append(status)
        root_layout.addWidget(status)

        self.setCentralWidget(root)

    def _build_settings_page(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 10, 0)
        content_layout.setSpacing(8)

        self._build_settings_content(content_layout)

        self.back_to_main_btn = QPushButton("← " + self._t("back"))
        self.back_to_main_btn.setMinimumHeight(34)
        self.back_to_main_btn.clicked.connect(self._show_main_page)
        content_layout.addWidget(self.back_to_main_btn)

        content_layout.addStretch(1)
        scroll.setWidget(content)
        page_layout.addWidget(scroll)
        return page

    def _build_top_section(self) -> QWidget:
        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(0)

        header_row = QHBoxLayout()
        logo = self._build_logo_label()
        if logo is not None:
            header_row.addWidget(logo)
        title = QLabel("Kokoro TTS")
        title.setObjectName("appTitle")
        version = QLabel(APP_DISPLAY_VERSION)
        version.setObjectName("appVersion")
        header_row.addWidget(title)
        header_row.addWidget(version)
        header_row.addStretch(1)
        wrapper_layout.addLayout(header_row)
        return wrapper

    def _build_logo_label(self) -> QLabel | None:
        candidates = [
            BASE_DIR / "kokorotts.png",
            BASE_DIR / "icons" / "kokorotts.png",
        ]
        for path in candidates:
            if not path.exists():
                continue
            pixmap = QPixmap(str(path))
            if pixmap.isNull():
                continue
            logo = QLabel()
            logo.setPixmap(
                pixmap.scaled(36, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation),
            )
            logo.setFixedSize(38, 38)
            return logo
        return None

    def _build_generate_tab(self) -> QWidget:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        self._generate_scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 10, 0)
        content_layout.setSpacing(8)

        text_card, text_layout = self._card(self._t("text"))
        self.text_input = QTextEdit()
        self.text_input.setObjectName("inputField")
        self._text_inner_padding = 6
        self.text_input.document().setDocumentMargin(0)
        self.text_input.setViewportMargins(
            self._text_inner_padding,
            self._text_inner_padding,
            self._text_inner_padding,
            self._text_inner_padding,
        )
        self._text_min_height = 150
        self._text_current_height = self._text_min_height
        self.text_input.setMinimumHeight(self._text_min_height)
        self.text_input.setTabChangesFocus(True)
        self.text_input.document().documentLayout().documentSizeChanged.connect(self._on_text_document_size_changed)
        self.text_input.viewport().setStyleSheet("background-color: #0c0c0e;")
        text_layout.addWidget(self.text_input)
        self._resize_text_input()
        content_layout.addWidget(text_card)

        voice_card, voice_layout = self._card(self._t("voice_group"))
        voice_grid = QGridLayout()
        voice_grid.setHorizontalSpacing(8)
        voice_grid.setVerticalSpacing(6)
        voice_grid.setColumnStretch(0, 3)
        voice_grid.setColumnStretch(1, 2)

        self.voice_combo = NoWheelComboBox()
        self.voice_combo.setObjectName("inputField")
        self.voice_combo.addItem("…")
        self.fav_combo = NoWheelComboBox()
        self.fav_combo.setObjectName("inputField")
        favs = load_favorites()
        self.fav_combo.addItems(favs if favs else ["—"])
        self.fav_combo.currentTextChanged.connect(self._on_fav_select)
        self.fav_btn = QPushButton("⭐")
        self.fav_btn.setFixedWidth(48)
        self.fav_btn.clicked.connect(self._on_toggle_fav)

        voice_grid.addWidget(QLabel(self._t("voice")), 0, 0)
        voice_grid.addWidget(QLabel(self._t("favorites")), 0, 1)
        voice_grid.addWidget(self.voice_combo, 1, 0)
        voice_grid.addWidget(self.fav_combo, 1, 1)
        voice_grid.addWidget(self.fav_btn, 1, 2)
        voice_layout.addLayout(voice_grid)
        content_layout.addWidget(voice_card)

        audio_card = QWidget()
        audio_card.setObjectName("card")
        audio_layout = QVBoxLayout(audio_card)
        audio_layout.setContentsMargins(12, 10, 12, 10)
        audio_layout.setSpacing(8)

        self.audio_toggle_btn = QToolButton()
        self.audio_toggle_btn.setObjectName("sectionToggle")
        self.audio_toggle_btn.setText(self._t("audio_settings"))
        self.audio_toggle_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.audio_toggle_btn.setArrowType(Qt.RightArrow)
        self.audio_toggle_btn.setCheckable(True)
        audio_layout.addWidget(self.audio_toggle_btn)

        self.audio_settings_body = QWidget()
        audio_body_layout = QVBoxLayout(self.audio_settings_body)
        audio_body_layout.setContentsMargins(0, 0, 0, 0)
        audio_body_layout.setSpacing(0)

        audio_grid = QGridLayout()
        audio_grid.setHorizontalSpacing(8)
        audio_grid.setVerticalSpacing(6)

        self.lang_combo = NoWheelComboBox()
        self.lang_combo.setObjectName("inputField")
        for name, code in LANGUAGE_CHOICES:
            self.lang_combo.addItem(name, code)
        self.lang_combo.currentTextChanged.connect(self._on_lang_change)

        self.style_combo = NoWheelComboBox()
        self.style_combo.setObjectName("inputField")
        self.style_combo.addItems(STYLES)

        self.preset_combo = NoWheelComboBox()
        self.preset_combo.setObjectName("inputField")
        self._refresh_preset_names()
        self.preset_combo.currentTextChanged.connect(self._on_preset)

        self.speed_spin = NoWheelDoubleSpinBox()
        self.speed_spin.setObjectName("inputField")
        self.speed_spin.setRange(0.5, 2.0)
        self.speed_spin.setSingleStep(0.05)
        self.speed_spin.setValue(1.0)

        self.gain_spin = NoWheelDoubleSpinBox()
        self.gain_spin.setObjectName("inputField")
        self.gain_spin.setRange(-12.0, 12.0)
        self.gain_spin.setSingleStep(0.5)
        self.gain_spin.setValue(0.0)

        self.format_combo = NoWheelComboBox()
        self.format_combo.setObjectName("inputField")
        self.format_combo.addItems(["wav", "mp3"])

        self.bitrate_combo = NoWheelComboBox()
        self.bitrate_combo.setObjectName("inputField")
        self.bitrate_combo.addItems(["96", "128", "160", "192", "256", "320"])
        self.bitrate_combo.setCurrentText("192")

        audio_grid.addWidget(QLabel(self._t("language_code")), 0, 0)
        audio_grid.addWidget(QLabel(self._t("style")), 0, 1)
        audio_grid.addWidget(QLabel(self._t("preset")), 0, 2)
        audio_grid.addWidget(self.lang_combo, 1, 0)
        audio_grid.addWidget(self.style_combo, 1, 1)
        audio_grid.addWidget(self.preset_combo, 1, 2)

        audio_grid.addWidget(QLabel(self._t("base_speed")), 2, 0)
        audio_grid.addWidget(QLabel(self._t("volume_db")), 2, 1)
        audio_grid.addWidget(QLabel(self._t("format")), 2, 2)
        audio_grid.addWidget(self.speed_spin, 3, 0)
        audio_grid.addWidget(self.gain_spin, 3, 1)
        audio_grid.addWidget(self.format_combo, 3, 2)
        audio_grid.addWidget(QLabel(self._t("bitrate")), 4, 2)
        audio_grid.addWidget(self.bitrate_combo, 5, 2)

        settings = load_settings()
        selected_format = str(settings.get("output_format", "wav")).strip().lower()
        if selected_format not in {"wav", "mp3"}:
            selected_format = "wav"
        self.format_combo.setCurrentText(selected_format)
        selected_bitrate = str(settings.get("mp3_bitrate_kbps", "192")).strip()
        if selected_bitrate in {"96", "128", "160", "192", "256", "320"}:
            self.bitrate_combo.setCurrentText(selected_bitrate)

        self.format_combo.currentTextChanged.connect(self._on_audio_setting_change)
        self.bitrate_combo.currentTextChanged.connect(self._on_audio_setting_change)
        self._update_mp3_bitrate_state()

        audio_body_layout.addLayout(audio_grid)

        audio_actions = QHBoxLayout()
        audio_actions.addStretch(1)
        self.restore_audio_defaults_btn = QPushButton("↺")
        self.restore_audio_defaults_btn.setObjectName("secondary")
        self.restore_audio_defaults_btn.setToolTip(self._t("restore_audio_defaults"))
        self.restore_audio_defaults_btn.setMinimumHeight(30)
        self.restore_audio_defaults_btn.setFixedWidth(38)
        self.restore_audio_defaults_btn.clicked.connect(self._on_restore_audio_defaults)
        audio_actions.addWidget(self.restore_audio_defaults_btn)
        audio_body_layout.addLayout(audio_actions)

        audio_open = bool(settings.get("audio_settings_open", False))
        self.audio_settings_body.setVisible(audio_open)
        self.audio_toggle_btn.setChecked(audio_open)
        self.audio_toggle_btn.setArrowType(Qt.DownArrow if audio_open else Qt.RightArrow)
        audio_layout.addWidget(self.audio_settings_body)

        def _toggle_audio_settings(opened: bool) -> None:
            self.audio_settings_body.setVisible(opened)
            self.audio_toggle_btn.setArrowType(Qt.DownArrow if opened else Qt.RightArrow)
            cfg = load_settings()
            cfg["audio_settings_open"] = bool(opened)
            save_settings(cfg)

        self.audio_toggle_btn.toggled.connect(_toggle_audio_settings)
        content_layout.addWidget(audio_card)

        self.generate_btn = QPushButton(self._t("generate"))
        self.generate_btn.setObjectName("primary")
        self.generate_btn.setMinimumHeight(40)
        self.generate_btn.clicked.connect(self._on_generate)

        self.generate_progress = QProgressBar()
        self.generate_progress.setObjectName("generateProgress")
        self.generate_progress.setRange(0, 0)
        self.generate_progress.setTextVisible(False)
        self.generate_progress.setMinimumHeight(40)

        self.generate_action_stack = QStackedWidget()
        self.generate_action_stack.addWidget(self.generate_btn)
        self.generate_action_stack.addWidget(self.generate_progress)
        self.generate_action_stack.setCurrentWidget(self.generate_btn)
        content_layout.addWidget(self.generate_action_stack)

        player_card, player_layout = self._card("Player")
        self.player = AudioPlayerWidget()
        player_layout.addWidget(self.player)
        content_layout.addWidget(player_card)

        history_card, history_layout = self._card(self._t("history"))
        self.history_list = QListWidget()
        self.history_list.setObjectName("inputField")
        self.history_list.viewport().setStyleSheet("background-color: #0c0c0e;")
        self.history_list.itemClicked.connect(self._on_history_select)
        self.history_list.setMinimumHeight(170)
        history_layout.addWidget(self.history_list)

        self.save_btn = QPushButton("⬇  " + self._t("download"))
        self.save_btn.setMinimumHeight(34)
        self.save_btn.clicked.connect(self._on_save)
        history_layout.addWidget(self.save_btn)
        content_layout.addWidget(history_card)

        self.open_settings_btn = QPushButton("⚙  " + self._t("tab_settings"))
        self.open_settings_btn.setMinimumHeight(36)
        self.open_settings_btn.clicked.connect(self._show_settings_page)
        content_layout.addWidget(self.open_settings_btn)

        content_layout.addStretch(1)
        scroll.setWidget(content)
        tab_layout.addWidget(scroll)
        return tab

    def _build_settings_content(self, content_layout: QVBoxLayout) -> None:
        lang_card, lang_layout = self._card(self._t("ui_language"))
        self.ui_lang_combo = NoWheelComboBox()
        self.ui_lang_combo.setObjectName("inputField")
        self.ui_lang_combo.addItems(["Norsk", "English"])
        self.ui_lang_combo.setCurrentText("Norsk" if self._L == "nb" else "English")
        self.ui_lang_combo.currentTextChanged.connect(self._on_uilang)
        lang_layout.addWidget(self.ui_lang_combo)

        self.restore_defaults_btn = QPushButton("↺")
        self.restore_defaults_btn.setObjectName("secondary")
        self.restore_defaults_btn.setToolTip(self._t("restore_defaults"))
        self.restore_defaults_btn.setMinimumHeight(34)
        self.restore_defaults_btn.setFixedWidth(42)
        self.restore_defaults_btn.clicked.connect(self._on_restore_defaults)
        lang_layout.addWidget(self.restore_defaults_btn)
        content_layout.addWidget(lang_card)

        voice_card, voice_layout = self._card(self._t("voice_settings"))
        show_all_row = QHBoxLayout()
        show_all_row.setContentsMargins(0, 0, 0, 0)
        show_all_row.setSpacing(8)
        show_all_label = QLabel(self._t("show_all_voices"))
        show_all_label.setWordWrap(True)
        self.show_all_chk = ToggleSwitch()
        self.show_all_chk.setChecked(bool(load_settings().get("show_all_voices", False)))
        self.show_all_chk.stateChanged.connect(self._on_show_all)
        show_all_row.addWidget(show_all_label, 1)
        show_all_row.addWidget(self.show_all_chk, 0, Qt.AlignRight)
        voice_layout.addLayout(show_all_row)
        content_layout.addWidget(voice_card)

        model_card, model_layout = self._card(self._t("model_settings"))

        self.engine_combo = NoWheelComboBox()
        self.engine_combo.setObjectName("inputField")
        self.engine_combo.addItem(self._t("engine_kokoro"), "kokoro")
        self.engine_combo.addItem(self._t("engine_piper"), "piper")
        self.engine_combo.setCurrentIndex(0 if get_tts_engine() == "kokoro" else 1)
        self.engine_combo.currentIndexChanged.connect(self._on_tts_engine_change)

        model_layout.addWidget(QLabel(self._t("tts_engine")))
        model_layout.addWidget(self.engine_combo)

        self.model_combo = NoWheelComboBox()
        self.model_combo.setObjectName("inputField")
        self.model_combo.addItems([f"Kokoro {k}" for k in MODEL_REGISTRY])
        self.model_combo.setCurrentText(f"Kokoro {get_model_version()}")
        self.model_combo.currentTextChanged.connect(self._on_model)
        self.model_info = QLabel("")
        self.model_info.setWordWrap(True)
        model_layout.addWidget(QLabel(self._t("model_version")))
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.model_info)

        self.piper_download_btn = QPushButton(self._t("piper_download_model"))
        self.piper_download_btn.setMinimumHeight(34)
        self.piper_download_btn.clicked.connect(self._on_download_piper_model)
        model_layout.addWidget(self.piper_download_btn)

        self.piper_progress = QProgressBar()
        self.piper_progress.setRange(0, 100)
        self.piper_progress.setValue(0)
        self.piper_progress.setVisible(False)
        model_layout.addWidget(self.piper_progress)

        self.piper_progress_label = QLabel("")
        self.piper_progress_label.setVisible(False)
        model_layout.addWidget(self.piper_progress_label)

        content_layout.addWidget(model_card)

        self._refresh_engine_ui_state()

        update_card, update_layout = self._card(self._t("app_status"))
        self.update_label = QLabel(APP_DISPLAY_VERSION)
        self.update_label.setWordWrap(True)
        update_layout.addWidget(self.update_label)

        self.release_notes_title = QLabel(self._t("release_notes_section"))
        self.release_notes_title.setObjectName("sectionTitle")
        self.release_notes_title.setVisible(False)
        update_layout.addWidget(self.release_notes_title)

        self.release_notes_view = QTextEdit()
        self.release_notes_view.setReadOnly(True)
        self.release_notes_view.setObjectName("inputField")
        self.release_notes_view.setMinimumHeight(140)
        self.release_notes_view.setVisible(False)
        update_layout.addWidget(self.release_notes_view)

        update_row = QHBoxLayout()
        self.check_update_btn = QPushButton(self._t("check_update"))
        self.check_update_btn.setMinimumHeight(34)
        self.check_update_btn.clicked.connect(self._on_check_update)
        self.auto_update_btn = QPushButton(self._t("update_now"))
        self.auto_update_btn.setObjectName("primary")
        self.auto_update_btn.setMinimumHeight(34)
        self.auto_update_btn.clicked.connect(self._on_auto_update)
        update_row.addWidget(self.check_update_btn)
        update_row.addWidget(self.auto_update_btn)
        update_row.addStretch(1)
        update_layout.addLayout(update_row)

        rollback_row = QHBoxLayout()
        self.rollback_combo = NoWheelComboBox()
        self.rollback_combo.setObjectName("inputField")
        self.rollback_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.rollback_btn = QPushButton(self._t("rollback_now"))
        self.rollback_btn.setMinimumHeight(34)
        self.rollback_btn.clicked.connect(self._on_rollback)
        rollback_row.addWidget(self.rollback_combo)
        rollback_row.addWidget(self.rollback_btn)
        update_layout.addLayout(rollback_row)

        self._rollback_releases: list[dict[str, str]] = []
        self._load_rollback_releases()
        content_layout.addWidget(update_card)

    def _configure_tab_navigation(self) -> None:
        self.voice_combo.setFocusPolicy(Qt.StrongFocus)
        self.fav_combo.setFocusPolicy(Qt.StrongFocus)
        self.lang_combo.setFocusPolicy(Qt.StrongFocus)
        self.style_combo.setFocusPolicy(Qt.StrongFocus)
        self.preset_combo.setFocusPolicy(Qt.StrongFocus)

        QWidget.setTabOrder(self.text_input, self.voice_combo)
        QWidget.setTabOrder(self.voice_combo, self.fav_combo)
        QWidget.setTabOrder(self.fav_combo, self.fav_btn)
        QWidget.setTabOrder(self.fav_btn, self.audio_toggle_btn)
        QWidget.setTabOrder(self.audio_toggle_btn, self.lang_combo)
        QWidget.setTabOrder(self.lang_combo, self.style_combo)
        QWidget.setTabOrder(self.style_combo, self.preset_combo)
        QWidget.setTabOrder(self.preset_combo, self.speed_spin)
        QWidget.setTabOrder(self.speed_spin, self.gain_spin)
        QWidget.setTabOrder(self.gain_spin, self.format_combo)
        QWidget.setTabOrder(self.format_combo, self.bitrate_combo)
        QWidget.setTabOrder(self.bitrate_combo, self.restore_audio_defaults_btn)
        QWidget.setTabOrder(self.restore_audio_defaults_btn, self.generate_btn)
        QWidget.setTabOrder(self.generate_btn, self.player.position_slider)
        QWidget.setTabOrder(self.player.position_slider, self.player.rewind_btn)
        QWidget.setTabOrder(self.player.rewind_btn, self.player.play_btn)
        QWidget.setTabOrder(self.player.play_btn, self.player.forward_btn)
        QWidget.setTabOrder(self.player.forward_btn, self.history_list)
        QWidget.setTabOrder(self.history_list, self.save_btn)
        QWidget.setTabOrder(self.save_btn, self.open_settings_btn)
        QWidget.setTabOrder(self.open_settings_btn, self.text_input)

        QWidget.setTabOrder(self.ui_lang_combo, self.show_all_chk)
        QWidget.setTabOrder(self.show_all_chk, self.engine_combo)
        QWidget.setTabOrder(self.engine_combo, self.model_combo)
        QWidget.setTabOrder(self.model_combo, self.check_update_btn)
        QWidget.setTabOrder(self.check_update_btn, self.auto_update_btn)
        QWidget.setTabOrder(self.auto_update_btn, self.rollback_combo)
        QWidget.setTabOrder(self.rollback_combo, self.rollback_btn)
        QWidget.setTabOrder(self.rollback_btn, self.restore_defaults_btn)
        QWidget.setTabOrder(self.restore_defaults_btn, self.back_to_main_btn)
        QWidget.setTabOrder(self.back_to_main_btn, self.ui_lang_combo)

    def _build_settings_tab(self) -> QWidget:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        self._settings_scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)

        content_layout.addWidget(self._build_top_section())

        lang_card, lang_layout = self._card(self._t("ui_language"))
        self.ui_lang_combo = NoWheelComboBox()
        self.ui_lang_combo.setObjectName("inputField")
        self.ui_lang_combo.addItems(["Norsk", "English"])
        self.ui_lang_combo.setCurrentText("Norsk" if self._L == "nb" else "English")
        self.ui_lang_combo.currentTextChanged.connect(self._on_uilang)
        lang_layout.addWidget(self.ui_lang_combo)

        self.restore_defaults_btn = QPushButton("↺")
        self.restore_defaults_btn.setObjectName("secondary")
        self.restore_defaults_btn.setToolTip(self._t("restore_defaults"))
        self.restore_defaults_btn.setMinimumHeight(34)
        self.restore_defaults_btn.setFixedWidth(42)
        self.restore_defaults_btn.clicked.connect(self._on_restore_defaults)
        lang_layout.addWidget(self.restore_defaults_btn)
        content_layout.addWidget(lang_card)

        voice_card, voice_layout = self._card(self._t("voice_settings"))
        self.show_all_chk = QCheckBox(self._t("show_all_voices"))
        self.show_all_chk.setChecked(bool(load_settings().get("show_all_voices", False)))
        self.show_all_chk.stateChanged.connect(self._on_show_all)
        voice_layout.addWidget(self.show_all_chk)
        content_layout.addWidget(voice_card)

        model_card, model_layout = self._card(self._t("model_settings"))
        self.model_combo = NoWheelComboBox()
        self.model_combo.setObjectName("inputField")
        self.model_combo.addItems([f"Kokoro {k}" for k in MODEL_REGISTRY])
        self.model_combo.setCurrentText(f"Kokoro {get_model_version()}")
        self.model_combo.currentTextChanged.connect(self._on_model)
        self.model_info = QLabel("")
        self.model_info.setWordWrap(True)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.model_info)
        content_layout.addWidget(model_card)

        update_card, update_layout = self._card(self._t("app_status"))
        self.update_label = QLabel(APP_DISPLAY_VERSION)
        self.update_label.setWordWrap(True)
        update_layout.addWidget(self.update_label)
        update_row = QHBoxLayout()
        self.check_update_btn = QPushButton(self._t("check_update"))
        self.check_update_btn.setMinimumHeight(34)
        self.check_update_btn.clicked.connect(self._on_check_update)
        self.auto_update_btn = QPushButton(self._t("update_now"))
        self.auto_update_btn.setObjectName("primary")
        self.auto_update_btn.setMinimumHeight(34)
        self.auto_update_btn.clicked.connect(self._on_auto_update)
        update_row.addWidget(self.check_update_btn)
        update_row.addWidget(self.auto_update_btn)
        update_row.addStretch(1)
        update_layout.addLayout(update_row)
        content_layout.addWidget(update_card)

        content_layout.addStretch(1)
        scroll.setWidget(content)
        tab_layout.addWidget(scroll)
        return tab

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #0c0c0e; color: #ebebf0; }

            QTabWidget { background: transparent; }
            #card {
                background: #0c0c0e;
                border: 1px solid #252533;
                border-radius: 14px;
            }
            #appTitle {
                color: #ebebf0;
                font-size: 28px;
                font-weight: 750;
                letter-spacing: 0.2px;
            }
            #appVersion {
                color: #7a7a88;
                font-size: 13px;
                padding-top: 2px;
            }
            #sectionTitle {
                color: #f4f4f8;
                font-size: 14px;
                font-weight: 700;
                padding-bottom: 2px;
            }
            QToolButton#sectionToggle {
                color: #f4f4f8;
                font-size: 14px;
                font-weight: 700;
                border: none;
                padding: 2px 0;
                text-align: left;
            }
            QToolButton#sectionToggle::menu-indicator { image: none; }

            QCheckBox#toggleSwitch {
                spacing: 0;
                min-width: 44px;
                min-height: 24px;
                max-width: 44px;
                max-height: 24px;
            }
            QCheckBox#toggleSwitch::indicator {
                width: 44px;
                height: 24px;
                border-radius: 12px;
                border: 1px solid #2a2a34;
                background: #1e1e24;
            }
            QCheckBox#toggleSwitch::indicator:checked {
                background: #f97316;
                border: 1px solid #c2410c;
            }

            #status {
                color: #5c5c6a;
                background: transparent;
                border: none;
                padding: 6px 4px;
                font-size: 12px;
            }

            #inputField {
                background: #0c0c0e;
                border: 1px solid #2a2a34;
                border-radius: 10px;
                padding: 7px 10px;
                min-height: 34px;
                selection-background-color: #f97316;
                selection-color: #ffffff;
            }
            #inputField:focus {
                background-color: #0c0c0e;
                border: 1px solid #f97316;
            }

            QTextEdit#inputField {
                padding: 0;
            }

            QComboBox QAbstractItemView, QListView, QAbstractItemView {
                background: #0c0c0e;
                color: #ebebf0;
                border: 1px solid #2a2a34;
                border-radius: 8px;
                padding: 4px;
                selection-background-color: #f97316;
                selection-color: #ffffff;
            }

            QAbstractSpinBox::up-button,
            QAbstractSpinBox::down-button,
            QComboBox::drop-down {
                background: #0c0c0e;
                border: none;
                width: 18px;
            }
            QPushButton {
                background: #1e1e24;
                color: #ebebf0;
                border: none;
                border-radius: 10px;
                padding: 8px 14px;
                min-height: 32px;
                font-weight: 600;
            }
            QPushButton:hover { background: #2a2a34; }
            QPushButton:pressed { background: #23232c; }
            QPushButton#primary { background: #f97316; color: white; font-weight: 700; }
            QPushButton#primary:hover { background: #c2410c; }
            QPushButton#primary:pressed { background: #9a3412; }
            QPushButton#secondary {
                background: #1e1e24;
                border: 1px solid #2a2a34;
                padding: 0;
                font-size: 18px;
                font-weight: 700;
            }
            QPushButton#secondary:hover { border: 1px solid #f97316; }

            QProgressBar#generateProgress {
                background: #1e1e24;
                border: 1px solid #2a2a34;
                border-radius: 10px;
                padding: 2px;
            }
            QProgressBar#generateProgress::chunk {
                background: #f97316;
                border-radius: 8px;
                margin: 0px;
            }

            QTabWidget::pane { border: none; }
            QTabBar::tab {
                background: #1e1e24;
                color: #ebebf0;
                padding: 8px 16px;
                border-radius: 10px;
                margin-right: 6px;
                min-width: 90px;
                font-weight: 600;
            }
            QTabBar::tab:selected { background: #f97316; color: white; }
            QTabBar::tab:hover:!selected { background: #272731; }

            QListWidget::item:selected {
                background: #f97316;
                color: #ffffff;
                border-radius: 8px;
            }
            QListWidget::item:hover {
                background: #c2410c;
                color: #ffffff;
                border-radius: 8px;
            }

            QListWidget::item {
                padding: 7px 8px;
                margin: 2px 0;
            }

            QScrollArea {
                border: none;
                margin: 0;
                padding: 0;
            }

            QScrollBar:vertical {
                background: #121218;
                width: 6px;
                margin: 3px 0px 3px 0px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background: #2a2a34;
                min-height: 24px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical:hover {
                background: #3b3b49;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0;
            }

            QSlider#timeline::groove:horizontal {
                background: #2a2a34;
                height: 6px;
                border-radius: 3px;
            }
            QSlider#timeline::sub-page:horizontal {
                background: #f97316;
                border-radius: 3px;
            }
            QSlider#timeline::add-page:horizontal {
                background: #2a2a34;
                border-radius: 3px;
            }
            QSlider#timeline::handle:horizontal {
                background: #f97316;
                border: 1px solid #c2410c;
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            """
        )

    def _current_lang_code(self) -> str:
        code = self.lang_combo.currentData()
        return str(code) if code else "en-us"

    def _refresh_preset_names(self) -> None:
        names = [self._t(f"preset_{k}") for k in self._preset_keys]
        current = names[0] if names else ""
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItems(names)
        self.preset_combo.setCurrentText(current)
        self.preset_combo.blockSignals(False)

    def _on_text_document_size_changed(self, _size) -> None:
        doc_height = self.text_input.document().documentLayout().documentSize().height()
        viewport = self.text_input.viewportMargins()
        frame = self.text_input.frameWidth() * 2
        vertical_padding = viewport.top() + viewport.bottom()
        target = max(self._text_min_height, int(math.ceil(doc_height + frame + vertical_padding)))
        self._text_current_height = target
        self._resize_text_input()

    def _resize_text_input(self) -> None:
        target = max(self._text_min_height, int(self._text_current_height))
        self._text_current_height = target
        scroll_bar = self._generate_scroll.verticalScrollBar() if hasattr(self, "_generate_scroll") else None
        previous_scroll = scroll_bar.value() if scroll_bar is not None else 0
        self.text_input.setFixedHeight(target)
        self.text_input.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        if scroll_bar is not None and self.text_input.hasFocus():
            QTimer.singleShot(0, lambda sb=scroll_bar, v=previous_scroll: sb.setValue(v))

    def _init_engine(self) -> None:
        self._load_voices()
        self._set_status(self._t("up_to_date", version=APP_VERSION))

    def _load_voices(self) -> None:
        self._set_status(self._t("loading_model"))
        QApplication.processEvents()
        try:
            _, voices = ensure_engine()
        except Exception as exc:
            self._set_status(f"Error: {exc}")
            QMessageBox.critical(self, "Kokoro TTS", str(exc))
            return

        if get_tts_engine() == "piper":
            filtered = voices
        else:
            filtered = voices_for_lang(self._current_lang_code(), voices, self.show_all_chk.isChecked())
        self._voices = filtered
        self.voice_combo.clear()
        self.voice_combo.addItems(filtered)
        if "af_heart" in filtered:
            self.voice_combo.setCurrentText("af_heart")
        self._set_status(self._t("up_to_date", version=APP_VERSION))

    def _on_lang_change(self, _=None) -> None:
        if get_tts_engine() == "piper":
            self._load_voices()
            return
        try:
            _, voices = ensure_engine()
        except Exception:
            return
        filtered = voices_for_lang(self._current_lang_code(), voices, self.show_all_chk.isChecked())
        current = self.voice_combo.currentText()
        self.voice_combo.clear()
        self.voice_combo.addItems(filtered)
        if current in filtered:
            self.voice_combo.setCurrentText(current)

    def _on_show_all(self, _=None) -> None:
        settings = load_settings()
        settings["show_all_voices"] = self.show_all_chk.isChecked()
        save_settings(settings)
        self._on_lang_change()

    def _on_fav_select(self, value: str) -> None:
        if value and value != "—":
            self.voice_combo.setCurrentText(value)

    def _on_toggle_fav(self) -> None:
        voice = self.voice_combo.currentText().strip()
        if not voice or voice == "…":
            return
        favs, status = toggle_favorite(voice)
        self.fav_combo.clear()
        self.fav_combo.addItems(favs if favs else ["—"])
        if favs:
            self.fav_combo.setCurrentText(favs[0])
        self._set_status(status)

    def _on_preset(self, display_name: str) -> None:
        names = [self._t(f"preset_{k}") for k in self._preset_keys]
        idx = names.index(display_name) if display_name in names else 0
        style, speed, gain = apply_preset(self._preset_keys[idx])
        self.style_combo.setCurrentText(style)
        self.speed_spin.setValue(speed)
        self.gain_spin.setValue(gain)

    def _update_mp3_bitrate_state(self) -> None:
        is_mp3 = self.format_combo.currentText().strip().lower() == "mp3"
        self.bitrate_combo.setEnabled(is_mp3)

    def _on_audio_setting_change(self, _value: str) -> None:
        self._update_mp3_bitrate_state()
        settings = load_settings()
        settings["output_format"] = self.format_combo.currentText().strip().lower()
        settings["mp3_bitrate_kbps"] = int(self.bitrate_combo.currentText())
        save_settings(settings)

    def _on_generate(self) -> None:
        text = self.text_input.toPlainText().strip()
        if not text:
            self._set_status(self._t("error_empty_text"))
            return

        if not self._voices:
            self._load_voices()
            if not self._voices:
                return

        voice = self.voice_combo.currentText().strip()
        if not voice:
            self._set_status(self._t("error_empty_text"))
            return

        self._set_status("Generating…")
        self._set_generate_loading(True)

        args = {
            "text": text,
            "voice": voice,
            "speed": self.speed_spin.value(),
            "lang": self._current_lang_code(),
            "style": self.style_combo.currentText(),
            "gain_db": self.gain_spin.value(),
            "output_format": self.format_combo.currentText(),
            "mp3_bitrate_kbps": int(self.bitrate_combo.currentText()),
        }

        def _worker() -> None:
            try:
                out_path, info = synthesize(**args)
                error = ""
            except Exception as exc:
                out_path, info = "", ""
                error = str(exc)
            self._generate_result_queue.put((args["text"], out_path, info, error))

        self._generate_result_queue = queue.Queue(maxsize=1)
        threading.Thread(target=_worker, daemon=True).start()
        self._poll_generate_result()

    def _poll_generate_result(self) -> None:
        try:
            text, out_path, info, error = self._generate_result_queue.get_nowait()
        except queue.Empty:
            QTimer.singleShot(80, self._poll_generate_result)
            return
        self._finish_generate(text, out_path, info, error)

    def _set_generate_loading(self, loading: bool) -> None:
        if loading:
            self.generate_btn.setEnabled(False)
            self.generate_action_stack.setCurrentWidget(self.generate_progress)
        else:
            self.generate_action_stack.setCurrentWidget(self.generate_btn)
            self.generate_btn.setEnabled(True)

    def _finish_generate(self, text: str, out_path: str, info: str, error: str) -> None:
        self._set_generate_loading(False)
        if error:
            self._set_status(error)
            QMessageBox.critical(self, "Kokoro TTS", error)
            return

        clean = " ".join(text.split())
        snippet = clean[:30] + ("…" if len(clean) > 30 else "")
        ts = datetime.now().strftime("%H:%M")
        label = f"{snippet}  ({ts})"
        self._history.insert(0, (label, out_path))
        self._selected_history = 0
        self._refresh_history()
        self.player.load_file(out_path)
        self.player._play(0.0)

        self._set_status(info)

    def _refresh_history(self) -> None:
        self.history_list.clear()
        for i, (label, _) in enumerate(self._history[:10]):
            item = QListWidgetItem(label)
            self.history_list.addItem(item)
        if self._history:
            self.history_list.setCurrentRow(min(self._selected_history, self.history_list.count() - 1))

    def _on_history_select(self, item: QListWidgetItem) -> None:
        row = self.history_list.row(item)
        if row < 0 or row >= len(self._history):
            return
        self._selected_history = row
        _, path = self._history[row]
        self.player.load_file(path)
        self.player._play(0.0)
        self._refresh_history()

    def _on_save(self) -> None:
        if not self._history:
            return
        _, path = self._history[self._selected_history]
        src = Path(path)
        if not src.exists():
            return
        selected_format = self.format_combo.currentText().strip().lower()
        preferred_ext = ".mp3" if selected_format == "mp3" else ".wav"
        if preferred_ext == ".mp3":
            initial_filter = "MP3 Audio (*.mp3)"
            filters = "MP3 Audio (*.mp3);;WAV Audio (*.wav);;All files (*.*)"
        else:
            initial_filter = "WAV Audio (*.wav)"
            filters = "WAV Audio (*.wav);;MP3 Audio (*.mp3);;All files (*.*)"
        suggested = src.with_suffix(preferred_ext)
        dest, _ = QFileDialog.getSaveFileName(
            self,
            self._t("download"),
            str(suggested),
            filters,
            initial_filter,
        )
        if dest:
            dest_path = Path(dest)
            if dest_path.suffix.lower() != preferred_ext:
                dest_path = dest_path.with_suffix(preferred_ext)
            shutil.copy2(src, dest_path)
            self._set_status(f"Saved → {dest_path.name}")

    def _on_uilang(self, value: str) -> None:
        code = "nb" if value == "Norsk" else "en"
        settings = load_settings()
        settings["ui_language"] = code
        save_settings(settings)
        self._L = code
        self._set_status(self._t("language_saved"))

    def _on_model(self, value: str) -> None:
        if get_tts_engine() == "piper":
            selected = value.strip()
            if selected not in get_available_piper_models():
                return
            settings = load_settings()
            settings["piper_model"] = selected
            save_settings(settings)
            self._refresh_engine_ui_state()
            self.model_info.setText(f"{self._t('model_switched', version=selected)}\n{self.model_info.text()}")
            self._load_voices()
            return

        version = value.replace("Kokoro ", "")
        settings = load_settings()
        settings["model_version"] = version
        save_settings(settings)
        self.model_info.setText("Loading…")

        try:
            _, voices = ensure_engine(version)
            filtered = voices_for_lang(self._current_lang_code(), voices, self.show_all_chk.isChecked())
            self.voice_combo.clear()
            self.voice_combo.addItems(filtered)
            self.model_info.setText(self._t("model_switched", version=version))
        except Exception as exc:
            self.model_info.setText(self._t("model_switch_error", error=str(exc)))

    def _refresh_engine_ui_state(self) -> None:
        engine = get_tts_engine()
        kokoro_selected = engine == "kokoro"
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        if kokoro_selected:
            self.model_combo.addItems([f"Kokoro {k}" for k in MODEL_REGISTRY])
            self.model_combo.setCurrentText(f"Kokoro {get_model_version()}")
            self.model_combo.setEnabled(True)
            self.model_info.setText("")
            self.piper_download_btn.setVisible(False)
            self.piper_progress.setVisible(False)
            self.piper_progress_label.setVisible(False)
        else:
            piper_models = get_available_piper_models()
            self.model_combo.addItems(piper_models)
            self.model_combo.setCurrentText(get_piper_model())
            self.model_combo.setEnabled(True)
            selected = get_piper_model()
            downloaded = is_piper_model_downloaded(selected)
            state_key = "piper_downloaded" if downloaded else "piper_not_downloaded"
            self.model_info.setText(f"{self._t('engine_piper_model_info')}\n{self._t(state_key)}")
            self.piper_download_btn.setVisible(not downloaded)
            self.piper_progress.setVisible(False)
            self.piper_progress_label.setVisible(False)
            self.piper_download_btn.setEnabled(not downloaded)
        self.model_combo.blockSignals(False)

    def _on_tts_engine_change(self, _index: int) -> None:
        engine = self.engine_combo.currentData()
        settings = load_settings()
        settings["tts_engine"] = str(engine) if engine else "kokoro"
        save_settings(settings)
        self._refresh_engine_ui_state()
        self._load_voices()

    def _format_mb_progress(self, current_bytes: int, total_bytes: int) -> str:
        current_mb = max(0.0, current_bytes / (1024 * 1024))
        total_mb = max(0.0, total_bytes / (1024 * 1024))
        if total_mb > 0:
            return f"{current_mb:.1f} MB / {total_mb:.1f} MB"
        return f"{current_mb:.1f} MB"

    def _on_download_piper_model(self) -> None:
        selected = get_piper_model()
        if not selected:
            return

        self.piper_download_btn.setEnabled(False)
        self.piper_progress.setVisible(True)
        self.piper_progress_label.setVisible(True)
        self.piper_progress.setValue(0)
        self.piper_progress_label.setText(self._t("piper_download_starting"))

        self._piper_download_queue = queue.Queue()

        def _progress(downloaded: int, total: int) -> None:
            self._piper_download_queue.put(("progress", int(downloaded), int(total or 0), ""))

        def _worker() -> None:
            try:
                download_piper_model(selected, progress_cb=_progress)
                self._piper_download_queue.put(("done", 0, 0, ""))
            except Exception as exc:
                self._piper_download_queue.put(("error", 0, 0, str(exc)))

        threading.Thread(target=_worker, daemon=True).start()
        self._poll_piper_download_result()

    def _poll_piper_download_result(self) -> None:
        try:
            kind, downloaded, total, message = self._piper_download_queue.get_nowait()
        except queue.Empty:
            QTimer.singleShot(100, self._poll_piper_download_result)
            return

        if kind == "progress":
            pct = int((downloaded / total) * 100) if total > 0 else 0
            self.piper_progress.setValue(max(0, min(100, pct)))
            self.piper_progress_label.setText(self._format_mb_progress(downloaded, total))
            QTimer.singleShot(50, self._poll_piper_download_result)
            return

        if kind == "done":
            self.piper_progress.setValue(100)
            self.piper_progress_label.setText(self._t("piper_download_done"))
            self._refresh_engine_ui_state()
            return

        self.piper_progress_label.setText(message)
        self.piper_download_btn.setEnabled(True)

    def _on_check_update(self) -> None:
        self.update_label.setText("…")
        QApplication.processEvents()
        details = check_updates_details()
        message = str(details.get("message") or check_updates_message())
        self.update_label.setText(message)
        if bool(details.get("update_available")):
            tag = str(details.get("tag") or "")
            release_notes = str(details.get("release_notes") or "").strip()
            if not release_notes:
                release_notes = self._t("release_notes_empty")
            self.release_notes_title.setText(self._t("release_notes_title", tag=tag))
            self.release_notes_title.setVisible(True)
            self.release_notes_view.setPlainText(release_notes)
            self.release_notes_view.setVisible(True)
        else:
            self.release_notes_title.setVisible(False)
            self.release_notes_view.clear()
            self.release_notes_view.setVisible(False)

    def _load_rollback_releases(self) -> None:
        self.rollback_combo.clear()
        self.rollback_combo.addItem(self._t("rollback_loading"), "")
        QApplication.processEvents()
        try:
            releases = get_recent_releases(20)
        except Exception:
            self._rollback_releases = []
            self.rollback_combo.clear()
            self.rollback_combo.addItem(self._t("rollback_unavailable"), "")
            self.rollback_btn.setEnabled(False)
            return

        self._rollback_releases = releases
        self.rollback_combo.clear()
        if not releases:
            self.rollback_combo.addItem(self._t("rollback_unavailable"), "")
            self.rollback_btn.setEnabled(False)
            return

        for release in releases:
            tag = release.get("tag", "")
            self.rollback_combo.addItem(tag, tag)
        self.rollback_btn.setEnabled(True)

    def _on_rollback(self) -> None:
        selected_tag = str(self.rollback_combo.currentData() or "")
        selected_release = None
        for release in self._rollback_releases:
            if release.get("tag") == selected_tag:
                selected_release = release
                break
        if not selected_release:
            self.update_label.setText(self._t("rollback_invalid_selection"))
            return

        self.rollback_btn.setEnabled(False)
        self.auto_update_btn.setEnabled(False)
        self.update_label.setText(self._t("rollback_starting", tag=selected_tag))
        self._update_progress = QProgressDialog(self._t("rollback_progress"), "", 0, 0, self)
        self._update_progress.setWindowTitle("Kokoro TTS")
        self._update_progress.setCancelButton(None)
        self._update_progress.setWindowModality(Qt.ApplicationModal)
        self._update_progress.setMinimumDuration(0)
        self._update_progress.show()
        QApplication.processEvents()

        def _worker() -> None:
            try:
                msg = rollback_to_release(
                    tag=selected_release.get("tag", ""),
                    dmg_url=selected_release.get("dmg_url", ""),
                    release_url=selected_release.get("url", ""),
                )
            except Exception as exc:
                msg = str(exc)
            self._update_result_queue.put(msg)

        self._update_result_queue = queue.Queue(maxsize=1)
        threading.Thread(target=_worker, daemon=True).start()
        self._poll_auto_update_result()

    def _on_auto_update(self) -> None:
        self.auto_update_btn.setEnabled(False)
        self.update_label.setText("Starter oppdatering…")
        self._update_progress = QProgressDialog("Laster ned og installerer oppdatering…", "", 0, 0, self)
        self._update_progress.setWindowTitle("Kokoro TTS")
        self._update_progress.setCancelButton(None)
        self._update_progress.setWindowModality(Qt.ApplicationModal)
        self._update_progress.setMinimumDuration(0)
        self._update_progress.show()
        QApplication.processEvents()

        def _worker() -> None:
            try:
                msg = auto_update()
            except Exception as exc:
                msg = str(exc)
            self._update_result_queue.put(msg)

        self._update_result_queue = queue.Queue(maxsize=1)
        threading.Thread(target=_worker, daemon=True).start()
        self._poll_auto_update_result()

    def _poll_auto_update_result(self) -> None:
        try:
            message = self._update_result_queue.get_nowait()
        except queue.Empty:
            QTimer.singleShot(120, self._poll_auto_update_result)
            return
        self._finish_auto_update(message)

    def _finish_auto_update(self, message: str) -> None:
        if hasattr(self, "_update_progress") and self._update_progress is not None:
            self._update_progress.close()
            self._update_progress = None
        self.auto_update_btn.setEnabled(True)
        if hasattr(self, "rollback_btn"):
            self.rollback_btn.setEnabled(bool(getattr(self, "_rollback_releases", [])))
        self.update_label.setText(message)

    def _show_settings_page(self) -> None:
        self.pages.setCurrentWidget(self.settings_page)

    def _show_main_page(self) -> None:
        self.pages.setCurrentWidget(self.main_page)

    def _on_restore_audio_defaults(self) -> None:
        self.style_combo.setCurrentText("Neutral")
        self._refresh_preset_names()
        self.speed_spin.setValue(1.0)
        self.gain_spin.setValue(0.0)
        self.format_combo.setCurrentText("wav")
        self.bitrate_combo.setCurrentText("192")
        self._update_mp3_bitrate_state()
        self._set_status(self._t("audio_defaults_restored"))

    def _on_restore_defaults(self) -> None:
        self._L = "nb"
        save_settings({})

        self.ui_lang_combo.blockSignals(True)
        self.ui_lang_combo.setCurrentText("Norsk")
        self.ui_lang_combo.blockSignals(False)

        self.show_all_chk.blockSignals(True)
        self.show_all_chk.setChecked(False)
        self.show_all_chk.blockSignals(False)

        self.model_combo.blockSignals(True)
        self.model_combo.setCurrentText(f"Kokoro {DEFAULT_MODEL_VERSION}")
        self.model_combo.blockSignals(False)

        self.engine_combo.blockSignals(True)
        self.engine_combo.setCurrentIndex(0)
        self.engine_combo.blockSignals(False)
        cfg = load_settings()
        cfg["tts_engine"] = "kokoro"
        cfg["piper_model"] = DEFAULT_PIPER_MODEL
        save_settings(cfg)
        self._refresh_engine_ui_state()

        self.lang_combo.blockSignals(True)
        self.lang_combo.setCurrentIndex(0)
        self.lang_combo.blockSignals(False)
        self.style_combo.setCurrentText("Neutral")
        self._refresh_preset_names()
        self.speed_spin.setValue(1.0)
        self.gain_spin.setValue(0.0)
        self.format_combo.setCurrentText("wav")
        self.bitrate_combo.setCurrentText("192")
        self._update_mp3_bitrate_state()

        self._on_lang_change()
        self._set_status(self._t("defaults_restored"))

    def closeEvent(self, event) -> None:
        self.player.stop()
        super().closeEvent(event)

def main() -> None:
    plugins_root = QLibraryInfo.path(QLibraryInfo.PluginsPath)
    if plugins_root:
        platform_plugins = str(Path(plugins_root) / "platforms")
        if not os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = platform_plugins

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = KokoroQtWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
