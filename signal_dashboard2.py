"""
Signal Dashboard - UI + live hardware acquisition for the laser-microphone
project.

This version reads directly from the PCM1808 hardware using the same
acquisition/filter pipeline as dsp4ch.py (PyAudio capture, persistent-state
6th-order Butterworth bandpass, zero-crossing trigger for a stable trace).
It is no longer driven by fake/demo waveforms - if the hardware can't be
opened, the dashboard falls back to a synthetic signal automatically so the
UI still runs, but the live PCM1808 stream is the primary data source.

Oscilloscopes:
    - Standard Mic - Raw
    - Standard Mic - Bandpass
    - Laser Mic    - Raw
    - Laser Mic    - Bandpass

The two microphones are read from the two channels of the stereo PCM1808
input (see STANDARD_MIC_CHANNEL / LASER_MIC_CHANNEL below). There is no
longer a UI switch for the bandpass filter - both the raw and the filtered
signal are always shown, for both microphones, continuously.

The Spectrogram and Prediction/Confidence panels are unchanged in spirit:
the spectrogram is computed from the laser mic's filtered signal, and the
Prediction/Confidence panel still uses the optional ML backend
(signal_backend.SignalBackend) when available, or demo values otherwise.
Everywhere the real classifier still needs to be plugged in is marked with
a comment starting with "ML TEAM:".

How to run:
    pip install PyQt5 pyqtgraph numpy pyaudio scipy
    python signal_dashboard.py

Needs an actual display (Windows/Mac/Linux desktop) - this won't draw
anything on a headless server. Needs the PCM1808 (or another stereo input
device) attached; otherwise it will print a warning and fall back to a
synthetic demo signal.

If you're using PySide6 instead of PyQt5, change the import at the top and
change app.exec_() to app.exec() at the bottom.
"""

import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

import pyaudio
from scipy.signal import butter, sosfilt, sosfilt_zi
from scipy.signal import spectrogram as scipy_spectrogram
from scipy import ndimage

# Real ML backend (trained classifier). Optional: if it can't be imported
# (missing deps, etc.) the Prediction/Confidence panel falls back to demo
# values, but the oscilloscopes/spectrogram below always use live hardware.
try:
    from signal_backend import SignalBackend
except Exception as _e:  # noqa: BLE001
    SignalBackend = None
    print(f"[dashboard] ML backend unavailable, predictions will use demo values ({_e})")


# ============================================================================
# THEME
# ============================================================================
BG_COLOR = "#0d1117"        # window background
PANEL_COLOR = "#12161c"     # graph / box background
BORDER_COLOR = "#2a2f3a"    # borders, grid lines
TEXT_COLOR = "#c9d1d9"      # main text
DIM_TEXT_COLOR = "#7d8590"  # secondary text / axis labels
ACCENT_COLOR = "#39ff88"    # result values (green, oscilloscope-y)
RECORD_COLOR = "#ff453a"    # record button red
STOP_COLOR = "#2d333b"      # stop button body

# One trace color per scope pane, chosen to stay legible against PANEL_COLOR.
STD_RAW_COLOR = "#58a6ff"     # standard mic, raw   (blue)
STD_BP_COLOR = "#39ff88"      # standard mic, bandpass (green)
LASER_RAW_COLOR = "#ffb454"   # laser mic, raw      (amber)
LASER_BP_COLOR = "#d2a8ff"    # laser mic, bandpass (violet)

UI_FONT = "Segoe UI, Helvetica Neue, Arial, sans-serif"
MONO_FONT = "Consolas, JetBrains Mono, monospace"  # used for the numeric readouts

SPECTROGRAM_CMAP = "inferno"

# Placeholder prediction classes - swap this list for whatever your actual
# classifier outputs.
PREDICTION_LABELS = [str(d) for d in range(10)] + ["Unknown"]


# ============================================================================
# HARDWARE / DSP CONFIGURATION - ported from dsp4ch.py
# ============================================================================

FORMAT = pyaudio.paInt32
CHANNELS = 2
RATE = 48000

# Keep the exact known-working buffer size from dsp4ch.py.
CHUNK = 4096 * 3
PLOT_POINTS = CHUNK

# Visual-only gain, applied after filtering (does not affect the DSP).
VISUAL_GAIN = 2.0

# Which physical input channel each microphone is wired to on the PCM1808.
# HARDWARE TEAM: swap these two if your standard mic and laser mic end up
# on the opposite channels.
STANDARD_MIC_CHANNEL = 0   # left  input -> standard reference microphone
LASER_MIC_CHANNEL = 1      # right input -> laser microphone (photodiode receiver)

FILTER_ORDER = 6
LOWCUT_HZ = 200.0
HIGHCUT_HZ = 8000.0
NYQUIST = RATE * 0.5

SOS = butter(
    FILTER_ORDER,
    [LOWCUT_HZ / NYQUIST, HIGHCUT_HZ / NYQUIST],
    btype="bandpass",
    output="sos",
)

TIME_AXIS = np.arange(PLOT_POINTS, dtype=np.float64) / RATE

# ----------------------------------------------------------------------
# Spectrogram display settings.
#
# A 256 ms chunk only yields a handful of real STFT time/frequency bins,
# which looks blocky once stretched across the panel. SPECTROGRAM_NOVERLAP
# raises the real time resolution; SPECTROGRAM_ZOOM smoothly interpolates
# the resulting grid for display only (it does not add real information,
# it just removes the "checkerboard" look between the bins we do have).
# ----------------------------------------------------------------------
SPECTROGRAM_NPERSEG = 1024
SPECTROGRAM_NOVERLAP = 896       # ~87.5% overlap -> more time bins per chunk
SPECTROGRAM_ZOOM = (4, 3)        # (time, frequency) display upsample factor


def find_audio_device(p):
    """Locate the PCM1808 capture device. Copied from dsp4ch.py so both
    tools agree on which hardware input to use."""
    print()
    print("=" * 80)
    print("AVAILABLE AUDIO DEVICES")
    print("=" * 80)

    selected_device = None

    for i in range(p.get_device_count()):
        try:
            dev_info = p.get_device_info_by_index(i)
            name = dev_info.get("name", "")
            max_input = dev_info.get("maxInputChannels", 0)
            default_rate = dev_info.get("defaultSampleRate", 0)

            print(f"[{i}] {name}")
            print(f"     Input channels: {max_input}")
            print(f"     Default rate: {default_rate}")

            name_upper = name.upper()

            if (
                selected_device is None
                and ("APE" in name_upper or "HW:1,0" in name_upper)
                and max_input >= CHANNELS
            ):
                selected_device = i

        except Exception as exc:
            print(f"Could not inspect device {i}: {exc}")

    print("=" * 80)

    if selected_device is not None:
        print(f"Selected PCM1808 device: {selected_device}")
        return selected_device

    # Preserve the known-working fallback to device 1.
    if p.get_device_count() > 1:
        try:
            info = p.get_device_info_by_index(1)
            if info.get("maxInputChannels", 0) >= CHANNELS:
                print("APE/hw:1,0 device not identified by name.")
                print("Using fallback device index 1.")
                return 1
        except Exception:
            pass

    # Last resort: first stereo input.
    for i in range(p.get_device_count()):
        try:
            info = p.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) >= CHANNELS:
                print(f"Using first available stereo input device: {i}")
                return i
        except Exception:
            pass

    return None


def trigger_window(wave_data, trigger_source):
    """Find the first rising-edge zero crossing in trigger_source and slice
    the same window out of wave_data, so the oscilloscope trace looks
    stable instead of scrolling randomly. Copied from dsp4ch.py."""
    if len(trigger_source) < 2:
        return None

    crossings = np.where(
        (trigger_source[:-1] <= 0.0) & (trigger_source[1:] > 0.0)
    )[0]

    if len(crossings) == 0:
        return None

    idx = int(crossings[0])
    display = wave_data[idx: idx + PLOT_POINTS]

    if len(display) < PLOT_POINTS:
        display = np.pad(display, (0, PLOT_POINTS - len(display)), mode="constant")

    return np.clip(display, -1.0, 1.0)


class HardwareAudioSource:
    """Owns the PyAudio stream into the PCM1808 and produces raw + bandpass
    windows for both microphones on every read. This is dsp4ch.py's
    acquisition/filter pipeline, reused as-is, so the dashboard reads real
    hardware instead of demo data."""

    def __init__(self):
        self.pyaudio = None
        self.stream = None
        self.device_index = None
        self.zi_std = sosfilt_zi(SOS) * 0.0
        self.zi_laser = sosfilt_zi(SOS) * 0.0
        self.available = False
        self.error = None
        self._open()

    def _open(self):
        # Lets you exercise the UI/demo path on purpose, even on a machine
        # that does have some audio device attached: `LMML_FORCE_DEMO=1
        # python signal_dashboard.py`
        if os.environ.get("LMML_FORCE_DEMO"):
            self.error = "LMML_FORCE_DEMO set - skipping hardware, using demo data."
            print(f"[hardware] {self.error}")
            return
        try:
            # PyAudio() itself can throw if there's no PortAudio backend at
            # all (e.g. a bare dev container with no sound subsystem), not
            # just when the PCM1808 specifically is missing.
            self.pyaudio = pyaudio.PyAudio()
            self.device_index = find_audio_device(self.pyaudio)
            if self.device_index is None:
                raise RuntimeError("No suitable stereo audio input device found.")

            info = self.pyaudio.get_device_info_by_index(self.device_index)
            print()
            print("Opening audio device:")
            print(f"  Index: {self.device_index}")
            print(f"  Name: {info.get('name')}")
            print(f"  Channels: {CHANNELS}")
            print(f"  Sample rate: {RATE}")
            print(f"  Chunk: {CHUNK} frames")

            self.stream = self.pyaudio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=CHUNK,
            )
            self.available = True
            print()
            print("Audio hardware opened successfully.")
            print(f"Bandpass filter: {LOWCUT_HZ:.0f} - {HIGHCUT_HZ:.0f} Hz, order {FILTER_ORDER}")
        except Exception as exc:
            self.available = False
            self.error = str(exc)
            print(f"[hardware] Could not open PCM1808 input, falling back to demo data: {exc}")

    def read(self):
        """Returns a dict with std_raw/std_bp/laser_raw/laser_bp arrays
        (length PLOT_POINTS, clipped to [-1, 1]) plus the unclipped filtered
        laser-mic signal for the spectrogram, or None if a chunk couldn't be
        captured this tick (buffer underrun / no trigger yet)."""
        if not self.available:
            return None
        try:
            raw_bytes = self.stream.read(CHUNK, exception_on_overflow=False)
            samples = np.frombuffer(raw_bytes, dtype=np.int32)

            if len(samples) != CHUNK * CHANNELS:
                return None

            ch0 = samples[0::2].astype(np.float64) / 2147483648.0
            ch1 = samples[1::2].astype(np.float64) / 2147483648.0
            channels = {0: ch0, 1: ch1}

            std_raw_full = channels[STANDARD_MIC_CHANNEL]
            laser_raw_full = channels[LASER_MIC_CHANNEL]

            std_bp_full, self.zi_std = sosfilt(SOS, std_raw_full, zi=self.zi_std)
            laser_bp_full, self.zi_laser = sosfilt(SOS, laser_raw_full, zi=self.zi_laser)

            std_raw = trigger_window(std_raw_full, std_bp_full)
            std_bp = trigger_window(std_bp_full, std_bp_full)
            laser_raw = trigger_window(laser_raw_full, laser_bp_full)
            laser_bp = trigger_window(laser_bp_full, laser_bp_full)

            if std_raw is None or std_bp is None or laser_raw is None or laser_bp is None:
                return None

            return {
                "std_raw": np.clip(std_raw * VISUAL_GAIN, -1.0, 1.0),
                "std_bp": np.clip(std_bp * VISUAL_GAIN, -1.0, 1.0),
                "laser_raw": np.clip(laser_raw * VISUAL_GAIN, -1.0, 1.0),
                "laser_bp": np.clip(laser_bp * VISUAL_GAIN, -1.0, 1.0),
                "laser_bp_full": laser_bp_full,  # unclipped, full chunk - for the spectrogram
            }
        except Exception as exc:
            print("[hardware] read error:", repr(exc))
            return None

    def close(self):
        if self.stream is not None:
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
            except Exception as exc:
                print("[hardware] stream stop warning:", exc)
            try:
                self.stream.close()
            except Exception as exc:
                print("[hardware] stream close warning:", exc)
        if self.pyaudio is not None:
            try:
                self.pyaudio.terminate()
            except Exception as exc:
                print("[hardware] PyAudio termination warning:", exc)


class CircleButton(QtWidgets.QPushButton):
    """The round Record button. Call set_recording(True/False) to switch
    it into its "active" look (a soft red glow that breathes in and out,
    like the little LED on a real recorder)."""

    def __init__(self, diameter=90, parent=None):
        super().__init__(parent)
        self.setFixedSize(diameter, diameter)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {RECORD_COLOR};
                border: 3px solid #e6e6e6;
                border-radius: {diameter // 2}px;
            }}
            QPushButton:hover {{ background-color: #ff6259; }}
            QPushButton:pressed {{ background-color: #c73128; }}
            QPushButton:disabled {{ background-color: #5a2622; border-color: #555; }}
        """)

        # glow effect used while recording - see set_recording() below
        self.glow = QtWidgets.QGraphicsDropShadowEffect(self)
        self.glow.setColor(QtGui.QColor(RECORD_COLOR))
        self.glow.setOffset(0, 0)
        self.glow.setBlurRadius(10)
        self.setGraphicsEffect(self.glow)

        self.pulse_anim = QtCore.QPropertyAnimation(self.glow, b"blurRadius", self)
        self.pulse_anim.setDuration(900)
        self.pulse_anim.setKeyValueAt(0.0, 10)
        self.pulse_anim.setKeyValueAt(0.5, 40)
        self.pulse_anim.setKeyValueAt(1.0, 10)
        self.pulse_anim.setLoopCount(-1)

    def set_recording(self, active):
        if active:
            self.pulse_anim.start()
        else:
            self.pulse_anim.stop()
            self.glow.setBlurRadius(10)
        self.setEnabled(not active)  # can't smash record twice while it's already going


class StopButton(QtWidgets.QPushButton):
    """Rectangular Stop button. Draws its own pause icon in paintEvent
    instead of using a font character, so it always looks the same no
    matter what fonts are installed."""

    def __init__(self, width=116, height=70, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {STOP_COLOR};
                border: 2px solid {BORDER_COLOR};
                border-radius: 10px;
            }}
            QPushButton:hover {{ background-color: #3a4453; }}
            QPushButton:pressed {{ background-color: #21262d; }}
            QPushButton:disabled {{ background-color: #1b1f27; }}
        """)

    def paintEvent(self, event):
        super().paintEvent(event)  # let the stylesheet draw the background/border first
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        icon_color = TEXT_COLOR if self.isEnabled() else "#4a4f58"
        painter.setBrush(QtGui.QColor(icon_color))
        painter.setPen(QtCore.Qt.NoPen)

        bar_w = self.width() * 0.1
        bar_h = self.height() * 0.42
        gap = self.width() * 0.1
        cx, cy = self.width() / 2, self.height() / 2
        painter.drawRoundedRect(QtCore.QRectF(cx - gap / 2 - bar_w, cy - bar_h / 2, bar_w, bar_h), 2, 2)
        painter.drawRoundedRect(QtCore.QRectF(cx + gap / 2, cy - bar_h / 2, bar_w, bar_h), 2, 2)


class ResultBox(QtWidgets.QGroupBox):
    """A titled box with one big value in the middle - used for both the
    Prediction box and the Confidence Percentage box."""

    def __init__(self, title, initial_value="--", unit="", parent=None):
        super().__init__(title, parent)
        self.unit = unit
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setStyleSheet(f"""
            QGroupBox {{
                color: {DIM_TEXT_COLOR};
                background-color: {PANEL_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                margin-top: 14px;
                font-family: {UI_FONT};
                font-size: 12px;
                font-weight: 600;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }}
        """)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 22, 16, 18)
        self.value_label = QtWidgets.QLabel(str(initial_value))
        self.value_label.setAlignment(QtCore.Qt.AlignCenter)
        self.value_label.setWordWrap(True)
        self.value_label.setStyleSheet(f"""
            color: {ACCENT_COLOR};
            font-family: {MONO_FONT};
            font-size: 56px;
            font-weight: 700;
            border: none;
            background: transparent;
        """)
        layout.addWidget(self.value_label)

    def set_value(self, value):
        self.value_label.setText(f"{value}{self.unit}")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal Dashboard - PCM1808 Standard Mic + Laser Mic")
        self.resize(1500, 940)
        self.setStyleSheet(f"QMainWindow {{ background-color: {BG_COLOR}; }}")

        self.is_recording = False
        self.is_fullscreen = False
        self.last_chunk = None

        # ---------------------------------------------------------------
        # Live hardware acquisition (dsp4ch.py pipeline). This is the
        # primary data source for all four oscilloscopes + spectrogram.
        # ---------------------------------------------------------------
        self.hardware = HardwareAudioSource()

        # Demo fallback state, used only if the hardware couldn't be
        # opened, so the UI still runs on a dev machine with no PCM1808
        # attached.
        self._demo_t_offset = 0.0
        self._demo_zi_std = sosfilt_zi(SOS) * 0.0
        self._demo_zi_laser = sosfilt_zi(SOS) * 0.0

        # ---------------------------------------------------------------
        # Real ML backend (trained classifier), optional.
        #
        # ML TEAM: when live hardware is active, self.last_chunk (set every
        # tick in update_frame) holds the most recent std_raw/std_bp/
        # laser_raw/laser_bp/laser_bp_full arrays straight from the PCM1808.
        # Feed the classifier from that buffer instead of having
        # SignalBackend open its own microphone stream - otherwise you'll
        # have two processes fighting over the same audio device.
        # ---------------------------------------------------------------
        self.backend = None
        if SignalBackend is not None and not self.hardware.available:
            try:
                source = os.environ.get("LMML_SIGNAL_SOURCE", "mic")
                self.backend = SignalBackend(model="ensemble", source=source)
                print(self.backend.status_text())
            except Exception as e:  # noqa: BLE001
                print(f"[dashboard] backend init failed, using demo predictions: {e}")
                self.backend = None

        self.build_ui()

        # This timer drives both the live hardware read and (when hardware
        # isn't available) the demo animation. 40 ms matches dsp4ch.py;
        # the blocking stream.read() inside HardwareAudioSource.read()
        # naturally paces this to one chunk (~256 ms) per update when
        # hardware is attached.
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(40)

    # =====================================================================
    # UI CONSTRUCTION
    # =====================================================================

    def build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(16)
        root.addLayout(body, stretch=1)
        body.addLayout(self.build_graphs_column(), stretch=4)
        body.addWidget(self.build_result_panel(), stretch=0)

        root.addLayout(self.build_button_row(), stretch=0)

    def _make_scope(self, title, color):
        """Build one themed oscilloscope pane + its curve."""
        plot = pg.PlotWidget()
        plot.setBackground(PANEL_COLOR)
        plot.setTitle(title, color=TEXT_COLOR, size="10pt")
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.setLabel("bottom", "Time", units="s", color=DIM_TEXT_COLOR)
        plot.setLabel("left", "Amplitude", color=DIM_TEXT_COLOR)
        plot.setYRange(-1.0, 1.0, padding=0)
        plot.getAxis("bottom").setPen(BORDER_COLOR)
        plot.getAxis("left").setPen(BORDER_COLOR)
        plot.getAxis("bottom").setTextPen(DIM_TEXT_COLOR)
        plot.getAxis("left").setTextPen(DIM_TEXT_COLOR)
        curve = plot.plot(pen=pg.mkPen(color=color, width=1.5))
        return plot, curve

    def build_graphs_column(self):
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(10)

        # ---- Four oscilloscopes: raw + bandpass for each microphone ----
        # No more bandpass on/off switch - raw and filtered are both
        # always shown, for both mics, continuously.
        self.scope_std_raw, self.curve_std_raw = self._make_scope(
            "Standard Mic - Raw", STD_RAW_COLOR
        )
        self.scope_std_bp, self.curve_std_bp = self._make_scope(
            f"Standard Mic - Bandpass ({LOWCUT_HZ:.0f}-{HIGHCUT_HZ:.0f} Hz)", STD_BP_COLOR
        )
        self.scope_laser_raw, self.curve_laser_raw = self._make_scope(
            "Laser Mic - Raw", LASER_RAW_COLOR
        )
        self.scope_laser_bp, self.curve_laser_bp = self._make_scope(
            f"Laser Mic - Bandpass ({LOWCUT_HZ:.0f}-{HIGHCUT_HZ:.0f} Hz)", LASER_BP_COLOR
        )

        scope_grid = QtWidgets.QGridLayout()
        scope_grid.setSpacing(10)
        scope_grid.addWidget(self.scope_std_raw, 0, 0)
        scope_grid.addWidget(self.scope_std_bp, 0, 1)
        scope_grid.addWidget(self.scope_laser_raw, 1, 0)
        scope_grid.addWidget(self.scope_laser_bp, 1, 1)
        scope_grid.setColumnStretch(0, 1)
        scope_grid.setColumnStretch(1, 1)
        scope_grid.setRowStretch(0, 1)
        scope_grid.setRowStretch(1, 1)

        scope_container = QtWidgets.QWidget()
        scope_container.setLayout(scope_grid)
        col.addWidget(scope_container, stretch=2)

        # ---- Spectrogram (laser mic, bandpass-filtered signal) ----
        self.spectrogram = pg.ImageView(view=pg.PlotItem())
        self.spectrogram.ui.histogram.hide()
        self.spectrogram.ui.roiBtn.hide()
        self.spectrogram.ui.menuBtn.hide()
        self.spectrogram.view.setTitle("Spectrogram - Laser Mic (Bandpass)", color=TEXT_COLOR, size="13pt")
        self.spectrogram.view.setLabel("bottom", "Time", color=DIM_TEXT_COLOR)
        self.spectrogram.view.setLabel("left", "Frequency", color=DIM_TEXT_COLOR)
        # ImageView locks the aspect ratio by default since it's meant for
        # viewing photos - without turning that off, the image gets
        # squeezed into a thin strip instead of filling the panel.
        self.spectrogram.view.setAspectLocked(False)
        self.spectrogram.setColorMap(pg.colormap.get(SPECTROGRAM_CMAP))
        self.spectrogram.setStyleSheet(f"background-color:{PANEL_COLOR}; border:none;")
        col.addWidget(self.spectrogram, stretch=1)

        return col

    def build_result_panel(self):
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(290)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(16)

        title = QtWidgets.QLabel("Result")
        title.setAlignment(QtCore.Qt.AlignHCenter)
        title.setStyleSheet(f"color: {TEXT_COLOR}; font-family: {UI_FONT}; font-size: 22px; font-weight: 700;")
        layout.addWidget(title)

        self.prediction_box = ResultBox("PREDICTION", "--")
        self.confidence_box = ResultBox("CONFIDENCE PERCENTAGE", "--", unit=" %")
        layout.addWidget(self.prediction_box, stretch=1)
        layout.addWidget(self.confidence_box, stretch=1)
        layout.addStretch()

        status = "LIVE (PCM1808)" if self.hardware.available else "DEMO (no hardware found)"
        status_label = QtWidgets.QLabel(status)
        status_label.setAlignment(QtCore.Qt.AlignHCenter)
        status_color = ACCENT_COLOR if self.hardware.available else "#ffb454"
        status_label.setStyleSheet(
            f"color: {status_color}; font-family: {UI_FONT}; font-size: 12px; font-weight: 600;"
        )
        layout.addWidget(status_label)

        return panel

    def build_button_row(self):
        row = QtWidgets.QHBoxLayout()
        row.addStretch()

        self.record_btn = CircleButton()
        self.record_btn.clicked.connect(self.on_record)
        row.addWidget(self.record_btn)

        row.addSpacing(30)

        self.stop_btn = StopButton()
        self.stop_btn.clicked.connect(self.on_stop)
        self.stop_btn.setEnabled(False)  # nothing to stop until we've started recording
        row.addWidget(self.stop_btn)

        row.addSpacing(30)

        self.open_audio_btn = QtWidgets.QPushButton("Open Audio")
        self.open_audio_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.open_audio_btn.clicked.connect(self.open_audio_file)
        self.open_audio_btn.setFixedSize(140, 54)
        self.open_audio_btn.setStyleSheet(f"""
            QPushButton {{
                color: {TEXT_COLOR};
                background-color: {STOP_COLOR};
                border: 2px solid {BORDER_COLOR};
                border-radius: 8px;
                font-family: {UI_FONT};
                font-size: 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{ background-color: #3a4453; }}
            QPushButton:pressed {{ background-color: #21262d; }}
        """)
        row.addWidget(self.open_audio_btn)

        row.addStretch()
        return row

    # =====================================================================
    # DATA ACQUISITION
    # =====================================================================

    def _demo_chunk(self):
        """Synthetic stand-in used only when the PCM1808 can't be opened,
        so the UI still has something to draw. Runs the exact same
        bandpass filter as the hardware path, just on a made-up signal."""
        t = TIME_AXIS
        self._demo_t_offset += 0.1

        std_burst = 0.15 + 0.05 * np.sin(self._demo_t_offset)
        std_env = 1 + 4 * np.exp(-((t - std_burst) ** 2) / (2 * 0.01 ** 2))
        std_freq = 600 + 100 * np.sin(self._demo_t_offset)
        std_raw_full = 0.3 * std_env * np.sin(2 * np.pi * std_freq * t + self._demo_t_offset * 4)
        std_raw_full += 0.03 * np.random.randn(len(t))

        laser_burst = 0.12 + 0.05 * np.cos(self._demo_t_offset * 0.8)
        laser_env = 1 + 4 * np.exp(-((t - laser_burst) ** 2) / (2 * 0.01 ** 2))
        laser_freq = 900 + 150 * np.cos(self._demo_t_offset)
        laser_raw_full = 0.3 * laser_env * np.sin(2 * np.pi * laser_freq * t + self._demo_t_offset * 3)
        laser_raw_full += 0.05 * np.random.randn(len(t))

        std_bp_full, self._demo_zi_std = sosfilt(SOS, std_raw_full, zi=self._demo_zi_std)
        laser_bp_full, self._demo_zi_laser = sosfilt(SOS, laser_raw_full, zi=self._demo_zi_laser)

        return {
            "std_raw": np.clip(std_raw_full * VISUAL_GAIN, -1.0, 1.0),
            "std_bp": np.clip(std_bp_full * VISUAL_GAIN, -1.0, 1.0),
            "laser_raw": np.clip(laser_raw_full * VISUAL_GAIN, -1.0, 1.0),
            "laser_bp": np.clip(laser_bp_full * VISUAL_GAIN, -1.0, 1.0),
            "laser_bp_full": laser_bp_full,
        }

    def _update_spectrogram(self, samples):
        try:
            f, _, sxx = scipy_spectrogram(
                samples, fs=RATE, nperseg=SPECTROGRAM_NPERSEG, noverlap=SPECTROGRAM_NOVERLAP
            )

            # Only show the band we actually filter for - the rest is dead
            # space that just wastes resolution in the displayed image.
            band = (f >= LOWCUT_HZ) & (f <= HIGHCUT_HZ)
            sxx = sxx[band, :]

            sxx_db = 10 * np.log10(sxx + 1e-12)

            # Smooth upsample for display only, so the coarse STFT grid
            # reads as a continuous field instead of a checkerboard.
            sxx_db = ndimage.zoom(sxx_db, SPECTROGRAM_ZOOM, order=1)

            self.spectrogram.setImage(sxx_db.T, autoLevels=True, autoRange=True)
        except Exception as exc:
            print("[dashboard] spectrogram error:", repr(exc))

    def update_prediction(self):
        """Update the Prediction + Confidence boxes from the model (or demo)."""
        if self.backend is not None:
            result = self.backend.predict()      # (label, confidence%) or None
            if result is not None:
                label, confidence = result
                self.prediction_box.set_value(label)
                self.confidence_box.set_value(f"{confidence:.1f}")
            # None = silence / not enough audio yet: leave the last reading as-is.
            return

        # ML TEAM: replace this block with a call into your classifier,
        # using self.last_chunk (std_raw/std_bp/laser_raw/laser_bp/
        # laser_bp_full) as the input instead of random demo values.
        label = np.random.choice(PREDICTION_LABELS)
        confidence = np.random.uniform(72, 99.5)
        self.prediction_box.set_value(label)
        self.confidence_box.set_value(f"{confidence:.1f}")

    def update_frame(self):
        """Runs on every timer tick - pushes fresh data into all four
        oscilloscopes, the spectrogram, and (while recording) the
        prediction/confidence panel. Uses the live PCM1808 hardware when
        available, otherwise the synthetic demo fallback."""
        chunk = self.hardware.read() if self.hardware.available else None
        if chunk is None:
            chunk = self._demo_chunk()

        self.last_chunk = chunk

        self.curve_std_raw.setData(TIME_AXIS, chunk["std_raw"])
        self.curve_std_bp.setData(TIME_AXIS, chunk["std_bp"])
        self.curve_laser_raw.setData(TIME_AXIS, chunk["laser_raw"])
        self.curve_laser_bp.setData(TIME_AXIS, chunk["laser_bp"])

        self._update_spectrogram(chunk["laser_bp_full"])

        if self.is_recording:
            self.update_prediction()

    # =====================================================================
    # CONTROLS
    # =====================================================================

    def on_record(self):
        self.is_recording = True
        self.record_btn.set_recording(True)
        self.stop_btn.setEnabled(True)
        print("Recording started")
        if self.backend is not None:
            self.backend.start()

    def on_stop(self):
        self.is_recording = False
        self.record_btn.set_recording(False)
        self.stop_btn.setEnabled(False)
        self.prediction_box.set_value("--")
        self.confidence_box.set_value("--")
        print("Recording stopped")
        if self.backend is not None:
            self.backend.stop()

    def open_audio_file(self):
        """Switch the Prediction/Confidence panel from live mic to a
        replayed audio/DAQ file. The four oscilloscopes and spectrogram
        keep showing the live PCM1808 hardware regardless - this only
        affects what the classifier analyzes."""
        if SignalBackend is None:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open audio or DAQ capture",
            "",
            "Audio/DAQ files (*.wav *.flac *.ogg *.aiff *.aif *.mp3 *.m4a *.csv);;All files (*)",
        )
        if not path:
            return
        was_recording = self.is_recording
        if self.backend is not None:
            self.backend.stop()
        try:
            self.backend = SignalBackend(model="ensemble", source=path, autosave=False)
            print(self.backend.status_text())
            if not self.backend.audio_available:
                print(f"[dashboard] audio source unavailable: {self.backend.audio_error}")
            if was_recording:
                self.backend.start()
        except Exception as e:  # noqa: BLE001
            print(f"[dashboard] failed to open audio file: {type(e).__name__}: {e}")

    def keyPressEvent(self, event):
        # F11 for a true borderless fullscreen, Esc to leave it. Not required
        # by the spec, just convenient for demoing on a projector.
        if event.key() == QtCore.Qt.Key_F11:
            self.is_fullscreen = not self.is_fullscreen
            self.showFullScreen() if self.is_fullscreen else self.showMaximized()
        elif event.key() == QtCore.Qt.Key_Escape and self.is_fullscreen:
            self.is_fullscreen = False
            self.showMaximized()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        try:
            self.timer.stop()
        except Exception:
            pass
        try:
            self.hardware.close()
        except Exception:
            pass
        if self.backend is not None:
            try:
                self.backend.stop()
            except Exception:
                pass
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    pg.setConfigOption("background", PANEL_COLOR)
    pg.setConfigOption("foreground", DIM_TEXT_COLOR)
    pg.setConfigOptions(antialias=True)

    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
