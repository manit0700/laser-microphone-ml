"""
Signal Dashboard - UI shell for the signal / ML project

This is just the UI layer. Oscilloscope and spectrogram are both showing
dummy/fake data right now, and the Prediction + Confidence panel just rolls
random values. Nothing here does real signal processing or classification.

Every spot where the real backend needs to plug in is marked with a comment
starting with "ML TEAM:" - search for that and you'll find all of them
(waveform acquisition, spectrogram generation, classifier prediction, and
the band-pass filter, which right now is just a UI switch that prints its
state and doesn't touch the signal at all).

How to run:
    pip install PyQt5 pyqtgraph numpy
    python signal_dashboard.py

Needs an actual display (Windows/Mac/Linux desktop) - this won't draw
anything on a headless server.

If you're using PySide6 instead of PyQt5, change the import at the top,
change app.exec_() to app.exec() at the bottom, and swap QtCore.pyqtProperty
for QtCore.Property in the ToggleSwitch class. Everything else is the same.
"""

import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

# Real ML backend (mic capture + trained model). Optional: if it can't be
# imported (missing deps, etc.) the dashboard still runs on its demo data.
try:
    from signal_backend import SignalBackend
except Exception as _e:  # noqa: BLE001
    SignalBackend = None
    print(f"[dashboard] ML backend unavailable, using demo data ({_e})")


# Colors and fonts live here so the whole theme can be tweaked in one place
# instead of hunting through every widget.
BG_COLOR = "#0d1117"        # window background
PANEL_COLOR = "#12161c"     # graph / box background
BORDER_COLOR = "#2a2f3a"    # borders, grid lines
TEXT_COLOR = "#c9d1d9"      # main text
DIM_TEXT_COLOR = "#7d8590"  # secondary text / axis labels
ACCENT_COLOR = "#39ff88"    # scope trace + result values (green, oscilloscope-y)
RECORD_COLOR = "#ff453a"    # record button red
STOP_COLOR = "#2d333b"      # stop button body

UI_FONT = "Segoe UI, Helvetica Neue, Arial, sans-serif"
MONO_FONT = "Consolas, JetBrains Mono, monospace"  # used for the numeric readouts

SPECTROGRAM_CMAP = "inferno"

# Placeholder prediction classes - swap this list for whatever your actual
# classifier outputs.
PREDICTION_LABELS = [str(d) for d in range(10)] + ["Unknown"]


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
        painter.drawRoundedRect(QtCore.QRectF(cx - gap/2 - bar_w, cy - bar_h/2, bar_w, bar_h), 2, 2)
        painter.drawRoundedRect(QtCore.QRectF(cx + gap/2, cy - bar_h/2, bar_w, bar_h), 2, 2)


class ToggleSwitch(QtWidgets.QCheckBox):
    """A little on/off slider switch, used for the Band-Pass Filter control.

    It's really just a QCheckBox under the hood (isChecked(), toggled
    signal, all of that still works normally) - it just paints itself as a
    sliding pill instead of the default checkbox square.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(50, 26)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self._thumb_x = 3.0

        self.slide_anim = QtCore.QPropertyAnimation(self, b"thumb_x", self)
        self.slide_anim.setDuration(150)
        self.toggled.connect(self._animate_to_state)

    def _animate_to_state(self, checked):
        thumb_d = self.height() - 6
        end_x = (self.width() - thumb_d - 3) if checked else 3
        self.slide_anim.stop()
        self.slide_anim.setStartValue(self._thumb_x)
        self.slide_anim.setEndValue(end_x)
        self.slide_anim.start()

    # thumb_x needs to be a real Qt property (not just a plain attribute)
    # so QPropertyAnimation is able to animate it
    def _get_thumb_x(self):
        return self._thumb_x

    def _set_thumb_x(self, value):
        self._thumb_x = value
        self.update()

    thumb_x = QtCore.pyqtProperty(float, _get_thumb_x, _set_thumb_x)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtCore.Qt.NoPen)

        track_color = ACCENT_COLOR if self.isChecked() else BORDER_COLOR
        painter.setBrush(QtGui.QColor(track_color))
        painter.drawRoundedRect(0, 0, self.width(), self.height(), self.height() / 2, self.height() / 2)

        thumb_d = self.height() - 6
        painter.setBrush(QtGui.QColor("#f0f0f0"))
        painter.drawEllipse(QtCore.QRectF(self._thumb_x, 3, thumb_d, thumb_d))


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
            font-size: 36px;
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
        self.setWindowTitle("Signal Dashboard")
        self.resize(1400, 880)
        self.setStyleSheet(f"QMainWindow {{ background-color: {BG_COLOR}; }}")

        self.is_recording = False
        self.is_fullscreen = False

        # state for the fake waveform generator, see make_fake_waveform()
        self.sample_rate = 2000
        self.chunk_len = 800
        self.time_axis = np.linspace(0, self.chunk_len / self.sample_rate, self.chunk_len)
        self.t_offset = 0.0

        # Try to bring up the real ML backend (mic + trained model). If anything
        # is missing it stays None and every hook below falls back to demo data,
        # so the UI always runs.
        self.backend = None
        if SignalBackend is not None:
            try:
                self.backend = SignalBackend()
                print(self.backend.status_text())
            except Exception as e:  # noqa: BLE001
                print(f"[dashboard] backend init failed, using demo data: {e}")
                self.backend = None

        self.build_ui()

        # Make the real filter match the Band-Pass switch's initial state so the
        # UI and the signal agree from the start (the switch defaults to OFF).
        if self.backend is not None:
            self.backend.set_bandpass(self.bandpass_switch.isChecked())

        # this timer is what makes the dummy data feel "live" - swap the
        # dummy generators for real data and everything downstream of them
        # (the graphs, the timer, all of it) keeps working the same way
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)  # ~20 updates/sec, just for the demo animation

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

    def build_graphs_column(self):
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(10)

        # Band-pass filter toggle lives here since it acts on the incoming
        # signal that's about to get plotted below, not on the ML result.
        filter_row = QtWidgets.QHBoxLayout()
        filter_label = QtWidgets.QLabel("Band-Pass Filter")
        filter_label.setStyleSheet(f"color: {DIM_TEXT_COLOR}; font-family: {UI_FONT}; font-size: 13px;")
        self.bandpass_switch = ToggleSwitch()
        # Default OFF: the current model is trained WITHOUT the band-pass filter
        # (ENABLE_FILTER=False), so inference should be unfiltered to match. If a
        # future model is trained with the filter, default this ON instead.
        self.bandpass_switch.toggled.connect(self.on_bandpass_toggled)
        filter_row.addWidget(filter_label)
        filter_row.addSpacing(10)
        filter_row.addWidget(self.bandpass_switch)
        filter_row.addStretch()
        col.addLayout(filter_row)

        # ---- Oscilloscope ----
        self.scope_plot = pg.PlotWidget()
        self.scope_plot.setBackground(PANEL_COLOR)
        self.scope_plot.setTitle("Oscilloscope", color=TEXT_COLOR, size="13pt")
        self.scope_plot.showGrid(x=True, y=True, alpha=0.25)
        self.scope_plot.setLabel("bottom", "Time", units="s", color=DIM_TEXT_COLOR)
        self.scope_plot.setLabel("left", "Amplitude", color=DIM_TEXT_COLOR)
        self.scope_plot.getAxis("bottom").setPen(BORDER_COLOR)
        self.scope_plot.getAxis("left").setPen(BORDER_COLOR)
        self.scope_plot.getAxis("bottom").setTextPen(DIM_TEXT_COLOR)
        self.scope_plot.getAxis("left").setTextPen(DIM_TEXT_COLOR)
        self.scope_curve = self.scope_plot.plot(pen=pg.mkPen(color=ACCENT_COLOR, width=1.5))
        col.addWidget(self.scope_plot, stretch=1)

        # ---- Spectrogram ----
        self.spectrogram = pg.ImageView(view=pg.PlotItem())
        self.spectrogram.ui.histogram.hide()
        self.spectrogram.ui.roiBtn.hide()
        self.spectrogram.ui.menuBtn.hide()
        self.spectrogram.view.setTitle("Spectrogram", color=TEXT_COLOR, size="13pt")
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
        panel.setFixedWidth(250)
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

        row.addStretch()
        return row

    def make_fake_waveform(self):
        # ML TEAM: insert waveform acquisition here
        #
        # Everything below is a made-up signal (a little burst that fades
        # in and out, plus noise) just so the oscilloscope has something to
        # draw. Replace this whole method with real samples from your ADC,
        # microphone, or sensor - just keep returning a 1D numpy array the
        # same length as self.time_axis and the plot will keep working.
        t = self.time_axis
        burst_center = 0.15 + 0.05 * np.sin(self.t_offset)
        envelope = 1 + 4 * np.exp(-((t - burst_center) ** 2) / (2 * 0.03 ** 2))
        freq = 60 + 10 * np.sin(self.t_offset)
        wave = envelope * np.sin(2 * np.pi * freq * t + self.t_offset * 4)
        wave += 0.05 * np.random.randn(len(t))
        self.t_offset += 0.1
        return wave

    def make_fake_spectrogram(self):
        # ML TEAM: insert spectrogram generation here
        #
        # In the real pipeline this would be an STFT (or similar) run on
        # the acquired signal. For now it's just random noise so the
        # heatmap has something to show - swap this out for a real
        # (freq_bins x time_bins) array and setImage() below will handle it.
        return np.random.rand(40, 60)

    def _backend_audio_live(self):
        """True when the real backend is present and actually capturing mic audio."""
        return self.backend is not None and self.backend.audio_available and self.backend.running

    def get_scope_data(self):
        """(samples, time_axis) for the oscilloscope - real mic if available, else demo."""
        if self._backend_audio_live():
            samples = self.backend.scope_samples()
            if samples is not None and len(samples) > 1:
                t = np.linspace(0, len(samples) / self.backend.sample_rate, len(samples))
                return samples, t
        return self.make_fake_waveform(), self.time_axis

    def get_spectrogram_data(self):
        """(freq x time) image for the spectrogram - real STFT if available, else demo."""
        if self._backend_audio_live():
            spec = self.backend.spectrogram()
            if spec is not None:
                return spec
        return self.make_fake_spectrogram()

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

        # Demo fallback: roll a random label + confidence so the panel has
        # something to display when no trained backend is connected.
        label = np.random.choice(PREDICTION_LABELS)
        confidence = np.random.uniform(72, 99.5)
        self.prediction_box.set_value(label)
        self.confidence_box.set_value(f"{confidence:.1f}")

    def update_frame(self):
        """Runs on every timer tick - pushes fresh data into all three panels
        (scope, spectrogram, and prediction/confidence). Uses the real ML
        backend when connected, otherwise the built-in demo generators."""
        wave, t = self.get_scope_data()
        self.scope_curve.setData(t, wave)

        spec_data = self.get_spectrogram_data()
        self.spectrogram.setImage(spec_data, autoLevels=True, autoRange=True)

        if self.is_recording:
            self.update_prediction()

    def on_record(self):
        self.is_recording = True
        self.record_btn.set_recording(True)
        self.stop_btn.setEnabled(True)
        print("Recording started")
        # Start real mic capture + streaming inference when the backend is present.
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

    def on_bandpass_toggled(self, checked):
        state = "ON" if checked else "OFF"
        print(f"Band-pass filter: {state}")
        # Apply the real band-pass + pre-emphasis filter (preprocess.reduce_noise)
        # so the toggle actually changes the signal the scope and model see.
        if self.backend is not None:
            self.backend.set_bandpass(checked)

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