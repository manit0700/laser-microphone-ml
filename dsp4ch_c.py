import sys
import numpy as np
import pyaudio

from scipy.signal import butter, sosfilt, sosfilt_zi

# High-speed graphics
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg


# =============================================================================
# CONFIGURATION
# =============================================================================

FORMAT = pyaudio.paInt32
CHANNELS = 2
RATE = 48000

# Keep the exact working buffer size.
CHUNK = 4096 * 3

# No downsampling.
COMPRESSION_FACTOR = 1

# Visual-only gain.
VISUAL_GAIN = 2.0
#VISUAL_GAIN = 3.0


# =============================================================================
# BANDPASS FILTER
# =============================================================================

FILTER_ORDER = 6

LOWCUT_HZ = 200.0
HIGHCUT_HZ = 8000.0
#HIGHCUT_HZ = 5000.0

NYQUIST = RATE * 0.5

Wn_low = LOWCUT_HZ / NYQUIST
Wn_high = HIGHCUT_HZ / NYQUIST

SOS = butter(
    FILTER_ORDER,
    [Wn_low, Wn_high],
    btype="bandpass",
    output="sos"
)


# =============================================================================
# DISPLAY
# =============================================================================

PLOT_POINTS = CHUNK // COMPRESSION_FACTOR

x = np.arange(
    PLOT_POINTS,
    dtype=np.float64
)


# =============================================================================
# OSCILLOSCOPE WINDOW
# =============================================================================

class JetsonOscilloscope(QtWidgets.QMainWindow):

    def __init__(self, stream):

        super().__init__()

        self.stream = stream

        # ---------------------------------------------------------------------
        # Continuous filter state
        #
        # Each stereo channel has its own independent filter state.
        # ---------------------------------------------------------------------

        self.zi_left = sosfilt_zi(SOS) * 0.0
        self.zi_right = sosfilt_zi(SOS) * 0.0

        # ---------------------------------------------------------------------
        # Window
        # ---------------------------------------------------------------------

        self.setWindowTitle(
            "Jetson Orin Nano Super - "
            "PCM1808 Raw vs 200-8000 Hz Bandpass"
        )

        self.resize(
            1400,
            850
        )

        # ---------------------------------------------------------------------
        # Central widget
        # ---------------------------------------------------------------------

        central = QtWidgets.QWidget()

        self.setCentralWidget(
            central
        )

        layout = QtWidgets.QVBoxLayout(
            central
        )

        layout.setContentsMargins(
            8,
            8,
            8,
            8
        )

        # ---------------------------------------------------------------------
        # Graphics layout
        #
        # We explicitly use a 2 x 2 matrix:
        #
        #             COLUMN 1             COLUMN 2
        #
        # ROW 1       LEFT RAW             LEFT FILTERED
        # ROW 2       RIGHT RAW            RIGHT FILTERED
        # ---------------------------------------------------------------------

        self.canvas = pg.GraphicsLayoutWidget()

        self.canvas.setBackground(
            "k"
        )

        layout.addWidget(
            self.canvas
        )

        # =====================================================================
        # ROW 1 / COLUMN 1
        # CHANNEL 1 LEFT - RAW
        # =====================================================================

        self.p1 = self.canvas.addPlot(
            row=0,
            col=0,
            title=(
                "<span style='color:#00ffff; "
                "font-size:11pt;'>"
                "CHANNEL 1 - LEFT - RAW"
                "</span>"
                "<br>"
                "<span style='color:#aaaaaa; "
                "font-size:9pt;'>"
                "Before Bandpass Filter"
                "</span>"
            )
        )

        self.p1.setYRange(
            -1.0,
            1.0,
            padding=0
        )

        self.p1.setXRange(
            0,
            PLOT_POINTS - 1,
            padding=0
        )

        self.p1.showGrid(
            x=True,
            y=True,
            alpha=0.30
        )

        self.p1.setLabel(
            "left",
            "Amplitude"
        )

        self.p1.setLabel(
            "bottom",
            "Sample"
        )

        self.p1.setMinimumHeight(
            300
        )

        self.raw_curve1 = self.p1.plot(
            pen=pg.mkPen(
                color="#00ffff",
                width=2
            )
        )

        # =====================================================================
        # ROW 1 / COLUMN 2
        # CHANNEL 1 LEFT - FILTERED
        # =====================================================================

        self.p2 = self.canvas.addPlot(
            row=0,
            col=1,
            title=(
                "<span style='color:#00ff80; "
                "font-size:11pt;'>"
                "CHANNEL 1 - LEFT - BANDPASS OUTPUT"
                "</span>"
                "<br>"
                "<span style='color:#aaaaaa; "
                "font-size:9pt;'>"
                f"{LOWCUT_HZ:.0f} - {HIGHCUT_HZ:.0f} Hz"
                "</span>"
            )
        )

        self.p2.setYRange(
            -1.0,
            1.0,
            padding=0
        )

        self.p2.setXRange(
            0,
            PLOT_POINTS - 1,
            padding=0
        )

        self.p2.showGrid(
            x=True,
            y=True,
            alpha=0.30
        )

        self.p2.setLabel(
            "left",
            "Amplitude"
        )

        self.p2.setLabel(
            "bottom",
            "Sample"
        )

        self.p2.setMinimumHeight(
            300
        )

        self.filt_curve1 = self.p2.plot(
            pen=pg.mkPen(
                color="#00ff80",
                width=2
            )
        )

        # =====================================================================
        # ROW 2 / COLUMN 1
        # CHANNEL 2 RIGHT - RAW
        # =====================================================================

        self.p3 = self.canvas.addPlot(
            row=1,
            col=0,
            title=(
                "<span style='color:#ffff00; "
                "font-size:11pt;'>"
                "CHANNEL 2 - RIGHT - RAW"
                "</span>"
                "<br>"
                "<span style='color:#aaaaaa; "
                "font-size:9pt;'>"
                "Before Bandpass Filter"
                "</span>"
            )
        )

        self.p3.setYRange(
            -1.0,
            1.0,
            padding=0
        )

        self.p3.setXRange(
            0,
            PLOT_POINTS - 1,
            padding=0
        )

        self.p3.showGrid(
            x=True,
            y=True,
            alpha=0.30
        )

        self.p3.setLabel(
            "left",
            "Amplitude"
        )

        self.p3.setLabel(
            "bottom",
            "Sample"
        )

        self.p3.setMinimumHeight(
            300
        )

        self.raw_curve2 = self.p3.plot(
            pen=pg.mkPen(
                color="#ffff00",
                width=2
            )
        )

        # =====================================================================
        # ROW 2 / COLUMN 2
        # CHANNEL 2 RIGHT - FILTERED
        # =====================================================================

        self.p4 = self.canvas.addPlot(
            row=1,
            col=1,
            title=(
                "<span style='color:#ff8000; "
                "font-size:11pt;'>"
                "CHANNEL 2 - RIGHT - BANDPASS OUTPUT"
                "</span>"
                "<br>"
                "<span style='color:#aaaaaa; "
                "font-size:9pt;'>"
                f"{LOWCUT_HZ:.0f} - {HIGHCUT_HZ:.0f} Hz"
                "</span>"
            )
        )

        self.p4.setYRange(
            -1.0,
            1.0,
            padding=0
        )

        self.p4.setXRange(
            0,
            PLOT_POINTS - 1,
            padding=0
        )

        self.p4.showGrid(
            x=True,
            y=True,
            alpha=0.30
        )

        self.p4.setLabel(
            "left",
            "Amplitude"
        )

        self.p4.setLabel(
            "bottom",
            "Sample"
        )

        self.p4.setMinimumHeight(
            300
        )

        self.filt_curve2 = self.p4.plot(
            pen=pg.mkPen(
                color="#ff8000",
                width=2
            )
        )

        # ---------------------------------------------------------------------
        # Make the columns equal width.
        # ---------------------------------------------------------------------

        self.canvas.ci.layout.setColumnStretchFactor(
            0,
            1
        )

        self.canvas.ci.layout.setColumnStretchFactor(
            1,
            1
        )

        # ---------------------------------------------------------------------
        # Status bar
        # ---------------------------------------------------------------------

        self.statusBar().showMessage(
            "Starting..."
        )

        # ---------------------------------------------------------------------
        # GUI refresh timer
        # ---------------------------------------------------------------------

        self.timer = QtCore.QTimer()

        self.timer.timeout.connect(
            self.update_waveforms
        )

        self.timer.start(
            40
        )

        self.update_counter = 0

    # =========================================================================
    # TRIGGER
    # =========================================================================

    @staticmethod
    def trigger_signal(
        wave_data,
        trigger_source
    ):
        """
        Find the first rising-edge zero crossing in the filtered signal.

        The trigger index is then used to extract the same time section
        from both the raw and filtered signals.
        """

        trigger_level = 0.0

        if len(trigger_source) < 2:
            return None

        crossings = np.where(
            (
                trigger_source[:-1]
                <= trigger_level
            )
            &
            (
                trigger_source[1:]
                > trigger_level
            )
        )[0]

        if len(crossings) == 0:
            return None

        trigger_index = int(
            crossings[0]
        )

        display_data = wave_data[
            trigger_index:
            trigger_index + PLOT_POINTS
        ]

        # Pad if necessary.
        if len(display_data) < PLOT_POINTS:

            display_data = np.pad(
                display_data,
                (
                    0,
                    PLOT_POINTS - len(display_data)
                ),
                mode="constant"
            )

        return np.clip(
            display_data,
            -1.0,
            1.0
        )

    # =========================================================================
    # UPDATE WAVEFORMS
    # =========================================================================

    def update_waveforms(self):

        try:

            # -----------------------------------------------------------------
            # READ AUDIO
            # -----------------------------------------------------------------

            raw_data = self.stream.read(
                CHUNK,
                exception_on_overflow=False
            )

            data_ints = np.frombuffer(
                raw_data,
                dtype=np.int32
            )

            expected_values = (
                CHUNK * CHANNELS
            )

            if len(data_ints) != expected_values:
                return

            # -----------------------------------------------------------------
            # DEMULTIPLEX STEREO
            # -----------------------------------------------------------------

            left_raw = data_ints[0::2]
            right_raw = data_ints[1::2]

            if (
                len(left_raw) != CHUNK
                or
                len(right_raw) != CHUNK
            ):
                return

            # -----------------------------------------------------------------
            # OPTIONAL COMPRESSION
            #
            # Currently = 1, so every sample is retained.
            # -----------------------------------------------------------------

            left_comp = left_raw[
                ::COMPRESSION_FACTOR
            ]

            right_comp = right_raw[
                ::COMPRESSION_FACTOR
            ]

            # -----------------------------------------------------------------
            # NORMALIZE
            # -----------------------------------------------------------------

            left_norm = (
                left_comp.astype(np.float64)
                / 2147483648.0
            )

            right_norm = (
                right_comp.astype(np.float64)
                / 2147483648.0
            )

            # -----------------------------------------------------------------
            # RAW SIGNAL
            #
            # This is BEFORE the bandpass filter.
            # -----------------------------------------------------------------

            left_raw_signal = left_norm.copy()
            right_raw_signal = right_norm.copy()

            # -----------------------------------------------------------------
            # BANDPASS FILTER
            #
            # 6th-order Butterworth
            # 200 Hz - 5000 Hz
            #
            # The filter state continues from the previous block.
            # -----------------------------------------------------------------

            filtered_left, self.zi_left = sosfilt(
                SOS,
                left_norm,
                zi=self.zi_left
            )

            filtered_right, self.zi_right = sosfilt(
                SOS,
                right_norm,
                zi=self.zi_right
            )

            # -----------------------------------------------------------------
            # LEFT CHANNEL TRIGGER
            #
            # Filtered LEFT determines the trigger position.
            # The same trigger is used for both LEFT plots.
            # -----------------------------------------------------------------

            left_raw_display = self.trigger_signal(
                left_raw_signal,
                filtered_left
            )

            left_filtered_display = self.trigger_signal(
                filtered_left,
                filtered_left
            )

            # -----------------------------------------------------------------
            # RIGHT CHANNEL TRIGGER
            #
            # Filtered RIGHT determines the trigger position.
            # -----------------------------------------------------------------

            right_raw_display = self.trigger_signal(
                right_raw_signal,
                filtered_right
            )

            right_filtered_display = self.trigger_signal(
                filtered_right,
                filtered_right
            )

            # -----------------------------------------------------------------
            # No valid trigger.
            # -----------------------------------------------------------------

            if left_raw_display is None:
                return

            if left_filtered_display is None:
                return

            if right_raw_display is None:
                return

            if right_filtered_display is None:
                return

            # -----------------------------------------------------------------
            # VISUAL GAIN
            #
            # This happens AFTER filtering and therefore does not affect
            # the DSP operation.
            # -----------------------------------------------------------------

            left_raw_display = np.clip(
                left_raw_display * VISUAL_GAIN,
                -1.0,
                1.0
            )

            left_filtered_display = np.clip(
                left_filtered_display * VISUAL_GAIN,
                -1.0,
                1.0
            )

            right_raw_display = np.clip(
                right_raw_display * VISUAL_GAIN,
                -1.0,
                1.0
            )

            right_filtered_display = np.clip(
                right_filtered_display * VISUAL_GAIN,
                -1.0,
                1.0
            )

            # -----------------------------------------------------------------
            # UPDATE FOUR PLOTS
            # -----------------------------------------------------------------

            # Top-left:
            # LEFT RAW
            self.raw_curve1.setData(
                x,
                left_raw_display
            )

            # Top-right:
            # LEFT FILTERED
            self.filt_curve1.setData(
                x,
                left_filtered_display
            )

            # Bottom-left:
            # RIGHT RAW
            self.raw_curve2.setData(
                x,
                right_raw_display
            )

            # Bottom-right:
            # RIGHT FILTERED
            self.filt_curve2.setData(
                x,
                right_filtered_display
            )

            # -----------------------------------------------------------------
            # STATUS
            # -----------------------------------------------------------------

            self.update_counter += 1

            if self.update_counter % 25 == 0:

                self.statusBar().showMessage(
                    f"LIVE | "
                    f"{RATE / 1000:.0f} kHz | "
                    f"{PLOT_POINTS} samples | "
                    f"Bandpass: "
                    f"{LOWCUT_HZ:.0f}-"
                    f"{HIGHCUT_HZ:.0f} Hz"
                )

        except Exception as exc:

            # Print errors instead of silently hiding them.
            print(
                "Waveform update error:",
                repr(exc)
            )

    # =========================================================================
    # CLEANUP
    # =========================================================================

    def closeEvent(self, event):

        try:
            self.timer.stop()
        except Exception:
            pass

        event.accept()


# =============================================================================
# FIND AUDIO DEVICE
# =============================================================================

def find_audio_device(p):

    print()
    print("=" * 80)
    print("AVAILABLE AUDIO DEVICES")
    print("=" * 80)

    selected_device = None

    for i in range(
        p.get_device_count()
    ):

        try:

            dev_info = (
                p.get_device_info_by_index(i)
            )

            name = dev_info.get(
                "name",
                ""
            )

            max_input = dev_info.get(
                "maxInputChannels",
                0
            )

            default_rate = dev_info.get(
                "defaultSampleRate",
                0
            )

            print(
                f"[{i}] {name}"
            )

            print(
                f"     Input channels: "
                f"{max_input}"
            )

            print(
                f"     Default rate: "
                f"{default_rate}"
            )

            name_upper = name.upper()

            # Preferred PCM1808 device.
            if (
                selected_device is None
                and (
                    "APE" in name_upper
                    or "HW:1,0" in name_upper
                )
                and max_input >= CHANNELS
            ):

                selected_device = i

        except Exception as exc:

            print(
                f"Could not inspect device {i}: "
                f"{exc}"
            )

    print("=" * 80)

    # -------------------------------------------------------------------------
    # Preferred device found.
    # -------------------------------------------------------------------------

    if selected_device is not None:

        print(
            f"Selected PCM1808 device: "
            f"{selected_device}"
        )

        return selected_device

    # -------------------------------------------------------------------------
    # Preserve your known-working fallback to device 1.
    # -------------------------------------------------------------------------

    if p.get_device_count() > 1:

        try:

            info = (
                p.get_device_info_by_index(1)
            )

            if info.get(
                "maxInputChannels",
                0
            ) >= CHANNELS:

                print(
                    "APE/hw:1,0 device not identified "
                    "by name."
                )

                print(
                    "Using fallback device index 1."
                )

                return 1

        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Last resort: first stereo input.
    # -------------------------------------------------------------------------

    for i in range(
        p.get_device_count()
    ):

        try:

            info = (
                p.get_device_info_by_index(i)
            )

            if info.get(
                "maxInputChannels",
                0
            ) >= CHANNELS:

                print(
                    f"Using first available stereo "
                    f"input device: {i}"
                )

                return i

        except Exception:
            pass

    return None


# =============================================================================
# MAIN
# =============================================================================

def main():

    p = pyaudio.PyAudio()

    stream = None
    scope = None

    try:

        # ---------------------------------------------------------------------
        # FIND AUDIO DEVICE
        # ---------------------------------------------------------------------

        device_index = find_audio_device(
            p
        )

        if device_index is None:

            print(
                "ERROR: No suitable stereo "
                "audio input device found."
            )

            return 1

        device_info = (
            p.get_device_info_by_index(
                device_index
            )
        )

        print()
        print(
            "Opening audio device:"
        )

        print(
            f"  Index: "
            f"{device_index}"
        )

        print(
            f"  Name: "
            f"{device_info.get('name')}"
        )

        print(
            f"  Channels: "
            f"{CHANNELS}"
        )

        print(
            f"  Sample rate: "
            f"{RATE}"
        )

        print(
            "  Format: paInt32"
        )

        print(
            f"  Chunk: "
            f"{CHUNK} frames"
        )

        # ---------------------------------------------------------------------
        # OPEN AUDIO STREAM
        # ---------------------------------------------------------------------

        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK
        )

        print()
        print(
            "Audio hardware opened successfully."
        )

        # ---------------------------------------------------------------------
        # FILTER INFORMATION
        # ---------------------------------------------------------------------

        print(
            f"Bandpass filter: "
            f"{LOWCUT_HZ:.0f} - "
            f"{HIGHCUT_HZ:.0f} Hz"
        )

        print(
            f"Butterworth order: "
            f"{FILTER_ORDER}"
        )

        print(
            f"Visual gain: "
            f"{VISUAL_GAIN}"
        )

        # ---------------------------------------------------------------------
        # CREATE QT APPLICATION
        # ---------------------------------------------------------------------

        app = QtWidgets.QApplication(
            sys.argv
        )

        scope = JetsonOscilloscope(
            stream
        )

        scope.show()

        print()
        print(
            "PCM1808 2x2 oscilloscope running."
        )

        print()
        print(
            "TOP LEFT     = LEFT RAW"
        )

        print(
            "TOP RIGHT    = LEFT BANDPASS"
        )

        print(
            "BOTTOM LEFT  = RIGHT RAW"
        )

        print(
            "BOTTOM RIGHT = RIGHT BANDPASS"
        )

        print()
        print(
            f"Bandpass: "
            f"{LOWCUT_HZ:.0f}-"
            f"{HIGHCUT_HZ:.0f} Hz"
        )

        print(
            "Press Ctrl+C or close the window to exit."
        )

        # ---------------------------------------------------------------------
        # RUN APPLICATION
        # ---------------------------------------------------------------------

        exit_code = app.exec_()

        return exit_code

    except KeyboardInterrupt:

        print()
        print(
            "Interrupted by user."
        )

        return 0

    except Exception as exc:

        print()
        print(
            "FATAL ERROR:"
        )

        print(
            repr(exc)
        )

        return 1

    finally:

        # ---------------------------------------------------------------------
        # STOP TIMER
        # ---------------------------------------------------------------------

        if scope is not None:

            try:
                scope.timer.stop()
            except Exception:
                pass

        # ---------------------------------------------------------------------
        # STOP AUDIO STREAM
        # ---------------------------------------------------------------------

        if stream is not None:

            try:

                if stream.is_active():
                    stream.stop_stream()

            except Exception as exc:

                print(
                    "Stream stop warning:",
                    exc
                )

            try:
                stream.close()
            except Exception as exc:

                print(
                    "Stream close warning:",
                    exc
                )

        # ---------------------------------------------------------------------
        # TERMINATE PYAUDIO
        # ---------------------------------------------------------------------

        try:
            p.terminate()
        except Exception as exc:

            print(
                "PyAudio termination warning:",
                exc
            )

        print(
            "Audio system shut down."
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    sys.exit(
        main()
    )

