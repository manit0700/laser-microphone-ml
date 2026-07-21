"""
signal_backend.py
=================
The bridge between the UI dashboard (signal_dashboard.py) and the real ML
pipeline in src/. The dashboard draws the oscilloscope, spectrogram, and the
prediction panel; THIS file gives it real numbers instead of the fake ones.

It fills the four "ML TEAM:" hooks the dashboard left open:
    1. waveform acquisition   -> live microphone samples (or laser/DAQ later)
    2. spectrogram generation -> real STFT of the captured signal
    3. classifier prediction  -> the trained LSTM/CNN/ensemble result
    4. band-pass filter switch -> our real reduce_noise() filter, toggled live

DESIGN GOALS
------------
- Keep the dashboard file almost untouched: it just asks this backend for data
  and falls back to its own dummy generators if the backend isn't ready.
- Degrade gracefully. If the mic library (sounddevice) isn't installed, or no
  trained model exists, the relevant `*_available` flag is False and the
  dashboard keeps showing its demo animation instead of crashing.
- Reuse the EXACT same inference path as predict.py / app.py, so what you see on
  the dashboard is what the rest of the system produces.

LASER NOTE
----------
Right now the audio source is the microphone (sounddevice). When the laser/DAQ
arrives, swap `_MicSource` for a DAQ source that fills the same rolling buffer
(a 1-D float array at some sample rate) and nothing else here changes: the
resampling, filtering, and model are all downstream of the buffer.
"""

import sys
import threading
from pathlib import Path

import numpy as np

# Make the src/ modules importable whether this file is run from the repo root
# or elsewhere.
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# These imports are the real pipeline. They only need numpy/torch/torchaudio,
# which the ML environment already has -- NOT the GUI or mic libraries.
from config import CONFIDENCE_THRESHOLD, SAMPLE_RATE, ENABLE_ENHANCE  # noqa: E402
from preprocess import load_waveform_from_array, preprocess_waveform, set_filter  # noqa: E402
from features import extract_spectrogram                       # noqa: E402
from enhance import enhance_waveform                           # noqa: E402
import predict as predict_mod                                  # noqa: E402


# ---------------------------------------------------------------------------
# Tuning knobs (mirror the values used in app.py's live mode so behavior matches)
# ---------------------------------------------------------------------------
# How much recent audio we keep. Long enough to hold one spoken digit (~1.3 s)
# plus a little slack for the scope/spectrogram views.
BUFFER_SECONDS = 1.5
# The window (seconds) actually fed to the model for each prediction.
PREDICT_WINDOW_SEC = 1.3
# Below this RMS loudness the window is treated as silence and NOT classified,
# so background quiet doesn't produce a confident random digit. Kept low because
# quiet speakers can sit around 0.008 RMS; the confidence threshold + enhancement
# handle the rest. (Auto-gain in enhance.py then brings the level up.)
SILENCE_RMS = 0.006
# Don't re-run the model on every GUI frame (that's ~20x/sec). Re-classify at
# most this often; between runs the dashboard shows the last result.
PREDICT_EVERY_SEC = 0.30

# --- Prediction stability (stops the readout flickering between digits) ---
# The sliding window classifies continuously, so one spoken digit passes through
# several windows (onset/middle/tail) that can disagree. To keep the displayed
# number STABLE we only "commit" a digit once the SAME label repeats on
# STABLE_HITS consecutive windows at >= STABLE_CONF confidence, and then hold it
# until a different digit is confirmed the same way.
STABLE_CONF = 0.75          # a window must be at least this confident to count
STABLE_HITS = 2             # this many agreeing windows in a row -> commit
# How many samples the oscilloscope trace shows (a short, recent slice).
SCOPE_SECONDS = 0.4


class _MicSource:
    """Microphone capture using sounddevice, kept in a rolling buffer.

    sounddevice is imported lazily inside here so that simply importing this
    module (e.g. on a headless ML box, or during tests) never fails just because
    the mic library isn't installed. `available` tells the caller whether real
    capture is possible.
    """

    def __init__(self, sample_rate: int, buffer_seconds: float):
        # IMPORTANT: we capture at the microphone's OWN native rate and let the
        # pipeline resample to the project rate downstream. Forcing the mic to
        # the project's 8 kHz produces distorted audio on Macs (native 44.1/48
        # kHz), which wrecks the MFCCs and the prediction. `self.rate` is the
        # rate the samples in the buffer are at (what latest() returns).
        self.rate = int(sample_rate)          # updated to the device rate below
        self._buffer_seconds = buffer_seconds
        self.max_len = int(buffer_seconds * self.rate)
        self._buf = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._stream = None
        self._sd = None
        self.available = False
        self.error = None

        try:
            import sounddevice as sd  # lazy: only needed for live mic capture
            self._sd = sd
            # Confirm at least one input device actually exists before claiming
            # capture is available (importing succeeds even with no mic).
            if any(d["max_input_channels"] > 0 for d in sd.query_devices()):
                self.available = True
                # Use the default input device's native sample rate for capture.
                try:
                    default_in = sd.query_devices(kind="input")
                    self.rate = int(round(default_in["default_samplerate"]))
                except Exception:  # noqa: BLE001 - fall back to project rate
                    pass
                self.max_len = int(self._buffer_seconds * self.rate)
            else:
                self.error = "No microphone input device found."
        except Exception as e:  # noqa: BLE001 - any import/query issue -> unavailable
            self.error = f"{type(e).__name__}: {e}"

    def _callback(self, indata, frames, time_info, status):
        """sounddevice calls this from an audio thread with each new mic chunk."""
        chunk = indata[:, 0].astype(np.float32)  # first channel -> mono
        with self._lock:
            self._buf = np.concatenate([self._buf, chunk])[-self.max_len:]

    def start(self):
        if not self.available or self._stream is not None:
            return
        with self._lock:
            self._buf = np.zeros(0, dtype=np.float32)
        # samplerate=None lets PortAudio open the device at its native rate.
        self._stream = self._sd.InputStream(
            samplerate=self.rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def latest(self, seconds: float | None = None) -> np.ndarray:
        """Return the most recent `seconds` of audio (all of it if None)."""
        with self._lock:
            buf = self._buf.copy()
        if seconds is None:
            return buf
        n = int(seconds * self.rate)
        return buf[-n:] if len(buf) > n else buf


def _read_signal_file(path: str):
    """Read a WAV or CSV capture into (samples_1d, sample_rate).

    - .wav            : standard audio, any rate (resampled downstream).
    - .csv            : laser/DAQ style. Uses a 'voltage' column if present,
                        else the last numeric column, as the signal. Sample rate
                        comes from a 'time' column (1/dt) if present, else falls
                        back to the project SAMPLE_RATE.

    This is the format the hardware team's laser captures are expected to land
    in (see docs/laser_daq_interface.md). Keeping the reader here means the
    dashboard can replay a real capture file the moment one exists.
    """
    import numpy as np
    p = str(path).lower()
    if p.endswith(".wav"):
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return np.asarray(data, dtype=np.float32).mean(axis=1), int(sr)

    # CSV: read the header, pick signal + optional time column.
    import csv as _csv
    with open(path, newline="") as f:
        rows = list(_csv.reader(f))
    header = [h.strip().lower() for h in rows[0]]
    body = np.array([[float(v) for v in r] for r in rows[1:] if r], dtype=np.float64)
    sig_col = header.index("voltage") if "voltage" in header else body.shape[1] - 1
    samples = body[:, sig_col].astype(np.float32)
    if "time" in header and body.shape[0] > 1:
        t = body[:, header.index("time")]
        dt = float(np.median(np.diff(t)))
        sr = int(round(1.0 / dt)) if dt > 0 else SAMPLE_RATE
    else:
        sr = SAMPLE_RATE
    return samples, sr


class _FileReplaySource:
    """Streams a WAV/CSV capture as if it were arriving live from a device.

    Same interface as _MicSource (available/start/stop/latest), so the dashboard
    and SignalBackend treat a replayed laser file exactly like a live mic. Loops
    the file so the scope keeps moving. This is how you demo/test the full
    pipeline BEFORE the laser hardware is connected (Sprint 4).
    """

    def __init__(self, path: str, sample_rate: int, buffer_seconds: float):
        import numpy as np
        # File data is resampled to the project rate on load, so that's the rate
        # the samples we hand out are at (mirrors _MicSource.rate).
        self.sample_rate = sample_rate
        self.rate = sample_rate
        self.max_len = int(buffer_seconds * sample_rate)
        self.available = False
        self.error = None
        self._pos = 0
        self._data = np.zeros(0, dtype=np.float32)
        self._playing = False
        try:
            samples, sr = _read_signal_file(path)
            wave = load_waveform_from_array(samples, sr)   # resample to SAMPLE_RATE
            self._data = wave.cpu().numpy().astype(np.float32)
            self.available = self._data.size > 0
            if not self.available:
                self.error = "Capture file is empty."
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"

    def start(self):
        self._playing = True

    def stop(self):
        self._playing = False

    def latest(self, seconds=None):
        import numpy as np
        if not self.available:
            return np.zeros(0, dtype=np.float32)
        # Advance the playback head a little each call so the view scrolls.
        if self._playing:
            self._pos = (self._pos + int(0.05 * self.sample_rate)) % self._data.size
        n = self.max_len if seconds is None else int(seconds * self.sample_rate)
        # Take a window ending at the current head, wrapping around the file.
        idx = (np.arange(self._pos - n, self._pos)) % self._data.size
        return self._data[idx]


class SignalBackend:
    """Everything the dashboard needs, behind one small, safe interface.

    The dashboard checks `.audio_available` and `.model_available`; when either
    is False it keeps using its own dummy data for that part. All methods are
    safe to call every GUI frame -- they're cheap and never raise.

    `source` selects where the signal comes from:
        "mic"          -> live microphone (default, for the current demo)
        path to a file -> replay a WAV/CSV capture as if live (test without
                          hardware, or play back a real laser capture in Sprint 4)
    """

    def __init__(self, threshold: float = CONFIDENCE_THRESHOLD, model: str = "lstm",
                 source: str = "mic", autosave: bool = True):
        self.sample_rate = SAMPLE_RATE
        self.threshold = threshold
        self.model = model                # "lstm", "cnn", or "ensemble"
        self.source_kind = source
        # Save each recognized capture (clip + CSV) to grow a real dataset.
        self.autosave = autosave
        self._last_logged = None
        # Live audio enhancement (denoise + auto-gain) before prediction.
        self.enhance = ENABLE_ENHANCE
        # Prediction-stability state (debounce): the committed/held result plus
        # the current agreement streak.
        self._committed = None       # (label, confidence_percent) shown on screen
        self._streak_label = None
        self._streak_count = 0
        self._running = False

        # --- signal source: live mic, or replay a capture file ---
        if source == "mic":
            self._mic = _MicSource(SAMPLE_RATE, BUFFER_SECONDS)
        else:
            self._mic = _FileReplaySource(source, SAMPLE_RATE, BUFFER_SECONDS)
        self.audio_available = self._mic.available
        self.audio_error = self._mic.error

        # --- model availability (don't crash if weights aren't trained yet) ---
        self.model_available = True
        self.model_error = None
        try:
            predict_mod._load("lstm" if model == "ensemble" else model)
        except Exception as e:  # noqa: BLE001
            self.model_available = False
            self.model_error = f"{type(e).__name__}: {e}"

        # Prediction throttle + last result cache.
        self._last_predict_t = 0.0
        self._last_result = None

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        """Begin live capture (called when the user hits Record)."""
        self._mic.start()
        self._running = True

    def stop(self):
        """Stop live capture (called on Stop)."""
        self._mic.stop()
        self._running = False
        # Clear the held result + streak so the next session starts clean.
        self._committed = None
        self._streak_label = None
        self._streak_count = 0

    @property
    def running(self) -> bool:
        return self._running

    # -- hook 4: band-pass filter -----------------------------------------
    def set_bandpass(self, enabled: bool):
        """Wire the dashboard's Band-Pass switch to the real filter."""
        set_filter(bool(enabled))

    # -- hook 1: waveform for the oscilloscope ----------------------------
    def scope_samples(self):
        """Recent mic samples for the scope trace, or None if nothing captured yet.

        Applies the SAME preprocessing filter as the model path when the band-pass
        switch is on, so the scope shows what the model actually sees.
        """
        if not self.audio_available:
            return None
        raw = self._mic.latest(SCOPE_SECONDS)
        if raw.size < 2:
            return None
        wave = load_waveform_from_array(raw, self._mic.rate)
        if self.enhance:
            wave = enhance_waveform(wave)      # show the cleaned signal on the scope
        from preprocess import reduce_noise    # honors the runtime filter toggle
        return reduce_noise(wave).cpu().numpy()

    # -- hook 2: spectrogram ----------------------------------------------
    def spectrogram(self):
        """Real (freq_bins, time_bins) magnitude spectrogram in dB, or None."""
        if not self.audio_available:
            return None
        raw = self._mic.latest(SCOPE_SECONDS)
        if raw.size < 64:
            return None
        wave = load_waveform_from_array(raw, self._mic.rate)
        spec = extract_spectrogram(wave).cpu().numpy()
        # ImageView expects (x=time, y=freq); our spec is (freq, time) -> transpose.
        return spec.T

    # -- hook 3: prediction -----------------------------------------------
    def predict(self):
        """Classify the current audio window.

        Returns (label, confidence_percent) or None when there's nothing
        confident to show (silence, too little audio, or no model). Throttled so
        the model runs a few times per second, not on every GUI frame.
        """
        if not (self.audio_available and self.model_available and self._running):
            return None

        import time
        now = time.monotonic()
        if now - self._last_predict_t < PREDICT_EVERY_SEC:
            return self._committed             # hold the committed result between runs
        self._last_predict_t = now

        buf = self._mic.latest(PREDICT_WINDOW_SEC)
        if buf.size < int(0.2 * self._mic.rate):   # need a bit of audio first
            return None

        # Silence gate: quiet window -> not speech. Reset the agreement streak so
        # the next spoken digit is judged fresh, but HOLD the committed result on
        # screen (don't blank it between words).
        rms = float(np.sqrt(np.mean(buf ** 2))) if buf.size else 0.0
        if rms < SILENCE_RMS:
            self._streak_label = None
            self._streak_count = 0
            self._last_logged = None      # allow a repeat of the same digit later
            return self._committed

        # Never let an inference hiccup crash the live loop / the demo: on any
        # error, hold the last committed result instead of raising.
        try:
            waveform = load_waveform_from_array(buf, self._mic.rate)
            if self.enhance:
                waveform = enhance_waveform(waveform)     # denoise + auto-gain for live audio
            if self.model == "ensemble":
                result, _ = predict_mod.infer_ensemble(waveform, threshold=self.threshold)
            else:
                result, _ = predict_mod.infer(waveform, threshold=self.threshold,
                                              model_type=self.model)
        except Exception:  # noqa: BLE001 - keep the UI alive no matter what
            return self._committed

        label = result["prediction"]                # digit string or "unknown"
        confident = result["status"] == "recognized" and result["confidence"] >= STABLE_CONF

        # Debounce: require the SAME confident label on several windows in a row
        # before changing the displayed number. This is what stops the readout
        # flickering as one spoken digit passes through the sliding window.
        if confident:
            if label == self._streak_label:
                self._streak_count += 1
            else:
                self._streak_label = label
                self._streak_count = 1

            committed_label = self._committed[0] if self._committed else None
            if self._streak_count >= STABLE_HITS and label != committed_label:
                # New digit confirmed -> commit it, and save it once.
                self._committed = (label, result["confidence"] * 100.0)
                if self.autosave:
                    try:
                        from utils import log_prediction
                        log_prediction(result, samples=buf,
                                       sample_rate=self._mic.rate, source="dashboard")
                    except Exception:  # noqa: BLE001 - never break the UI
                        pass
        # If not confident, leave the streak as-is and keep showing the committed
        # value (a brief mid-word dip shouldn't wipe the number).
        return self._committed

    # -- diagnostics -------------------------------------------------------
    def status_text(self) -> str:
        """One-line summary for logging/console when the dashboard starts."""
        parts = []
        src = "mic" if self.source_kind == "mic" else f"file:{self.source_kind}"
        parts.append(f"source={src} {'ok' if self.audio_available else 'OFF'}"
                     + ("" if self.audio_available else f" ({self.audio_error})"))
        parts.append(f"model={'ok' if self.model_available else 'OFF'}"
                     + ("" if self.model_available else f" ({self.model_error})"))
        parts.append(f"rate={self.sample_rate}Hz")
        return "SignalBackend: " + ", ".join(parts)
