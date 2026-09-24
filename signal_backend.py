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
# laptop/demo microphones can be quiet; the confidence threshold + enhancement
# handle the rest. (Auto-gain in enhance.py then brings the level up.)
SILENCE_RMS = 0.003

# The PCM1808/DAQ path has a much higher electrical noise floor than a laptop
# mic -- idle RMS measured ~0.017-0.02 on the Jetson rig (2026-09-24, see
# scripts/test_daq_signal.py), well above SILENCE_RMS above. Using the mic
# threshold there means every window looks "loud enough" even at idle. Set
# above the measured floor with margin; override with LMML_DAQ_SILENCE_RMS if
# the real hardware's noise floor drifts (different cabling, gain, etc.).
DAQ_SILENCE_RMS = 0.03
# Don't re-run the model on every GUI frame (that's ~20x/sec). Re-classify at
# most this often; between runs the dashboard shows the last result.
PREDICT_EVERY_SEC = 0.30

# --- Prediction stability (stops the readout flickering between digits) ---
# The sliding window classifies continuously, so one spoken digit passes through
# several windows (onset/middle/tail) that can disagree. To keep the displayed
# number STABLE we only "commit" a digit once the SAME label repeats on
# STABLE_HITS consecutive windows at >= STABLE_CONF confidence, and then hold it
# until a different digit is confirmed the same way.
STABLE_CONF = 0.60          # a window must be at least this confident to count
STABLE_HITS = 1             # demo mode: show a recognized digit immediately
# How many samples the oscilloscope trace shows (a short, recent slice).
SCOPE_SECONDS = 0.4


# Bluetooth earbuds/headsets frequently expose a mic that returns pure silence
# (they only enable it in call/HFP mode). If one is the system default we skip it
# in favor of a real built-in mic, which is what wrecked live capture once when
# AirPods became the default input device.
_BLUETOOTH_HINTS = ("airpods", "buds", "bluetooth", "headphone", "headset", "beats")
_BUILTIN_HINTS = ("macbook", "built-in", "internal", "microphone array")


def _pick_input_device(sd):
    """Choose a good input device index, or None to let PortAudio decide.

    Order: LMML_MIC_DEVICE env override -> the system default (unless it looks
    like Bluetooth earbuds, in which case prefer a built-in mic) -> first input.
    """
    import os
    override = os.environ.get("LMML_MIC_DEVICE")
    if override is not None:
        try:
            return int(override)
        except ValueError:
            pass

    devices = sd.query_devices()
    inputs = [i for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    if not inputs:
        return None

    try:
        default_idx = sd.default.device[0]
    except Exception:  # noqa: BLE001
        default_idx = None

    if default_idx in inputs:
        name = devices[default_idx]["name"].lower()
        if any(b in name for b in _BLUETOOTH_HINTS):
            for i in inputs:                       # prefer a built-in mic instead
                if any(k in devices[i]["name"].lower() for k in _BUILTIN_HINTS):
                    return i
        return default_idx
    return inputs[0]


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
        self.device = None            # chosen input-device index (None = default)
        self.device_name = None
        self.available = False
        self.error = None

        try:
            import sounddevice as sd  # lazy: only needed for live mic capture
            self._sd = sd
            # Confirm at least one input device actually exists before claiming
            # capture is available (importing succeeds even with no mic).
            if any(d["max_input_channels"] > 0 for d in sd.query_devices()):
                self.available = True
                self.device = _pick_input_device(sd)   # avoid dead Bluetooth mics
                try:
                    info = sd.query_devices(self.device if self.device is not None
                                            else None, kind="input")
                    self.rate = int(round(info["default_samplerate"]))
                    self.device_name = info["name"]
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
        self._stream = self._sd.InputStream(
            samplerate=self.rate,
            channels=1,
            dtype="float32",
            device=self.device,          # chosen built-in mic (skips Bluetooth)
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


# PCM1808 ADC boards typically show up under ALSA as a card whose name
# contains "APE" (the breakout board's label) or as hw:1,0. If neither is
# found we fall back to device index 1 (the known-working slot on the Jetson
# bring-up rig), then to the first stereo-capable input -- same order the
# hardware team's bring-up script uses.
_DAQ_NAME_HINTS = ("ape", "hw:1,0")
_DAQ_CHUNK = 4096 * 3  # frames per read; matches the value validated on hardware


def _pick_daq_device(pa, channels: int):
    """Find the PCM1808 input device index, or None if nothing suitable exists."""
    import os
    override = os.environ.get("LMML_DAQ_DEVICE")
    if override is not None:
        try:
            return int(override)
        except ValueError:
            pass

    count = pa.get_device_count()
    for i in range(count):
        try:
            info = pa.get_device_info_by_index(i)
        except Exception:  # noqa: BLE001
            continue
        name = str(info.get("name", "")).lower()
        if info.get("maxInputChannels", 0) >= channels and any(h in name for h in _DAQ_NAME_HINTS):
            return i

    if count > 1:
        try:
            info = pa.get_device_info_by_index(1)
            if info.get("maxInputChannels", 0) >= channels:
                return 1
        except Exception:  # noqa: BLE001
            pass

    for i in range(count):
        try:
            info = pa.get_device_info_by_index(i)
        except Exception:  # noqa: BLE001
            continue
        if info.get("maxInputChannels", 0) >= channels:
            return i
    return None


class _DAQSource:
    """Live capture from the PCM1808 ADC (laser DAQ) via PyAudio.

    Same available/start/stop/latest interface as `_MicSource`, so the rest of
    SignalBackend (resampling, filtering, prediction) doesn't change at all --
    this is the DAQ source promised in the LASER NOTE at the top of this file.

    Captured at the ADC's own native rate (2-channel, 32-bit int), then
    averaged to mono float32 into the same rolling buffer `_MicSource` uses.
    No band-pass filtering happens here -- that stays downstream in
    `preprocess.reduce_noise()` so the DAQ and mic paths share one filter.
    """

    CHANNELS = 2

    def __init__(self, buffer_seconds: float, rate: int = 48000, chunk: int = _DAQ_CHUNK):
        self.rate = int(rate)
        self._buffer_seconds = buffer_seconds
        self._chunk = chunk
        self.max_len = int(buffer_seconds * self.rate)
        self._buf = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._pa = None
        self._stream = None
        self._thread = None
        self._stop_flag = threading.Event()
        self.device = None
        self.device_name = None
        self.available = False
        self.error = None

        try:
            import pyaudio
            self._pyaudio = pyaudio
            self._pa = pyaudio.PyAudio()
            self.device = _pick_daq_device(self._pa, self.CHANNELS)
            if self.device is None:
                self.error = "No PCM1808/DAQ input device found."
                self._pa.terminate()
                self._pa = None
                return
            info = self._pa.get_device_info_by_index(self.device)
            self.device_name = info.get("name")
            # Use the device's own default rate if it reports one; otherwise
            # keep the caller's rate (matches the hardware team's validated 48 kHz).
            device_rate = info.get("defaultSampleRate") or 0
            if device_rate:
                self.rate = int(round(device_rate))
            self.max_len = int(self._buffer_seconds * self.rate)
            self.available = True
        except Exception as e:  # noqa: BLE001 - pyaudio missing, no device, etc.
            self.error = f"{type(e).__name__}: {e}"
            self._pa = None

    def _run(self):
        while not self._stop_flag.is_set():
            try:
                raw = self._stream.read(self._chunk, exception_on_overflow=False)
            except Exception:  # noqa: BLE001 - device hiccup, keep the loop alive
                continue
            ints = np.frombuffer(raw, dtype=np.int32)
            if ints.size != self._chunk * self.CHANNELS:
                continue
            # Stereo -> mono (average channels), matching the WAV path's
            # convention (docs/laser_daq_interface.md).
            stereo = ints.reshape(-1, self.CHANNELS).astype(np.float32)
            mono = stereo.mean(axis=1) / 2147483648.0  # int32 full-scale
            with self._lock:
                self._buf = np.concatenate([self._buf, mono])[-self.max_len:]

    def start(self):
        if not self.available or self._stream is not None:
            return
        with self._lock:
            self._buf = np.zeros(0, dtype=np.float32)
        self._stream = self._pa.open(
            format=self._pyaudio.paInt32,
            channels=self.CHANNELS,
            rate=self.rate,
            input=True,
            input_device_index=self.device,
            frames_per_buffer=self._chunk,
        )
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._stream is not None:
            try:
                if self._stream.is_active():
                    self._stream.stop_stream()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None

    def latest(self, seconds: float | None = None) -> np.ndarray:
        with self._lock:
            buf = self._buf.copy()
        if seconds is None:
            return buf
        n = int(seconds * self.rate)
        return buf[-n:] if len(buf) > n else buf


def _read_signal_file(path: str):
    """Read an audio or CSV capture into (samples_1d, sample_rate).

    - audio           : WAV/FLAC/OGG/AIFF and any other format supported by
                        soundfile/libsndfile on this machine. MP3/M4A are
                        best-effort and may require ffmpeg or OS codec support.
    - .csv            : laser/DAQ style. Uses a 'voltage' column if present,
                        else the last numeric column, as the signal. Sample rate
                        comes from a 'time' column (1/dt) if present, else falls
                        back to the project SAMPLE_RATE.

    This is the format the hardware team's laser captures are expected to land
    in (see docs/laser_daq_interface.md). Keeping the reader here means the
    dashboard can replay a real capture file the moment one exists.
    """
    import numpy as np
    path = str(path)
    p = path.lower()
    if not p.endswith(".csv"):
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)
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
        if n >= self._data.size:
            return self._data.copy()
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
        "daq"          -> live PCM1808/laser ADC capture via PyAudio (Sprint 4 hardware)
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

        # The DAQ's electrical noise floor sits well above a laptop mic's, so
        # it needs its own (higher) silence threshold. See DAQ_SILENCE_RMS.
        if source == "daq":
            import os
            override = os.environ.get("LMML_DAQ_SILENCE_RMS")
            self._silence_rms = float(override) if override else DAQ_SILENCE_RMS
        else:
            self._silence_rms = SILENCE_RMS

        # --- signal source: live mic, live DAQ, or replay a capture file ---
        if source == "mic":
            self._mic = _MicSource(SAMPLE_RATE, BUFFER_SECONDS)
        elif source == "daq":
            self._mic = _DAQSource(BUFFER_SECONDS)
        else:
            self._mic = _FileReplaySource(source, SAMPLE_RATE, BUFFER_SECONDS)
        self.audio_available = self._mic.available
        self.audio_error = self._mic.error

        # --- model availability (don't crash if weights aren't trained yet) ---
        self.model_available = True
        self.model_error = None
        try:
            if model == "ensemble":
                predict_mod._load("lstm")
                predict_mod._load("cnn")
            else:
                predict_mod._load(model)
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
        # torchaudio's STFT reflect-padding needs the signal to be longer than
        # its padding. Right after Record is clicked the live buffer can contain
        # only a few samples, so wait until the buffer is safely long enough.
        if raw.size < 512:
            return None
        try:
            wave = load_waveform_from_array(raw, self._mic.rate)
            spec = extract_spectrogram(wave).cpu().numpy()
        except Exception as e:  # noqa: BLE001 - never let the plot crash the demo
            print(f"[dashboard] spectrogram error: {type(e).__name__}: {e}")
            return None
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
        if rms < self._silence_rms:
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
        except Exception as e:  # noqa: BLE001 - keep the UI alive no matter what
            print(f"[dashboard] prediction error: {type(e).__name__}: {e}")
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
        elif self._committed is None:
            # During a demo, avoid a blank result panel. If the model heard
            # something but rejected it, show Unknown with the observed confidence.
            self._committed = ("Unknown", result["confidence"] * 100.0)
        # If not confident after a committed value exists, keep showing the
        # committed value (a brief mid-word dip shouldn't wipe the number).
        return self._committed

    # -- diagnostics -------------------------------------------------------
    def status_text(self) -> str:
        """One-line summary for logging/console when the dashboard starts."""
        parts = []
        if self.source_kind in ("mic", "daq"):
            src = self.source_kind
        else:
            src = f"file:{self.source_kind}"
        parts.append(f"source={src} {'ok' if self.audio_available else 'OFF'}"
                     + ("" if self.audio_available else f" ({self.audio_error})"))
        parts.append(f"model={'ok' if self.model_available else 'OFF'}"
                     + ("" if self.model_available else f" ({self.model_error})"))
        if self.source_kind in ("mic", "daq") and getattr(self._mic, "device_name", None):
            parts.append(f"device='{self._mic.device_name}'")
        parts.append(f"rate={self.sample_rate}Hz")
        return "SignalBackend: " + ", ".join(parts)


class _LiveClassifier:
    """Real classifier for a caller that already owns its own audio stream.

    SignalBackend pulls samples itself (from _MicSource/_DAQSource/a file) on
    every predict() call. Some callers -- signal_dashboard2.py's
    HardwareAudioSource is the reason this exists -- already have their own
    open PyAudio stream and hand out chunks on a GUI timer tick. Opening a
    second SignalBackend(source="daq") there would mean two streams fighting
    over the same PCM1808 device. This class gets the real model/silence-gate/
    debounce/autosave behavior without owning any audio device itself: push()
    chunks in as they arrive, predict() on the same cadence SignalBackend uses.

    One instance per independent signal (e.g. one per microphone channel).
    Not thread-safe: push()/predict() are meant to be called from the same
    thread, typically a GUI timer tick.
    """

    def __init__(self, model: str = "ensemble", threshold: float = CONFIDENCE_THRESHOLD,
                 silence_rms: float = DAQ_SILENCE_RMS, enhance: bool | None = None,
                 autosave: bool = True, source_label: str = "live"):
        # Default is the DAQ threshold, not the mic one: this class exists so a
        # caller with its own hardware audio stream (PCM1808) gets real
        # predictions, and that hardware's noise floor is much higher than a
        # laptop mic's. Pass silence_rms explicitly to override.
        self.model = model
        self.threshold = threshold
        self.silence_rms = silence_rms
        self.enhance = ENABLE_ENHANCE if enhance is None else enhance
        self.autosave = autosave
        self.source_label = source_label

        self.rate = SAMPLE_RATE
        self._buf = np.zeros(0, dtype=np.float32)
        self._max_len = int(BUFFER_SECONDS * self.rate)

        self._committed = None
        self._streak_label = None
        self._streak_count = 0
        self._last_predict_t = 0.0

        self.model_available = True
        self.model_error = None
        try:
            if model == "ensemble":
                predict_mod._load("lstm")
                predict_mod._load("cnn")
            else:
                predict_mod._load(model)
        except Exception as e:  # noqa: BLE001
            self.model_available = False
            self.model_error = f"{type(e).__name__}: {e}"

    def push(self, samples: np.ndarray, sample_rate: int) -> None:
        """Feed in the newest chunk of mono float samples at `sample_rate`."""
        if samples is None or len(samples) == 0:
            return
        if int(sample_rate) != self.rate:
            self.rate = int(sample_rate)
            self._max_len = int(BUFFER_SECONDS * self.rate)
        chunk = np.asarray(samples, dtype=np.float32)
        self._buf = np.concatenate([self._buf, chunk])[-self._max_len:]

    def reset(self) -> None:
        """Clear the rolling buffer and held result -- call when (re)starting."""
        self._buf = np.zeros(0, dtype=np.float32)
        self._committed = None
        self._streak_label = None
        self._streak_count = 0

    def predict(self):
        """Same throttle/silence-gate/debounce contract as SignalBackend.predict()."""
        if not self.model_available:
            return None

        import time
        now = time.monotonic()
        if now - self._last_predict_t < PREDICT_EVERY_SEC:
            return self._committed
        self._last_predict_t = now

        buf = self._buf[-int(PREDICT_WINDOW_SEC * self.rate):]
        if buf.size < int(0.2 * self.rate):
            return None

        rms = float(np.sqrt(np.mean(buf ** 2))) if buf.size else 0.0
        if rms < self.silence_rms:
            self._streak_label = None
            self._streak_count = 0
            return self._committed

        try:
            waveform = load_waveform_from_array(buf, self.rate)
            if self.enhance:
                waveform = enhance_waveform(waveform)
            if self.model == "ensemble":
                result, _ = predict_mod.infer_ensemble(waveform, threshold=self.threshold)
            else:
                result, _ = predict_mod.infer(waveform, threshold=self.threshold,
                                              model_type=self.model)
        except Exception as e:  # noqa: BLE001 - keep the UI alive no matter what
            print(f"[dashboard] prediction error ({self.source_label}): "
                  f"{type(e).__name__}: {e}")
            return self._committed

        label = result["prediction"]
        confident = result["status"] == "recognized" and result["confidence"] >= STABLE_CONF

        if confident:
            if label == self._streak_label:
                self._streak_count += 1
            else:
                self._streak_label = label
                self._streak_count = 1

            committed_label = self._committed[0] if self._committed else None
            if self._streak_count >= STABLE_HITS and label != committed_label:
                self._committed = (label, result["confidence"] * 100.0)
                if self.autosave:
                    try:
                        from utils import log_prediction
                        log_prediction(result, samples=buf, sample_rate=self.rate,
                                       source=self.source_label)
                    except Exception:  # noqa: BLE001
                        pass
        elif self._committed is None:
            self._committed = ("Unknown", result["confidence"] * 100.0)
        return self._committed
