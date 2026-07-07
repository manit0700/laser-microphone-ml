"""
preprocess.py
=============
Turns a raw audio file (or raw signal array) into a clean, fixed-length, mono,
normalized waveform that is ready for MFCC feature extraction.

PIPELINE STAGE
--------------
raw signal/audio  ->  [THIS FILE]  ->  MFCC features (features.py)

Steps performed here (in order):
    1. load audio                -> read samples from a WAV file
    2. convert to mono           -> average stereo channels into one
    3. resample if needed        -> force a consistent SAMPLE_RATE
    4. normalize amplitude       -> scale so the loudest sample is ~1.0
    5. endpoint detection / trim -> keep only the spoken-digit region
    6. pad or truncate           -> force a fixed length (MAX_AUDIO_SAMPLES)
    (optional) noise filtering   -> placeholder for the future

INPUT  : path to a .wav file  (or a raw 1-D numpy/torch waveform + its sample rate)
OUTPUT : a 1-D torch.FloatTensor of length MAX_AUDIO_SAMPLES at SAMPLE_RATE

LASER NOTE
----------
When laser-vibrometry data arrives, the *only* part that changes is HOW we get
the raw waveform into this file. Audio comes from WAV; laser data may come from
WAV, CSV (voltage vs. time), or a live DAQ stream. Use `load_waveform_from_array`
for non-WAV sources, then the rest of the steps below are reused unchanged.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from config import (
    ENERGY_THRESHOLD_RATIO,
    FRAME_HOP,
    FRAME_LENGTH,
    MAX_AUDIO_SAMPLES,
    SAMPLE_RATE,
)


# ---------------------------------------------------------------------------
# Step 1-3: load, mono, resample
# ---------------------------------------------------------------------------
def load_audio(path: str | Path) -> torch.Tensor:
    """Load a WAV file and return a mono waveform at the project SAMPLE_RATE.

    Returns a 1-D tensor of shape (num_samples,).

    We use `soundfile` for reading because it loads WAV files without needing
    ffmpeg/torchcodec, which keeps setup simple for the whole team. (torchaudio
    is still used for MFCC/mel feature extraction in features.py.)
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (samples, channels)
    waveform = torch.from_numpy(np.asarray(data, dtype=np.float32)).transpose(0, 1)  # (channels, samples)
    waveform = to_mono(waveform)
    waveform = resample_if_needed(waveform, sr)
    return waveform


def save_wav(samples, path: str | Path, sample_rate: int = SAMPLE_RATE) -> Path:
    """Write a 1-D signal (numpy array, list, or torch tensor) to a WAV file.

    This is the shared "make a WAV file" helper used by the scripts/ tools
    (recording, CSV->WAV, array->WAV). It peak-normalizes to avoid clipping and
    saves as 16-bit PCM, which every audio tool can read.

    Args:
        samples     : 1-D signal values (e.g. mic samples or laser voltages).
        path        : output .wav path.
        sample_rate : samples per second to record in the WAV header.
    Returns the written Path.
    """
    arr = np.asarray(
        samples.detach().cpu().numpy() if isinstance(samples, torch.Tensor) else samples,
        dtype=np.float32,
    ).reshape(-1)

    peak = np.max(np.abs(arr)) if arr.size else 0.0
    if peak > 0:
        arr = arr / peak  # normalize to [-1, 1] so nothing clips

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), arr, int(sample_rate), subtype="PCM_16")
    return path


def load_waveform_from_array(samples, sample_rate: int) -> torch.Tensor:
    """Build a clean mono waveform from a raw array (numpy or list or tensor).

    Use this for laser CSV/stream data: read the voltage values into `samples`,
    pass the DAQ sample_rate, and you get the same format as load_audio().

    Example (future laser CSV):
        import pandas as pd
        df = pd.read_csv("laser_capture.csv")        # columns: time, voltage
        wave = load_waveform_from_array(df["voltage"].values, sample_rate=50000)
    """
    waveform = torch.as_tensor(samples, dtype=torch.float32)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # (1, samples)
    waveform = to_mono(waveform)
    waveform = resample_if_needed(waveform, sample_rate)
    return waveform


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Average all channels into a single mono channel. Returns shape (samples,)."""
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0)  # -> (samples,)


def resample_if_needed(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    """Resample to SAMPLE_RATE only if the source rate differs."""
    if sr != SAMPLE_RATE:
        waveform = AF.resample(waveform, orig_freq=sr, new_freq=SAMPLE_RATE)
    return waveform


# ---------------------------------------------------------------------------
# Step 4: normalize amplitude
# ---------------------------------------------------------------------------
def normalize(waveform: torch.Tensor) -> torch.Tensor:
    """Scale the waveform so its loudest sample is ~1.0 (peak normalization).

    This removes differences in recording volume between speakers/microphones,
    so the model focuses on the *shape* of the sound, not how loud it was.
    """
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    return waveform


# ---------------------------------------------------------------------------
# Step 5: endpoint detection / silence trimming
# ---------------------------------------------------------------------------
def endpoint_detection(waveform: torch.Tensor) -> torch.Tensor:
    """Keep only the spoken-digit region; drop leading/trailing silence.

    HOW IT WORKS (simple energy-based method):
      1. Slide a short window across the signal and compute the energy
         (sum of squared samples) in each window/frame.
      2. The loudest frame tells us how loud the actual speech is.
      3. Any frame with energy above ENERGY_THRESHOLD_RATIO * (loudest frame)
         is considered "speech"; everything else is "silence".
      4. We cut the signal from the first speech frame to the last speech frame.

    This is intentionally simple and easy to explain. More advanced methods
    (e.g. voice activity detection, spectral gating) can replace this later
    without changing the rest of the pipeline.

    If no frame passes the threshold (e.g. pure silence/noise), we return the
    original waveform unchanged so downstream code still gets a valid signal.
    """
    if waveform.numel() < FRAME_LENGTH:
        return waveform

    # Break the signal into overlapping frames: shape (num_frames, FRAME_LENGTH).
    frames = waveform.unfold(0, FRAME_LENGTH, FRAME_HOP)
    energy = frames.pow(2).sum(dim=1)  # energy per frame

    peak_energy = energy.max()
    if peak_energy <= 0:
        return waveform

    threshold = ENERGY_THRESHOLD_RATIO * peak_energy
    speech_frames = (energy >= threshold).nonzero(as_tuple=False).squeeze(-1)
    if speech_frames.numel() == 0:
        return waveform

    first_frame = int(speech_frames[0].item())
    last_frame = int(speech_frames[-1].item())

    start = first_frame * FRAME_HOP
    end = last_frame * FRAME_HOP + FRAME_LENGTH
    end = min(end, waveform.numel())
    return waveform[start:end]


# ---------------------------------------------------------------------------
# Step 6: pad or truncate to a fixed length
# ---------------------------------------------------------------------------
def pad_or_truncate(waveform: torch.Tensor, length: int = MAX_AUDIO_SAMPLES) -> torch.Tensor:
    """Force the waveform to exactly `length` samples.

    - Shorter signals are padded with zeros at the end.
    - Longer signals are cut down to the HIGHEST-ENERGY window of `length` samples
      (not just the first `length`). This keeps the actual spoken digit centered
      even when the recording is longer than one second or the digit isn't at the
      very start -- important for real mic/laser recordings that aren't already
      tightly trimmed. Short clips (e.g. FSDD) are unaffected, so training is not
      changed.
    This guarantees every sample has the same shape so they can be batched.
    """
    n = waveform.numel()
    if n < length:
        pad = torch.zeros(length - n, dtype=waveform.dtype)
        return torch.cat([waveform, pad], dim=0)
    if n > length:
        # Slide a `length`-sized window (coarse hop for speed) and keep the loudest.
        hop = max(1, length // 8)
        best_start, best_energy = 0, -1.0
        for start in range(0, n - length + 1, hop):
            energy = waveform[start:start + length].pow(2).sum().item()
            if energy > best_energy:
                best_energy, best_start = energy, start
        return waveform[best_start:best_start + length]
    return waveform


# ---------------------------------------------------------------------------
# Optional: noise filtering placeholder
# ---------------------------------------------------------------------------
def reduce_noise(waveform: torch.Tensor) -> torch.Tensor:
    """Placeholder for future noise reduction / filtering.

    For now this is a pass-through. Laser-vibrometry signals may need band-pass
    filtering or spectral noise gating here. Implement it later (e.g. with
    scipy.signal) without changing the rest of the pipeline.
    """
    # TODO (laser team): add band-pass / denoising tuned to the laser signal.
    return waveform


# ---------------------------------------------------------------------------
# Full preprocessing entry points
# ---------------------------------------------------------------------------
def center_on_peak(waveform: torch.Tensor, length: int = MAX_AUDIO_SAMPLES) -> torch.Tensor:
    """For a long recording, cut out a `length` window CENTERED on the loudest point.

    Unlike endpoint detection (which trims from the first to the last loud frame and
    can merge a digit with a later cough/noise), this keeps a single window with
    natural lead-in/lead-out around the loudest moment -- which looks like the
    training clips and gives the model a clean, well-aligned digit.
    """
    n = waveform.numel()
    if n <= length:
        return waveform
    w, hop = 1024, 256
    best_i, best_e = 0, -1.0
    for i in range(0, n - w, hop):
        e = waveform[i:i + w].pow(2).sum().item()
        if e > best_e:
            best_e, best_i = e, i
    peak = best_i + w // 2
    start = max(0, min(peak - length // 2, n - length))  # clamp inside bounds
    return waveform[start:start + length]


def preprocess_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """Turn a raw mono waveform into a clean, fixed-length clip for MFCC extraction.

    - Long recordings (> one window): isolate the loudest 1s window centered on the
      digit (center_on_peak), which handles real mic/laser captures that aren't
      pre-trimmed and may contain extra sounds.
    - Short clips (e.g. FSDD, < one window): trim leading/trailing silence with
      endpoint detection, then pad. (Behavior unchanged, so training is unaffected.)
    """
    waveform = reduce_noise(waveform)
    waveform = normalize(waveform)
    if waveform.numel() > MAX_AUDIO_SAMPLES:
        waveform = center_on_peak(waveform, MAX_AUDIO_SAMPLES)
    else:
        waveform = endpoint_detection(waveform)
    waveform = normalize(waveform)
    waveform = pad_or_truncate(waveform)
    return waveform


def preprocess_file(path: str | Path) -> torch.Tensor:
    """Full path from a WAV file to a clean, fixed-length waveform.

    This is the convenient one-call function used by dataset.py and predict.py.
    """
    waveform = load_audio(path)
    return preprocess_waveform(waveform)


if __name__ == "__main__":
    # Quick self-test: prints the output shape for a file passed on the CLI.
    import sys

    if len(sys.argv) > 1:
        wav = preprocess_file(sys.argv[1])
        print(f"Preprocessed waveform shape: {tuple(wav.shape)} "
              f"(expected: ({MAX_AUDIO_SAMPLES},))")
    else:
        print("Usage: python preprocess.py path/to/file.wav")
