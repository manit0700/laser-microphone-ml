"""
augment.py
==========
Data augmentation for training robustness.

WHY
---
The Free Spoken Digit Dataset is clean, studio-quality audio. Real microphone
and (later) laser/DAQ signals are NOT: they have background noise, vary in
loudness, and the digit isn't always perfectly aligned in time. A model trained
only on clean data can do great on FSDD and then fall apart on real signal.

Augmentation fixes this WITHOUT collecting more data: during training we make
randomized, realistic copies of each clip -- a little quieter/louder, with some
added noise, shifted slightly in time. The model then learns the digit itself,
not the pristine recording conditions, so it generalizes to real input.

WHERE IT RUNS
-------------
Applied per-sample in dataset.py, ONLY on the training split (never val/test),
and only when training is launched with `--augment`. Each transform is length-
preserving so the fixed (MAX_AUDIO_SAMPLES,) shape the model expects is kept.

LASER NOTE
----------
These defaults model microphone-style degradation. Once real laser captures
exist, tune the noise profile / SNR range here to match the laser signal's
actual noise, so augmentation reflects the real deployment conditions.
"""

import torch

# Each augmentation is applied with this probability (independently), so most
# clips get a mix of a few effects and some pass through nearly clean.
_P_GAIN = 0.8
_P_NOISE = 0.7
_P_SHIFT = 0.7

# Random gain range (multiply amplitude): models speaker distance / mic volume.
_GAIN_MIN, _GAIN_MAX = 0.6, 1.4
# Additive-noise strength as signal-to-noise ratio in dB (higher = cleaner).
_SNR_MIN_DB, _SNR_MAX_DB = 8.0, 30.0
# Max time shift as a fraction of the clip length (rolls the digit left/right).
_SHIFT_FRAC = 0.15


def _rand(lo: float, hi: float) -> float:
    """Uniform random float in [lo, hi] using torch's RNG (seed-respecting)."""
    return float(torch.empty(1).uniform_(lo, hi).item())


def random_gain(wave: torch.Tensor) -> torch.Tensor:
    """Scale the whole clip up or down in volume."""
    return wave * _rand(_GAIN_MIN, _GAIN_MAX)


def add_noise(wave: torch.Tensor) -> torch.Tensor:
    """Add Gaussian noise at a random SNR (models background hiss/hum)."""
    signal_power = wave.pow(2).mean()
    if signal_power <= 0:
        return wave
    snr_db = _rand(_SNR_MIN_DB, _SNR_MAX_DB)
    # SNR(dB) = 10*log10(signal_power / noise_power) -> solve for noise_power.
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = torch.randn_like(wave) * torch.sqrt(noise_power)
    return wave + noise


def time_shift(wave: torch.Tensor) -> torch.Tensor:
    """Shift the clip in time by a random amount, zero-filling the gap.

    Length is preserved. This teaches the model that the digit can start a bit
    earlier or later, which matters for real captures that aren't tightly trimmed.
    """
    n = wave.numel()
    max_shift = int(_SHIFT_FRAC * n)
    if max_shift < 1:
        return wave
    shift = int(_rand(-max_shift, max_shift))
    if shift == 0:
        return wave
    out = torch.zeros_like(wave)
    if shift > 0:
        out[shift:] = wave[:n - shift]
    else:
        out[:n + shift] = wave[-shift:]
    return out


def augment_waveform(wave: torch.Tensor) -> torch.Tensor:
    """Apply a random subset of augmentations to one fixed-length waveform.

    Input/Output: 1-D tensor of the same length. Safe to call every epoch; the
    randomness means the model sees a slightly different version each time.
    """
    if torch.rand(1).item() < _P_GAIN:
        wave = random_gain(wave)
    if torch.rand(1).item() < _P_NOISE:
        wave = add_noise(wave)
    if torch.rand(1).item() < _P_SHIFT:
        wave = time_shift(wave)
    # Keep amplitude in a sane range so downstream normalization behaves.
    peak = wave.abs().max()
    if peak > 1.0:
        wave = wave / peak
    return wave


if __name__ == "__main__":
    # Self-test: shape is preserved and the output actually differs from input.
    from config import MAX_AUDIO_SAMPLES
    x = torch.randn(MAX_AUDIO_SAMPLES) * 0.3
    y = augment_waveform(x)
    print(f"in shape {tuple(x.shape)} -> out shape {tuple(y.shape)}")
    print(f"changed: {not torch.allclose(x, y)}")
