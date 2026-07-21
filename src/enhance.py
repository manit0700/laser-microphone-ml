"""
enhance.py
==========
Live audio enhancement applied to microphone (and future laser) captures BEFORE
feature extraction, to make real-world input look more like the clean clips the
model was trained on.

WHY (and why NOT a band-pass filter here)
-----------------------------------------
The model is trained on FSDD + Speech Commands, which are already fairly clean
and were NOT band-pass filtered. So applying a band-pass at inference would push
live audio into a DIFFERENT distribution than training and hurt accuracy. Instead
we do enhancements that make noisy/quiet mic audio look MORE like the clean
training clips, without changing its frequency character:

  1. DC-offset removal   : subtract the mean (removes electrical bias).
  2. Spectral-gate denoise: estimate the noise floor per frequency (from the
     quietest frames) and subtract it, so background hiss/hum is reduced while
     speech is kept. This raises SNR -> closer to clean training data.
  3. Auto-gain (AGC)     : scale to a consistent target level, so whether the
     speaker is quiet or loud the model sees a similar amplitude. Fixes the
     "too quiet -> noise amplified" and inconsistent-level problems on real mics.

All three are gentle and speech-preserving. Controlled by ENABLE_ENHANCE in
config (default on for live use). On already-clean clips these are near no-ops,
so training/eval on FSDD+SC is essentially unchanged.
"""

import torch

from config import SAMPLE_RATE

# STFT settings for the denoiser (independent of the MFCC settings).
_N_FFT = 512
_HOP = 128
_WINDOW = torch.hann_window(_N_FFT)

# Denoise strength: how much of the estimated noise floor to subtract, and the
# floor below which we never push a bin (prevents "musical noise" artifacts).
# Kept GENTLE so speech isn't distorted -- the trained 'unknown' class handles
# noise rejection, so denoise only needs a light touch here.
_NOISE_OVERSUB = 0.8     # subtract <1x the estimated noise magnitude (light)
_SPECTRAL_FLOOR = 0.30   # keep at least 30% of the original magnitude (less artifacts)

# Auto-gain target RMS (in [-1,1]). Speech around this level is comfortably above
# the noise floor without clipping.
_TARGET_RMS = 0.12


def remove_dc(x: torch.Tensor) -> torch.Tensor:
    """Subtract the mean to remove any constant offset."""
    return x - x.mean()


def spectral_denoise(x: torch.Tensor) -> torch.Tensor:
    """Reduce stationary background noise via per-frequency spectral subtraction.

    Estimates the noise magnitude in each frequency bin from the QUIETEST frames
    (background, not speech), subtracts an over-scaled version of it, and rebuilds
    the signal with the original phase. Length is preserved.
    """
    if x.numel() < _N_FFT:
        return x
    win = _WINDOW.to(x.dtype)
    spec = torch.stft(x, n_fft=_N_FFT, hop_length=_HOP, window=win,
                      return_complex=True, center=True)     # (freq, frames)
    mag = spec.abs()
    phase = spec.angle()

    # Noise floor per freq bin: the 10th-percentile magnitude across time
    # (frames with the least energy are assumed to be background noise).
    k = max(1, int(0.10 * mag.shape[1]))
    noise = mag.sort(dim=1).values[:, :k].mean(dim=1, keepdim=True)

    clean_mag = mag - _NOISE_OVERSUB * noise
    clean_mag = torch.maximum(clean_mag, _SPECTRAL_FLOOR * mag)   # spectral floor

    clean = clean_mag * torch.exp(1j * phase)
    out = torch.istft(clean, n_fft=_N_FFT, hop_length=_HOP, window=win,
                      center=True, length=x.numel())
    return out


def auto_gain(x: torch.Tensor, target_rms: float = _TARGET_RMS) -> torch.Tensor:
    """Scale the signal to a consistent RMS level, then guard against clipping."""
    rms = x.pow(2).mean().sqrt()
    if rms > 1e-6:
        x = x * (target_rms / rms)
    peak = x.abs().max()
    if peak > 0.99:                     # never clip after gain
        x = x * (0.99 / peak)
    return x


def enhance_waveform(x: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Full live-audio enhancement: DC removal -> denoise -> auto-gain.

    Input/Output: 1-D float tensor of the same length. Safe on clean clips
    (near no-op). `sample_rate` is accepted for interface symmetry; the STFT
    settings are rate-independent.
    """
    if x.numel() < 32:
        return x
    x = remove_dc(x)
    x = spectral_denoise(x)
    x = auto_gain(x)
    return x


if __name__ == "__main__":
    # Self-test: shape preserved; noisy signal comes out with higher SNR.
    import numpy as np
    sr = SAMPLE_RATE
    t = torch.linspace(0, 1, sr)
    clean = 0.3 * torch.sin(2 * np.pi * 300 * t)
    noisy = clean + 0.05 * torch.randn(sr)
    out = enhance_waveform(noisy)
    print(f"in {tuple(noisy.shape)} -> out {tuple(out.shape)}")
    print(f"noisy RMS {noisy.pow(2).mean().sqrt():.3f} -> enhanced RMS {out.pow(2).mean().sqrt():.3f}")
