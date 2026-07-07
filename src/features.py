"""
features.py
===========
Converts a clean waveform into MFCC features (the main feature) and, optionally,
mel spectrograms (the backup/comparison feature) and plain spectrograms (for
visualization only).

PIPELINE STAGE
--------------
clean waveform (preprocess.py)  ->  [THIS FILE]  ->  LSTM model input (model.py)

WHY MFCCs?
----------
Raw audio has thousands of samples per second; that is too much, and most of it
is irrelevant to *which digit* was said. MFCCs (Mel-Frequency Cepstral
Coefficients) summarize each short time-frame of audio into a small set of
numbers (here N_MFCC = 13) that capture the *timbre/shape* of the sound the way
the human ear perceives it. They are the classic, reliable features for speech
tasks and are cheap to compute, which keeps the model laptop-friendly.

OUTPUT SHAPE  (very important for the LSTM)
-------------------------------------------
We return MFCCs as a 2-D tensor of shape:

        (time_steps, n_mfcc)

    - time_steps : how many frames the clip was split into (the sequence length)
    - n_mfcc     : features per frame (= N_MFCC, default 13)

The LSTM reads this as a SEQUENCE: one vector of `n_mfcc` numbers per time step.
This layout is exactly `batch_first=True` friendly: a batch becomes
(batch, time_steps, n_mfcc). See model.py.

LASER NOTE
----------
MFCC settings (N_FFT, HOP_LENGTH, N_MELS) are tuned for speech-rate audio. Laser
signals at a very different sample rate may need different window sizes. Adjust
them in config.py; the function signatures here stay the same.
"""

import torch
import torchaudio.transforms as T

from config import (
    HOP_LENGTH,
    MEL_N_MELS,
    N_FFT,
    N_MELS,
    N_MFCC,
    SAMPLE_RATE,
)


# ---------------------------------------------------------------------------
# Reusable transform objects (built once, reused for every sample = faster).
# ---------------------------------------------------------------------------
_mfcc_transform = T.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=N_MFCC,
    melkwargs={
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "center": True,
    },
)

_mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=MEL_N_MELS,
)

_spectrogram_transform = T.Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH)
_db_transform = T.AmplitudeToDB()


def extract_mfcc(waveform: torch.Tensor) -> torch.Tensor:
    """MAIN FEATURE. Compute MFCCs from a 1-D waveform.

    Input : waveform tensor of shape (num_samples,)
    Output: tensor of shape (time_steps, n_mfcc)  -- ready for the LSTM.

    We also standardize each coefficient (zero mean, unit variance) across time,
    which helps the LSTM train faster and more stably.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # MFCC transform expects (channel, samples)

    mfcc = _mfcc_transform(waveform)      # shape: (1, n_mfcc, time_steps)
    mfcc = mfcc.squeeze(0)                # shape: (n_mfcc, time_steps)

    # Per-coefficient normalization over the time axis.
    mean = mfcc.mean(dim=1, keepdim=True)
    std = mfcc.std(dim=1, keepdim=True) + 1e-8
    mfcc = (mfcc - mean) / std

    # Transpose to (time_steps, n_mfcc) so the LSTM sees a sequence of frames.
    return mfcc.transpose(0, 1).contiguous()


def extract_mel_spectrogram(waveform: torch.Tensor) -> torch.Tensor:
    """BACKUP/COMPARISON FEATURE. Compute a log-mel spectrogram.

    Useful for a future CNN comparison model and for visualization.
    Input : (num_samples,)   Output: (n_mels, time_steps) in decibels.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    mel = _mel_transform(waveform)        # (1, n_mels, time_steps)
    mel = _db_transform(mel)              # convert power -> decibels
    return mel.squeeze(0)                 # (n_mels, time_steps)


def extract_mel_for_cnn(waveform: torch.Tensor) -> torch.Tensor:
    """Mel spectrogram prepared as a CNN input feature.

    Same as extract_mel_spectrogram (log-mel), but standardized (zero mean, unit
    variance) so the CNN trains stably. Output shape: (n_mels, time_steps). The
    CNN adds the channel dimension itself, treating this like a 1-channel image.
    """
    mel = extract_mel_spectrogram(waveform)          # (n_mels, time_steps), dB
    mean = mel.mean()
    std = mel.std() + 1e-8
    return ((mel - mean) / std).contiguous()


def extract_features(waveform: torch.Tensor, kind: str) -> torch.Tensor:
    """Single entry point that returns the right feature for the chosen model.

    kind="mfcc" -> (time_steps, n_mfcc) for the LSTM
    kind="mel"  -> (n_mels, time_steps) for the CNN
    Keeping this in one place guarantees train/evaluate/predict never mismatch the
    feature to the model.
    """
    if kind == "mfcc":
        return extract_mfcc(waveform)
    if kind == "mel":
        return extract_mel_for_cnn(waveform)
    raise ValueError(f"Unknown feature kind: {kind!r} (expected 'mfcc' or 'mel')")


def extract_spectrogram(waveform: torch.Tensor) -> torch.Tensor:
    """VISUALIZATION ONLY. Plain magnitude spectrogram in decibels.

    Input : (num_samples,)   Output: (freq_bins, time_steps) in decibels.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    spec = _spectrogram_transform(waveform)
    spec = _db_transform(spec)
    return spec.squeeze(0)


if __name__ == "__main__":
    # Self-test with a fake 1-second signal so teammates can see the shapes.
    dummy = torch.randn(SAMPLE_RATE)
    mfcc = extract_mfcc(dummy)
    mel = extract_mel_spectrogram(dummy)
    print(f"MFCC shape           : {tuple(mfcc.shape)}  -> (time_steps, n_mfcc)")
    print(f"Mel spectrogram shape: {tuple(mel.shape)}  -> (n_mels, time_steps)")
    print(f"LSTM input_size should be n_mfcc = {N_MFCC}")
