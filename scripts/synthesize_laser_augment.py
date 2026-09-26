"""
synthesize_laser_augment.py
============================
No public laser-microphone spoken-digit dataset exists to download (checked --
see the LDV/laser-speech research papers this project's ML lead looked at).
Recording thousands of real laser digit clips by hand isn't practical either.
What published LDV speech-recognition research actually does instead: record a
SMALL set of real laser clips, learn how the laser/DAQ path colors a voice
signal (its rough frequency response + noise floor), then apply that same
coloring to a large existing clean-voice dataset to synthesize realistic
laser-domain training data at scale.

This script does that:
    1. Loads your real laser clips (scripts/record_daq_wav.py output).
    2. Estimates the laser path's average spectral "coloring" relative to
       clean speech (a smoothed magnitude filter + noise floor sample).
    3. Applies that coloring to clean clips already in data/raw/ (FSDD +
       Speech Commands + your own voice), writing new synthetic clips.

This is NOT a replacement for real laser data -- it's a way to get orders of
magnitude more training examples out of the small amount of real laser data
you do have. Keep recording real clips over time; retrain periodically with
a mix of real + synthetic.

REQUIRES
--------
At least ~10-20 real laser clips already recorded:
    python scripts/record_daq_wav.py --all --speaker manit_laser --takes 2

USAGE
-----
    python scripts/synthesize_laser_augment.py --laser-speaker manit_laser \
        --out-speaker lasersynth --per-digit 200

Then retrain:
    python src/train.py --model lstm --augment
    python src/train.py --model cnn --augment
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import DIGIT_LABELS, RAW_DATA_DIR, SAMPLE_RATE  # noqa: E402
from preprocess import load_waveform_from_array, save_wav  # noqa: E402


def _load_mono(path: Path) -> np.ndarray:
    """Read a WAV and return it as mono float32 at the project SAMPLE_RATE."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    wave = load_waveform_from_array(mono, sr)  # resamples to SAMPLE_RATE
    return wave.cpu().numpy().astype(np.float32)


def _avg_spectrum(clips: list[np.ndarray], n_fft: int) -> np.ndarray:
    """Average magnitude spectrum across a list of variable-length clips."""
    total = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    count = 0
    for clip in clips:
        if clip.size < n_fft:
            clip = np.pad(clip, (0, n_fft - clip.size))
        # Average over non-overlapping n_fft windows so long/short clips
        # contribute comparably.
        n_windows = clip.size // n_fft
        for i in range(n_windows):
            seg = clip[i * n_fft:(i + 1) * n_fft]
            spec = np.abs(np.fft.rfft(seg * np.hanning(n_fft)))
            total += spec
            count += 1
    if count == 0:
        raise RuntimeError("No usable audio windows found to build a spectrum from.")
    return total / count


def _smooth(spectrum: np.ndarray, width: int = 5) -> np.ndarray:
    """Light moving-average smoothing so the filter follows the coarse shape
    of the laser path's response, not per-clip noise."""
    kernel = np.ones(width) / width
    return np.convolve(spectrum, kernel, mode="same")


def build_laser_filter(laser_clips: list[np.ndarray], clean_clips: list[np.ndarray],
                        n_fft: int = 1024, max_gain_db: float = 18.0) -> np.ndarray:
    """Estimate the laser path's relative frequency response vs. clean speech.

    Returns a magnitude filter (n_fft//2 + 1 bins) to multiply onto a clean
    clip's spectrum. Clipped to +/-max_gain_db so a couple of noisy bins can't
    blow up the result.
    """
    laser_spec = _smooth(_avg_spectrum(laser_clips, n_fft))
    clean_spec = _smooth(_avg_spectrum(clean_clips, n_fft))
    eps = clean_spec.max() * 1e-6 + 1e-12
    ratio = laser_spec / (clean_spec + eps)
    max_gain = 10 ** (max_gain_db / 20)
    return np.clip(ratio, 1.0 / max_gain, max_gain)


def laser_noise_floor(laser_clips: list[np.ndarray], seconds: float = 0.2) -> np.ndarray:
    """Grab a short quiet-ish slice from a real laser clip to use as a noise
    bed, so synthesized clips also carry the sensor's characteristic hiss."""
    n = int(seconds * SAMPLE_RATE)
    candidates = [c for c in laser_clips if c.size >= n]
    if not candidates:
        return np.zeros(n, dtype=np.float32)
    clip = random.choice(candidates)
    # Use the quietest window in the clip (likely between/before words).
    best_start, best_rms = 0, float("inf")
    step = max(1, n // 4)
    for start in range(0, clip.size - n, step):
        seg = clip[start:start + n]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        if rms < best_rms:
            best_rms, best_start = rms, start
    return clip[best_start:best_start + n]


def apply_filter(clean: np.ndarray, filt: np.ndarray, n_fft: int,
                  noise_bed: np.ndarray, noise_gain: float = 0.3) -> np.ndarray:
    """Color one clean clip with the laser filter + a bit of laser noise."""
    pad_len = ((clean.size + n_fft - 1) // n_fft) * n_fft
    padded = np.pad(clean, (0, pad_len - clean.size))
    out = np.zeros_like(padded)
    hop = n_fft // 2
    window = np.hanning(n_fft)
    for start in range(0, pad_len - n_fft + 1, hop):
        seg = padded[start:start + n_fft] * window
        spec = np.fft.rfft(seg)
        colored = np.fft.irfft(spec * filt, n=n_fft)
        out[start:start + n_fft] += colored
    result = out[:clean.size]

    if noise_bed.size > 0:
        reps = int(np.ceil(result.size / noise_bed.size))
        noise = np.tile(noise_bed, reps)[:result.size]
        peak = np.abs(result).max() + 1e-9
        result = result + noise_gain * (peak / (np.abs(noise_bed).max() + 1e-9)) * noise

    peak = np.abs(result).max()
    if peak > 1e-6:
        result = result / peak * 0.95
    return result.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--laser-speaker", type=str, required=True,
                        help="speaker tag used when recording real laser clips "
                             "(the <speaker> in <digit>_<speaker>_<n>.wav)")
    parser.add_argument("--out-speaker", type=str, default="lasersynth",
                        help="speaker tag for the synthesized output clips")
    parser.add_argument("--per-digit", type=int, default=200,
                        help="how many synthetic clips to generate per digit")
    parser.add_argument("--n-fft", type=int, default=1024,
                        help="FFT window size for the spectral filter")
    args = parser.parse_args()

    raw_dir = Path(RAW_DATA_DIR)
    laser_paths = sorted(raw_dir.glob(f"*_{args.laser_speaker}_*.wav"))
    if len(laser_paths) < 5:
        print(f"ERROR: found only {len(laser_paths)} clips matching "
              f"'*_{args.laser_speaker}_*.wav' in {raw_dir}.")
        print("Record more first, e.g.:")
        print(f"  python scripts/record_daq_wav.py --all --speaker {args.laser_speaker} --takes 3")
        sys.exit(1)
    print(f"Loading {len(laser_paths)} real laser clips (speaker='{args.laser_speaker}')...")
    laser_clips = [_load_mono(p) for p in laser_paths]

    # "Clean" reference: everything else in data/raw/ that isn't laser or
    # already-synthetic, sampled down to roughly match the laser set's size
    # so neither dominates the averaged spectrum unfairly.
    all_clean_paths = [p for p in raw_dir.glob("*.wav")
                        if f"_{args.laser_speaker}_" not in p.name
                        and f"_{args.out_speaker}_" not in p.name]
    if len(all_clean_paths) < 10:
        print(f"ERROR: not enough clean reference clips found in {raw_dir}.")
        sys.exit(1)
    sample_n = min(len(all_clean_paths), max(200, len(laser_clips) * 5))
    clean_sample_paths = random.sample(all_clean_paths, sample_n)
    print(f"Loading {len(clean_sample_paths)} clean reference clips for comparison...")
    clean_sample = [_load_mono(p) for p in clean_sample_paths]

    print("Estimating laser path's spectral coloring...")
    filt = build_laser_filter(laser_clips, clean_sample, n_fft=args.n_fft)
    noise_bed = laser_noise_floor(laser_clips)

    print(f"Synthesizing {args.per_digit} clips per digit (speaker='{args.out_speaker}')...")
    total = 0
    for digit in DIGIT_LABELS:
        # Pull clean source clips labeled with this digit to color.
        digit_paths = [p for p in all_clean_paths if p.name.startswith(f"{digit}_")]
        if not digit_paths:
            print(f"  digit {digit}: no clean source clips found, skipping")
            continue
        existing = len(list(raw_dir.glob(f"{digit}_{args.out_speaker}_*.wav")))
        made = 0
        attempts = 0
        while made < args.per_digit and attempts < args.per_digit * 3:
            attempts += 1
            src_path = random.choice(digit_paths)
            try:
                clean = _load_mono(src_path)
            except Exception:  # noqa: BLE001 - skip unreadable source clips
                continue
            colored = apply_filter(clean, filt, args.n_fft, noise_bed)
            out_path = raw_dir / f"{digit}_{args.out_speaker}_{existing + made}.wav"
            save_wav(colored, out_path, SAMPLE_RATE)
            made += 1
        print(f"  digit {digit}: {made} clips")
        total += made

    print(f"\nDone. Wrote {total} synthetic laser-domain clips into {raw_dir}.")
    print("Retrain with:\n  python src/train.py --model lstm --augment\n  python src/train.py --model cnn --augment")


if __name__ == "__main__":
    main()
