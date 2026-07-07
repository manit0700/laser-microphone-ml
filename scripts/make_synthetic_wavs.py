"""
make_synthetic_wavs.py
======================
Generate FAKE labeled spoken-digit WAV files so you can run the ENTIRE pipeline
(train -> evaluate -> predict) without downloading any dataset.

WHY
---
Useful for: first-time setup checks, CI/testing, demoing the pipeline, or when
you have no internet. Each digit is given a distinct tone pattern, so the model
can actually learn to tell them apart (you should see accuracy climb). These are
NOT real speech -- use scripts/download_fsdd.py for real audio.

USAGE
-----
    python scripts/make_synthetic_wavs.py                 # 6 takes x 3 speakers x 10 digits
    python scripts/make_synthetic_wavs.py --per-digit 10  # more samples per digit

Files are written FSDD-style to data/raw/ as  <digit>_<speaker>_<index>.wav
so dataset.py labels them automatically.

LASER NOTE
----------
This is only for pipeline plumbing tests. Replace with real audio (FSDD) or laser
captures (wav_from_csv.py / record_wav.py) for any real results.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from config import DIGIT_LABELS, RAW_DATA_DIR, SAMPLE_RATE  # noqa: E402
from preprocess import save_wav  # noqa: E402


def make_digit_clip(digit: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Build a 1-second clip whose tone depends on the digit (so it's learnable)."""
    base = 200 + digit * 60  # each digit gets a different fundamental frequency
    t = np.linspace(0, 1, sr, dtype="float32")
    clip = np.zeros(sr, dtype="float32")

    # Place the "spoken" burst at a random position to mimic real timing variation.
    start = 1500 + int(rng.integers(0, 1500))
    end = min(start + 3000, sr)
    seg = t[start:end]
    clip[start:end] = (0.5 * np.sin(2 * np.pi * base * seg)
                       + 0.2 * np.sin(2 * np.pi * 2 * base * seg))
    clip += 0.01 * rng.standard_normal(sr).astype("float32")  # light noise
    return clip


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic digit WAVs.")
    parser.add_argument("--per-digit", type=int, default=18,
                        help="total clips to generate per digit (default: 18)")
    parser.add_argument("--speakers", type=int, default=3,
                        help="number of pseudo-speakers (default: 3)")
    args = parser.parse_args()

    out_dir = Path(RAW_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    takes = max(1, args.per_digit // args.speakers)
    count = 0
    for d in range(len(DIGIT_LABELS)):
        for spk in range(args.speakers):
            for idx in range(takes):
                clip = make_digit_clip(d, SAMPLE_RATE, rng)
                save_wav(clip, out_dir / f"{d}_spk{spk}_{idx}.wav", SAMPLE_RATE)
                count += 1

    print(f"Generated {count} synthetic WAV files in {out_dir}")
    print("Next: python src/train.py")


if __name__ == "__main__":
    main()
