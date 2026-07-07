"""
array_to_wav.py
===============
General helper to save ANY signal array (numpy .npy, plain text, or a Python
list/array in your own code) as a WAV file at a chosen sample rate.

WHY
---
Sometimes the signal is not a CSV and not a live mic -- e.g. a live DAQ buffer you
captured into a numpy array, or a .npy file. This is the catch-all "array -> WAV"
converter so any signal source can join the same pipeline.

TWO WAYS TO USE IT
------------------
1) From the command line, with a .npy or whitespace/CSV text file of numbers:

       python scripts/array_to_wav.py buffer.npy --sample-rate 50000 \
           --out data/raw/3_laser_0.wav

2) From your own Python code (e.g. a live DAQ loop) -- import the helper directly:

       from preprocess import save_wav
       save_wav(daq_buffer, "data/raw/3_laser_0.wav", sample_rate=50000)

   (save_wav lives in src/preprocess.py and is the shared core used everywhere.)

LASER NOTE
----------
For a live stream you usually do NOT need a WAV at all -- feed the buffer straight
to predict.predict_waveform(). Use this tool when you want to SAVE captures to
disk for training or debugging.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from config import SAMPLE_RATE  # noqa: E402
from preprocess import save_wav  # noqa: E402


def load_array(path: Path) -> np.ndarray:
    """Load a 1-D signal from a .npy file or a text/CSV file of numbers."""
    if path.suffix.lower() == ".npy":
        return np.load(path).reshape(-1)
    # Text fallback: handles newline- or comma-separated numbers.
    return np.loadtxt(path, delimiter=",").reshape(-1)


def main():
    parser = argparse.ArgumentParser(description="Save a signal array as a WAV file.")
    parser.add_argument("array", help="input .npy or text file of numbers")
    parser.add_argument("--out", required=True, help="output .wav path")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE,
                        help=f"sample rate in Hz (default: {SAMPLE_RATE})")
    args = parser.parse_args()

    path = Path(args.array)
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        sys.exit(1)

    signal = load_array(path)
    out = save_wav(signal, args.out, args.sample_rate)
    print(f"Loaded {len(signal)} samples; wrote {args.sample_rate} Hz WAV -> {out}")


if __name__ == "__main__":
    main()
