"""
wav_from_csv.py
===============
Convert laser / DAQ data stored as CSV (voltage vs. time) into a WAV file that
the pipeline can load and classify.

WHY
---
The laser-vibrometry hardware will likely export captures as CSV: one column of
time stamps and one column of measured voltage (the vibration signal). This tool
turns that CSV into a standard WAV so the EXISTING pipeline (preprocess -> MFCC ->
LSTM) works on laser data with no other changes.

EXPECTED CSV FORMAT
-------------------
Flexible. By default it looks for a 'voltage' column (the signal) and, if present,
a 'time' column (used to auto-detect the sample rate). You can override the column
names and sample rate from the command line.

    time,voltage
    0.00000,0.0123
    0.00002,0.0156
    ...

USAGE
-----
Auto-detect sample rate from the 'time' column:
    python scripts/wav_from_csv.py laser_capture.csv --out data/raw/7_laser_0.wav

Specify columns and an explicit sample rate (Hz):
    python scripts/wav_from_csv.py capture.csv --signal-col v --sample-rate 50000 \
        --out data/raw/3_laser_0.wav

Tip: name the output <digit>_laser_<index>.wav so dataset.py auto-labels it.

LASER NOTE
----------
This is one of the main hardware->ML integration points. If your DAQ format
differs (extra header rows, different delimiters, multiple channels), adjust the
pandas.read_csv call below; everything after it is reused.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Make src/ importable for the shared save_wav helper + default sample rate.
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from config import SAMPLE_RATE  # noqa: E402
from preprocess import save_wav  # noqa: E402


def detect_sample_rate(df: pd.DataFrame, time_col: str) -> int | None:
    """Estimate sample rate (Hz) from a time column: 1 / average time step."""
    if time_col not in df.columns:
        return None
    times = pd.to_numeric(df[time_col], errors="coerce").dropna().values
    if len(times) < 2:
        return None
    dt = (times[-1] - times[0]) / (len(times) - 1)
    if dt <= 0:
        return None
    return int(round(1.0 / dt))


def main():
    parser = argparse.ArgumentParser(description="Convert laser/DAQ CSV to WAV.")
    parser.add_argument("csv", help="path to the input CSV file")
    parser.add_argument("--out", required=True, help="output .wav path")
    parser.add_argument("--signal-col", default="voltage",
                        help="name of the signal/voltage column (default: voltage)")
    parser.add_argument("--time-col", default="time",
                        help="name of the time column for rate auto-detect (default: time)")
    parser.add_argument("--sample-rate", type=int, default=None,
                        help="force the sample rate (Hz); overrides auto-detect")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Find the signal column (fall back to the last numeric column if needed).
    signal_col = args.signal_col
    if signal_col not in df.columns:
        numeric = df.select_dtypes("number").columns.tolist()
        if not numeric:
            print(f"ERROR: no numeric columns found in {csv_path}. Columns: {list(df.columns)}")
            sys.exit(1)
        signal_col = numeric[-1]
        print(f"WARNING: column '{args.signal_col}' not found; using '{signal_col}'.")

    signal = pd.to_numeric(df[signal_col], errors="coerce").dropna().values

    # Decide the sample rate: explicit > auto-detected from time > project default.
    sr = args.sample_rate or detect_sample_rate(df, args.time_col) or SAMPLE_RATE
    source = ("explicit" if args.sample_rate else
              "auto-detected" if detect_sample_rate(df, args.time_col) else
              "default")
    print(f"Signal: {len(signal)} samples from column '{signal_col}'.")
    print(f"Sample rate: {sr} Hz ({source}).")

    out = save_wav(signal, args.out, sr)
    print(f"Wrote WAV -> {out}")
    print("You can now run:  python src/predict.py " + str(out))


if __name__ == "__main__":
    main()
