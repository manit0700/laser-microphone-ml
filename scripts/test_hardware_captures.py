"""
Batch-test real hardware captures against the ML model.

Put PCM1808 / DAQ test files in:
    data/hardware_test/

Recommended filenames:
    0_test1.wav
    1_test1.wav
    2_pcm1808_01.wav
    3_daq_capture.csv

The first digit in the filename is treated as the expected label. The script
prints a table and writes results/hardware_test_report.csv for Sprint 4 notes.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.append(str(SRC))

from config import SAMPLE_RATE  # noqa: E402
from predict import infer, infer_ensemble, predict_file  # noqa: E402
from preprocess import load_waveform_from_array  # noqa: E402


SUPPORTED_AUDIO = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
SUPPORTED_CAPTURE = SUPPORTED_AUDIO | {".csv"}


def expected_digit(path: Path) -> str:
    """Use the first digit in the filename as the expected label."""
    for char in path.stem:
        if char.isdigit():
            return char
    return ""


def detect_sample_rate(df: pd.DataFrame, time_col: str) -> int | None:
    """Estimate sample rate from a CSV time column."""
    if time_col not in df.columns:
        return None
    times = pd.to_numeric(df[time_col], errors="coerce").dropna().values
    if len(times) < 2:
        return None
    dt = (times[-1] - times[0]) / (len(times) - 1)
    if dt <= 0:
        return None
    return int(round(1.0 / dt))


def read_csv_capture(
    path: Path,
    signal_col: str,
    time_col: str,
    sample_rate: int | None,
) -> tuple[np.ndarray, int]:
    """Read a DAQ-style CSV capture into signal samples and sample rate."""
    df = pd.read_csv(path)
    selected_col = signal_col
    if selected_col not in df.columns:
        numeric = df.select_dtypes("number").columns.tolist()
        if not numeric:
            raise ValueError(f"no numeric columns found in CSV. Columns: {list(df.columns)}")
        selected_col = numeric[-1]
    signal = pd.to_numeric(df[selected_col], errors="coerce").dropna().values.astype(np.float32)
    sr = sample_rate or detect_sample_rate(df, time_col) or SAMPLE_RATE
    return signal, int(sr)


def read_capture(
    path: Path,
    csv_signal_col: str,
    csv_time_col: str,
    sample_rate: int | None,
) -> tuple[np.ndarray, int]:
    """Read WAV/audio or DAQ CSV into raw mono samples and sample rate."""
    if path.suffix.lower() == ".csv":
        return read_csv_capture(path, csv_signal_col, csv_time_col, sample_rate)

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr, int(sr)


def capture_stats(samples: np.ndarray, sr: int) -> dict:
    """Basic signal quality numbers for debugging hardware captures."""
    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    rms = float(np.sqrt(np.mean(arr**2))) if arr.size else 0.0
    duration = float(arr.size / sr) if sr else 0.0
    if peak < 0.02:
        issue = "very quiet / possible input permission or gain issue"
    elif peak > 0.98:
        issue = "possible clipping"
    else:
        issue = "ok"
    return {"sample_rate": int(sr), "duration_sec": duration, "peak": peak, "rms": rms, "issue": issue}


def predict_capture(path: Path, samples: np.ndarray, sr: int, model: str) -> dict:
    """Predict from audio path or raw CSV samples."""
    if path.suffix.lower() in SUPPORTED_AUDIO:
        return predict_file(path, model=model)

    waveform = load_waveform_from_array(samples, sample_rate=sr)
    if model == "ensemble":
        result, _ = infer_ensemble(waveform)
    else:
        result, _ = infer(waveform, model_type=model)
    return result


def run(
    input_dir: Path,
    out_csv: Path,
    model: str,
    limit: int | None,
    csv_signal_col: str,
    csv_time_col: str,
    sample_rate: int | None,
) -> int:
    input_dir = input_dir.resolve()
    out_csv = out_csv.resolve()
    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED_CAPTURE)
    if limit:
        files = files[:limit]
    if not files:
        print(f"No WAV/CSV captures found in {input_dir}")
        print("Add files like data/hardware_test/0_test1.wav or 0_capture.csv and run again.")
        return 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    print("file, expected, predicted, confidence, status, peak, rms, sample_rate, issue")
    for path in files:
        expected = expected_digit(path)
        try:
            samples, sr = read_capture(path, csv_signal_col, csv_time_col, sample_rate)
            stats = capture_stats(samples, sr)
            result = predict_capture(path, samples, sr, model)
            predicted = str(result.get("prediction", ""))
            confidence = float(result.get("confidence", 0.0))
            status = str(result.get("status", ""))
            correct = bool(expected and predicted == expected and status == "recognized")
            error = ""
        except Exception as exc:  # noqa: BLE001 - report every capture failure
            predicted = ""
            confidence = 0.0
            status = "error"
            correct = False
            error = f"{type(exc).__name__}: {exc}"
            stats = {"sample_rate": 0, "duration_sec": 0.0, "peak": 0.0, "rms": 0.0, "issue": "read/predict error"}

        row = {
            "file": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
            "expected": expected,
            "predicted": predicted,
            "confidence": f"{confidence:.4f}",
            "status": status,
            "correct": correct,
            "sample_rate": stats["sample_rate"],
            "duration_sec": f"{stats['duration_sec']:.3f}",
            "peak": f"{stats['peak']:.4f}",
            "rms": f"{stats['rms']:.4f}",
            "issue": stats["issue"],
            "error": error,
        }
        rows.append(row)
        print(
            f"{path.name}, {expected or '-'}, {predicted or '-'}, {confidence:.2f}, "
            f"{status}, {stats['peak']:.3f}, {stats['rms']:.3f}, "
            f"{stats['sample_rate']}, {stats['issue']}"
        )
        if error:
            print(json.dumps({"file": path.name, "error": error}, indent=2))

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    known = [r for r in rows if r["expected"]]
    if known:
        correct = sum(1 for r in known if r["correct"] is True)
        print(f"\nAccuracy on labeled captures: {correct}/{len(known)}")
    print(f"Report written to: {out_csv}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Test PCM1808/DAQ captures with the ML model.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "data" / "hardware_test",
        help="folder containing captured audio files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "hardware_test_report.csv",
        help="CSV report output path",
    )
    parser.add_argument(
        "--model",
        choices=["lstm", "cnn", "ensemble"],
        default="ensemble",
        help="model to use for predictions",
    )
    parser.add_argument("--limit", type=int, default=None, help="test only the first N capture files")
    parser.add_argument(
        "--csv-signal-col",
        default="voltage",
        help="CSV signal column name; falls back to last numeric column",
    )
    parser.add_argument(
        "--csv-time-col",
        default="time",
        help="CSV time column used to auto-detect sample rate",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="force sample rate for CSV captures when no reliable time column exists",
    )
    args = parser.parse_args()
    raise SystemExit(
        run(
            args.input_dir,
            args.out,
            args.model,
            args.limit,
            args.csv_signal_col,
            args.csv_time_col,
            args.sample_rate,
        )
    )


if __name__ == "__main__":
    main()
