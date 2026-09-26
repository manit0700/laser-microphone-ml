"""
record_daq_wav.py
==================
Record spoken digits through the PCM1808/laser DAQ hardware and save them as
WAV files into data/raw/, ready for training. Same idea as record_wav.py, but
captures from the real hardware (signal_backend._DAQSource) instead of a
laptop microphone, so the model can actually learn what a laser-transduced
digit looks like instead of only ever seeing clean voice recordings.

Files are named in the FSDD style the dataset already understands:

        <digit>_<speaker>_<index>.wav      e.g.  7_manit_laser_0.wav

so `dataset.py` picks up the label (the first token) automatically -- just
run `python src/train.py` afterward to fold these into training.

REQUIRES
--------
    pip install pyaudio
(needs the PCM1808 connected; see docs/laser_daq_interface.md)

USAGE
-----
Record 5 takes of digit 7 as speaker "manit_laser":
    python scripts/record_daq_wav.py --digit 7 --speaker manit_laser --takes 5

Record one take of every digit 0-9:
    python scripts/record_daq_wav.py --all --speaker manit_laser --takes 1

Options:
    --digit N        which digit to record (0-9)
    --all            loop over all digits 0-9 instead of a single --digit
    --speaker NAME   speaker label used in the filename (default: "laser")
    --takes K        how many recordings per digit (default: 3)
    --seconds S      length of each recording in seconds (default: 1.5)

Wrong device picked up? Same override as the rest of the DAQ tooling:
    LMML_DAQ_DEVICE=<index> python scripts/record_daq_wav.py --all
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import DIGIT_LABELS, RAW_DATA_DIR, SAMPLE_RATE  # noqa: E402
from preprocess import load_waveform_from_array, save_wav  # noqa: E402
from signal_backend import _DAQSource  # noqa: E402


def _next_index(out_dir: Path, digit: str, speaker: str) -> int:
    """Find the next free index so we never overwrite existing recordings."""
    existing = list(out_dir.glob(f"{digit}_{speaker}_*.wav"))
    return len(existing)


def record_one(daq: _DAQSource, digit: str, speaker: str, index: int,
               seconds: float, out_dir: Path):
    """Record one clip from the already-open DAQ stream and save it."""
    print(f"  >> Say '{digit}' into the laser NOW ({seconds:.1f}s)")
    time.sleep(seconds)
    raw = daq.latest(seconds)

    if raw.size < int(0.5 * daq.rate):
        print(f"     WARNING: only {raw.size} samples captured, skipping this take")
        return None

    # Resamples from the DAQ's native rate down to the project SAMPLE_RATE,
    # same conversion every other capture path (mic, file replay) goes through.
    waveform = load_waveform_from_array(raw, daq.rate)
    path = out_dir / f"{digit}_{speaker}_{index}.wav"
    save_wav(waveform, path, SAMPLE_RATE)
    print(f"     saved {path.name}  ({raw.size} samples @ {daq.rate}Hz -> {SAMPLE_RATE}Hz)")
    return path


def main():
    parser = argparse.ArgumentParser(description="Record spoken-digit WAV files via the PCM1808/laser DAQ.")
    parser.add_argument("--digit", type=str, choices=DIGIT_LABELS,
                        help="single digit to record (0-9)")
    parser.add_argument("--all", action="store_true",
                        help="record every digit 0-9")
    parser.add_argument("--speaker", type=str, default="laser",
                        help="speaker label used in the filename")
    parser.add_argument("--takes", type=int, default=3,
                        help="recordings per digit")
    parser.add_argument("--seconds", type=float, default=1.5,
                        help="length of each recording (seconds)")
    args = parser.parse_args()

    if not args.all and args.digit is None:
        parser.error("specify --digit N or --all")

    out_dir = Path(RAW_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    daq = _DAQSource(buffer_seconds=max(args.seconds + 1.0, 2.0))
    if not daq.available:
        print(f"ERROR: DAQ not available: {daq.error}")
        print("Check: pyaudio installed, PCM1808 connected, LMML_DAQ_DEVICE if auto-detect picked wrong.")
        sys.exit(1)

    print(f"Device: [{daq.device}] {daq.device_name}")
    print(f"Recording at {daq.rate} Hz (resampled to {SAMPLE_RATE} Hz), speaker='{args.speaker}', "
          f"{args.takes} take(s) per digit. Files go to {out_dir}")

    digits = DIGIT_LABELS if args.all else [args.digit]

    daq.start()
    try:
        for digit in digits:
            start = _next_index(out_dir, digit, args.speaker)
            print(f"\nDigit '{digit}':")
            for t in range(args.takes):
                input(f"  Press ENTER to record take {t + 1}/{args.takes} of '{digit}'...")
                record_one(daq, digit, args.speaker, start + t, args.seconds, out_dir)
    finally:
        daq.stop()

    total = len(list(out_dir.glob("*.wav")))
    print(f"\nDone. data/raw/ now has {total} WAV files. Next: python src/train.py --augment")


if __name__ == "__main__":
    main()
