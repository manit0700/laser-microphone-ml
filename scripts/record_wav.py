"""
record_wav.py
=============
Record spoken digits from the laptop/computer microphone and save them as WAV
files into data/raw/, ready for training/testing.

WHY
---
This lets the team build its OWN small spoken-digit set (e.g. to sanity-check the
pipeline with our own voices) without downloading anything. Files are named in
the FSDD style the dataset already understands:

        <digit>_<speaker>_<index>.wav      e.g.  7_manit_0.wav

so `dataset.py` picks up the label (the first token) automatically.

REQUIRES
--------
    pip install sounddevice
(sounddevice needs PortAudio; on macOS/Windows the pip wheel includes it.)

USAGE
-----
Record 5 takes of digit 7 as speaker "manit":
    python scripts/record_wav.py --digit 7 --speaker manit --takes 5

Record one take of every digit 0-9 for speaker "alex":
    python scripts/record_wav.py --all --speaker alex --takes 1

Options:
    --digit N        which digit to record (0-9)
    --all            loop over all digits 0-9 instead of a single --digit
    --speaker NAME   speaker label used in the filename (default: "me")
    --takes K        how many recordings per digit (default: 3)
    --seconds S      length of each recording in seconds (default: 1.5)

LASER NOTE
----------
This records from a normal microphone. Laser/DAQ capture will instead produce
CSV or a live stream -- use scripts/wav_from_csv.py or scripts/array_to_wav.py
for those. All three end up as WAV files the same pipeline can read.
"""

import argparse
import sys
from pathlib import Path

# Make src/ importable so we can reuse config + the shared save_wav helper.
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from config import DIGIT_LABELS, RAW_DATA_DIR, SAMPLE_RATE  # noqa: E402
from preprocess import save_wav  # noqa: E402


def _next_index(out_dir: Path, digit: str, speaker: str) -> int:
    """Find the next free index so we never overwrite existing recordings."""
    existing = list(out_dir.glob(f"{digit}_{speaker}_*.wav"))
    return len(existing)


def record_one(digit: str, speaker: str, index: int, seconds: float, out_dir: Path):
    """Record a single clip and save it. Returns the written path."""
    import sounddevice as sd  # imported here so the script loads even if missing

    print(f"  >> Get ready to say '{digit}' ... recording {seconds:.1f}s NOW")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()  # block until the recording finishes
    path = out_dir / f"{digit}_{speaker}_{index}.wav"
    save_wav(audio.reshape(-1), path, SAMPLE_RATE)
    print(f"     saved {path.name}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Record spoken-digit WAV files.")
    parser.add_argument("--digit", type=str, choices=DIGIT_LABELS,
                        help="single digit to record (0-9)")
    parser.add_argument("--all", action="store_true",
                        help="record every digit 0-9")
    parser.add_argument("--speaker", type=str, default="me",
                        help="speaker label used in the filename")
    parser.add_argument("--takes", type=int, default=3,
                        help="recordings per digit")
    parser.add_argument("--seconds", type=float, default=1.5,
                        help="length of each recording (seconds)")
    args = parser.parse_args()

    if not args.all and args.digit is None:
        parser.error("specify --digit N or --all")

    try:
        import sounddevice  # noqa: F401
    except Exception:
        print("ERROR: this tool needs the 'sounddevice' package.")
        print("Install it with:  pip install sounddevice")
        sys.exit(1)

    out_dir = Path(RAW_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    digits = DIGIT_LABELS if args.all else [args.digit]
    print(f"Recording at {SAMPLE_RATE} Hz, speaker='{args.speaker}', "
          f"{args.takes} take(s) per digit. Files go to {out_dir}")

    for digit in digits:
        start = _next_index(out_dir, digit, args.speaker)
        print(f"\nDigit '{digit}':")
        for t in range(args.takes):
            input(f"  Press ENTER to record take {t + 1}/{args.takes} of '{digit}'...")
            record_one(digit, args.speaker, start + t, args.seconds, out_dir)

    total = len(list(out_dir.glob("*.wav")))
    print(f"\nDone. data/raw/ now has {total} WAV files. Next: python src/train.py")


if __name__ == "__main__":
    main()
