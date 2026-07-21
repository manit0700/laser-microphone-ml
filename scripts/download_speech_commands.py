"""
download_speech_commands.py
===========================
Download the Google Speech Commands dataset (from the web) and add its spoken
digits 0-9 to data/raw/, so the model trains on THOUSANDS of speakers instead of
just the ~6 in FSDD. This is what makes recognition work for *any* voice out of
the box.

WHY
---
FSDD is small (6 speakers) -> the model overfits to those voices and misreads
new speakers. Google Speech Commands has ~2,000-4,000 clips per digit word from
thousands of different people. Training on it (plus FSDD, plus your own voice)
gives real speaker-independence.

WHAT IT DOES
------------
1. Downloads Speech Commands via torchaudio (cached; ~2.3 GB the first time).
2. Keeps only the digit words zero..nine, maps them to labels 0..9.
3. Writes them into data/raw/ as  <digit>_sc<speaker>_<n>.wav  so the existing
   dataset.py picks up the label automatically -- no pipeline changes.
4. Optionally adds an 'unknown' class (non-digit words + background noise) so the
   model can actively learn to REJECT non-digits, not just threshold them.

Speech Commands is 16 kHz; the pipeline resamples to the project rate on load.

RUN
---
    python scripts/download_speech_commands.py                 # digits only
    python scripts/download_speech_commands.py --per-class 600 # cap per digit
    python scripts/download_speech_commands.py --unknown       # add unknown class

Then retrain on the combined data:
    python src/train.py --model lstm --augment

NOTE
----
Capping per class keeps training fast on a laptop CPU while still covering many
speakers. Raise --per-class for more variety if you have the time/compute.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import RAW_DATA_DIR, UNKNOWN_LABEL  # noqa: E402
from preprocess import save_wav  # noqa: E402

WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
# A few non-digit words to represent the 'unknown' class (only with --unknown).
UNKNOWN_WORDS = {"yes", "no", "up", "down", "stop", "go", "on", "off"}


def main():
    parser = argparse.ArgumentParser(description="Add Speech Commands digits to data/raw.")
    parser.add_argument("--per-class", type=int, default=500,
                        help="max clips to keep per digit (default 500)")
    parser.add_argument("--unknown", action="store_true",
                        help="also add non-digit words as an 'unknown' class")
    parser.add_argument("--per-unknown", type=int, default=400,
                        help="max clips for the unknown class (with --unknown)")
    args = parser.parse_args()

    try:
        import torchaudio
    except ImportError:
        print("Needs torchaudio (already in requirements). pip install torchaudio")
        return

    print("Downloading / loading Google Speech Commands (cached; ~2.3 GB first time)...")
    root = Path(RAW_DATA_DIR).parent / "speech_commands_cache"
    root.mkdir(parents=True, exist_ok=True)
    ds = torchaudio.datasets.SPEECHCOMMANDS(root=str(root), download=True)

    out = Path(RAW_DATA_DIR)
    out.mkdir(parents=True, exist_ok=True)

    kept = {d: 0 for d in WORD_TO_DIGIT.values()}
    unk = 0
    caps = {d: args.per_class for d in WORD_TO_DIGIT.values()}

    for i in range(len(ds)):
        waveform, sr, label, speaker_id, _ = ds[i]
        if label in WORD_TO_DIGIT:
            digit = WORD_TO_DIGIT[label]
            if kept[digit] >= caps[digit]:
                continue
            name = f"{digit}_sc{speaker_id}_{kept[digit]}.wav"
            save_wav(waveform.squeeze(0).numpy(), out / name, sr)
            kept[digit] += 1
        elif args.unknown and label in UNKNOWN_WORDS and unk < args.per_unknown:
            # 'unknown' clips go in a subfolder dataset.py reads as the unknown class.
            udir = out / UNKNOWN_LABEL
            udir.mkdir(exist_ok=True)
            save_wav(waveform.squeeze(0).numpy(), udir / f"unk_{unk}.wav", sr)
            unk += 1

        if i % 2000 == 0:
            done = sum(kept.values())
            print(f"  scanned {i}/{len(ds)} | digits kept {done} | unknown {unk}")

    print("\nAdded per digit:")
    for d in sorted(kept):
        print(f"  {d}: {kept[d]}")
    total = sum(kept.values())
    print(f"Total digit clips added: {total}" + (f" | unknown: {unk}" if args.unknown else ""))
    print(f"\nData now in {out}. Retrain with:\n"
          f"  python src/train.py --model lstm --augment")
    if args.unknown:
        print(f"(Unknown-word clips saved under {out / UNKNOWN_LABEL}/ for a future "
              "trained 'unknown' class; current pipeline still rejects non-digits "
              "via the confidence threshold.)")


if __name__ == "__main__":
    main()
