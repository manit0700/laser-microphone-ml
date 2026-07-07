"""
download_fsdd.py
================
Convenience downloader for the Free Spoken Digit Dataset (FSDD).

FSDD is a small, public set of ~3,000 WAV recordings of digits 0-9 spoken by
several people, sampled at 8 kHz. It is perfect for developing this pipeline
before real laser-microphone data exists.

WHAT IT DOES
------------
Clones the FSDD GitHub repo and copies its WAV files into data/raw/ where
dataset.py expects them. FSDD filenames look like "7_jackson_32.wav", which our
dataset already understands (label = first token).

RUN
---
    python scripts/download_fsdd.py

Requires `git` to be installed. If you cannot use git, manually download the
recordings folder from https://github.com/Jakobovski/free-spoken-digit-dataset
and copy the .wav files into data/raw/.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
FSDD_REPO = "https://github.com/Jakobovski/free-spoken-digit-dataset.git"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        print(f"Cloning FSDD into a temporary folder...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", FSDD_REPO, tmp + "/fsdd"],
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"ERROR: git clone failed ({e}).")
            print("Install git, or download WAVs manually (see this file's docstring).")
            sys.exit(1)

        recordings = Path(tmp) / "fsdd" / "recordings"
        wavs = list(recordings.glob("*.wav"))
        if not wavs:
            print("ERROR: no WAV files found in the cloned repo.")
            sys.exit(1)

        print(f"Copying {len(wavs)} WAV files into {RAW_DIR} ...")
        for wav in wavs:
            shutil.copy2(wav, RAW_DIR / wav.name)

    print(f"Done. {len(list(RAW_DIR.glob('*.wav')))} WAV files now in data/raw/.")
    print("Next: python src/train.py")


if __name__ == "__main__":
    main()
