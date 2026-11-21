"""
add_modified_commands.py
========================
Add the MODIFIED Speech Commands digits (google_commands_0_9_mod) to data/raw/,
so the model trains on the original clips AND their modified copies.

WHAT THE DATA IS
----------------
google_commands_0_9_mod has one folder per digit word (zero_mod ... nine_mod).
Every file is a processed copy of the Speech Commands v0.02 clip with the SAME
filename (<speakerhash>_nohash_<k>.wav): same length and about the same loudness,
but a reshaped spectrum (less energy below 300 Hz and above 4 kHz, more at
300-1000 Hz).

WHAT THIS SCRIPT DOES
---------------------
Writes every clip into data/raw/ as     <digit>_sc<speaker>_mod<k>.wav
  - the leading digit is the label dataset.py reads,
  - 'sc<speaker>' is the SAME speaker id the original clip carries
    (<digit>_sc<speaker>_<n>.wav from download_speech_commands.py), so the
    speaker-grouped split (src/splits.py) keeps a clip and its modified copy on
    the same side of train / val / test. Without that the model would be tested
    on re-filtered copies of audio it trained on.
  - 'mod<k>' marks the clip as modified; evaluate.py scores original and
    modified test clips separately.

Clips go through the same save_wav() as the originals (peak-normalised 16-bit
PCM at the source rate); the pipeline resamples on load.

RUN
---
    python scripts/add_modified_commands.py                         # ~/Downloads/google_commands_0_9_mod
    python scripts/add_modified_commands.py --src /path/to/folder
    python scripts/add_modified_commands.py --per-class 500         # cap per digit (same clips as
                                                                    # download_speech_commands.py --per-class 500)
Then retrain:
    python src/train.py --model cnn --augment --unknown-class

On Kaggle this is run by notebooks/kaggle_train.ipynb when USE_MOD = True
(see scripts/kaggle_train_with_mod.sh).
"""

import argparse
import re
import sys
import tempfile
import zipfile
from pathlib import Path

WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
MOD_SUFFIX = "_mod"
# Speech Commands filename: <8-hex speaker hash>_nohash_<utterance number>.wav
_SC_NAME_RE = re.compile(r"^([0-9a-f]{8})_nohash_(\d+)$")


def word_of_folder(name: str):
    """'seven_mod' (or 'seven_mod.zip') -> 'seven'; anything else -> None."""
    if name.lower().endswith(".zip"):
        name = name[:-4]
    if name.endswith(MOD_SUFFIX) and name[: -len(MOD_SUFFIX)] in WORD_TO_DIGIT:
        return name[: -len(MOD_SUFFIX)]
    return None


def mod_name(word: str, stem: str):
    """Output filename for one modified clip, or None if the name is not a
    Speech Commands name (then the speaker is unknown and the clip is skipped).

        mod_name('seven', '1b4c9b89_nohash_2') -> '7_sc1b4c9b89_mod2.wav'
    """
    m = _SC_NAME_RE.match(stem)
    if m is None:
        return None
    return f"{WORD_TO_DIGIT[word]}_sc{m.group(1)}_mod{m.group(2)}.wav"


def find_clips(src: Path, unzip_to: Path):
    """{word: sorted [wav paths]} for every <word>_mod folder (or .zip) under src.

    Handles the folder being nested (Kaggle sometimes mounts an uploaded folder as
    <word>_mod/<word>_mod/) and un-extracted <word>_mod.zip archives.
    """
    roots = [src]
    for z in sorted(src.rglob("*.zip")):
        if word_of_folder(z.name) is None:
            continue
        dest = unzip_to / z.stem
        if not dest.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dest)
        roots.append(unzip_to)
    clips: dict = {}
    for root in dict.fromkeys(roots):
        for wav in root.rglob("*.wav"):
            word = next((w for w in map(word_of_folder, reversed(wav.relative_to(root).parts[:-1]))
                         if w is not None), None)
            if word is not None:
                clips.setdefault(word, {}).setdefault(wav.name, wav)   # same clip found twice: keep one
    return {w: [byname[n] for n in sorted(byname)] for w, byname in clips.items()}


def main():
    parser = argparse.ArgumentParser(description="Add the modified Speech Commands digits to data/raw.")
    parser.add_argument("--src", default=str(Path.home() / "Downloads" / "google_commands_0_9_mod"),
                        help="folder that contains zero_mod/ ... nine_mod/")
    parser.add_argument("--per-class", type=int, default=0,
                        help="max clips per digit (0 = all, the default)")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))       # Kaggle: src/*.py sit next to this file
    import soundfile as sf
    from config import RAW_DATA_DIR
    from preprocess import save_wav

    src = Path(args.src).expanduser()
    if not src.exists():
        sys.exit(f"Folder not found: {src}")
    clips = find_clips(src, Path(tempfile.gettempdir()) / "lmml_mod_unzipped")
    missing = [w for w in WORD_TO_DIGIT if w not in clips]
    if missing:
        sys.exit(f"No clips found for: {', '.join(m + MOD_SUFFIX for m in missing)} under {src}")

    out = Path(RAW_DATA_DIR)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Reading modified clips from {src}")
    total = skipped = 0
    for word, digit in WORD_TO_DIGIT.items():
        wavs = clips[word][: args.per_class] if args.per_class > 0 else clips[word]
        n = 0
        for wp in wavs:
            name = mod_name(word, wp.stem)
            if name is None:
                skipped += 1
                continue
            data, sr = sf.read(str(wp), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            save_wav(data, out / name, sr)
            n += 1
        total += n
        print(f"  digit {digit} ({word}{MOD_SUFFIX}): {n} clips")
    print(f"Total modified clips added: {total}"
          + (f" | skipped {skipped} with a non-Speech-Commands name" if skipped else ""))
    print(f"Data now in {out}.")


if __name__ == "__main__":
    main()
