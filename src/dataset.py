"""
dataset.py
==========
PyTorch Dataset that turns a folder of WAV files into (MFCC, label) pairs the
training loop can consume.

PIPELINE STAGE
--------------
WAV files on disk  ->  [THIS FILE: preprocess + MFCC per sample]  ->  DataLoader -> model

WHAT IT DOES
------------
- Scans data/raw for .wav files and figures out each file's digit label.
- Supports the THREE common labeling styles automatically:
    1. FSDD filename style : "7_jackson_32.wav"  -> label is the first token "7"
    2. Folder-per-class    : raw/3/anything.wav  -> label is the folder name "3"
    3. Metadata CSV        : a CSV with columns  filename,label
- Optionally includes an "unknown" class (silence/noise/non-digit clips) if you
  provide such files (see UNKNOWN handling below).
- For each file: preprocess (preprocess.py) then extract MFCC (features.py).

INPUT  : a directory of WAV files (default: data/raw) OR a metadata CSV path.
OUTPUT : each item is a tuple (mfcc_tensor, label_index)
            mfcc_tensor : shape (time_steps, n_mfcc)  -- float32
            label_index : int in [0..NUM_CLASSES-1]

LASER NOTE
----------
For laser data, you do NOT need to rewrite this class. Either:
  (a) export laser captures as WAV files named like the FSDD style, OR
  (b) add a new loader for CSV voltage/time using
      preprocess.load_waveform_from_array(), then reuse __getitem__.
A clearly marked hook (`_load_one_waveform`) is the single place to extend.
"""

import csv
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from config import DIGIT_LABELS, RAW_DATA_DIR, UNKNOWN_LABEL
from features import extract_features
from preprocess import preprocess_file


def _build_label_index(include_unknown: bool) -> dict:
    """Map each label string to an integer index used by the model."""
    labels = list(DIGIT_LABELS)
    if include_unknown:
        labels = labels + [UNKNOWN_LABEL]
    return {label: i for i, label in enumerate(labels)}


def _label_from_filename(path: Path) -> Optional[str]:
    """FSDD style: '7_jackson_32.wav' -> '7'. Returns None if no digit prefix."""
    token = path.stem.split("_")[0]
    return token if token in DIGIT_LABELS else None


def _label_from_folder(path: Path) -> Optional[str]:
    """Folder-per-class style: raw/3/foo.wav -> '3'."""
    parent = path.parent.name
    return parent if parent in DIGIT_LABELS else None


class SpokenDigitDataset(Dataset):
    """A Dataset of spoken-digit WAV files producing MFCC features.

    Args:
        data_dir       : folder to scan for .wav files (recursively).
        metadata_csv   : optional CSV with columns 'filename','label'. If given,
                         it overrides directory scanning.
        include_unknown: if True, the label set includes 'unknown'. Files whose
                         label is literally 'unknown' (via CSV) or that live in a
                         folder named 'unknown' are mapped to that class.
        cache_in_memory: if True, compute every MFCC once and keep it in RAM.
                         FSDD is small (~3000 short clips) so this is fast and
                         makes training much quicker. Turn off for huge datasets.
    """

    def __init__(
        self,
        data_dir: str | Path = RAW_DATA_DIR,
        metadata_csv: Optional[str | Path] = None,
        include_unknown: bool = False,
        cache_in_memory: bool = True,
        feature: str = "mfcc",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.include_unknown = include_unknown
        self.cache_in_memory = cache_in_memory
        # Which feature to compute per sample: "mfcc" (LSTM) or "mel" (CNN).
        self.feature = feature
        self.label_to_index = _build_label_index(include_unknown)

        # Each entry is (file_path, label_index).
        self.samples: List[Tuple[Path, int]] = []

        if metadata_csv is not None:
            self._load_from_csv(Path(metadata_csv))
        else:
            self._scan_directory()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No usable WAV files found in '{self.data_dir}'. "
                "Expected FSDD-style names like '7_jackson_0.wav', or a "
                "folder-per-digit layout, or pass a metadata CSV. "
                "See README for how to download the dataset."
            )

        # Optional in-memory cache of computed features.
        self._cache: dict = {}

    # ----- discovery ------------------------------------------------------
    def _scan_directory(self) -> None:
        """Find WAV files and infer labels from filename or folder name."""
        for path in sorted(self.data_dir.rglob("*.wav")):
            label = _label_from_filename(path)
            if label is None:
                label = _label_from_folder(path)
            # Handle an explicit 'unknown' folder when enabled.
            if label is None and self.include_unknown and path.parent.name == UNKNOWN_LABEL:
                label = UNKNOWN_LABEL
            if label is None:
                continue  # skip files we cannot label
            self.samples.append((path, self.label_to_index[label]))

    def _load_from_csv(self, csv_path: Path) -> None:
        """Load (filename,label) rows from a metadata CSV.

        'filename' may be absolute or relative to the CSV's folder or data_dir.
        """
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_name = row["filename"].strip()
                label = row["label"].strip()
                if label not in self.label_to_index:
                    continue  # skip labels we are not modeling
                path = Path(raw_name)
                if not path.is_absolute():
                    candidate = csv_path.parent / raw_name
                    path = candidate if candidate.exists() else self.data_dir / raw_name
                if path.exists():
                    self.samples.append((path, self.label_to_index[label]))

    # ----- the extension hook for laser data ------------------------------
    def _load_one_waveform(self, path: Path):
        """Return a clean fixed-length waveform for one sample.

        THIS is the single place to extend for non-WAV laser inputs. For CSV
        voltage/time data, branch on the file suffix and call
        preprocess.load_waveform_from_array(...) instead of preprocess_file().
        """
        return preprocess_file(path)

    # ----- required Dataset interface -------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.cache_in_memory and idx in self._cache:
            return self._cache[idx]

        path, label_index = self.samples[idx]
        waveform = self._load_one_waveform(path)
        feat = extract_features(waveform, self.feature)  # mfcc -> (time,n_mfcc); mel -> (n_mels,time)
        item = (feat.float(), label_index)

        if self.cache_in_memory:
            self._cache[idx] = item
        return item

    # ----- convenience ----------------------------------------------------
    def class_counts(self) -> dict:
        """Return how many samples exist per label (useful sanity check)."""
        index_to_label = {v: k for k, v in self.label_to_index.items()}
        counts = {label: 0 for label in self.label_to_index}
        for _, label_index in self.samples:
            counts[index_to_label[label_index]] += 1
        return counts


if __name__ == "__main__":
    # Quick check that the dataset can find files and produce features.
    ds = SpokenDigitDataset()
    print(f"Found {len(ds)} samples.")
    print(f"Class counts: {ds.class_counts()}")
    mfcc, label = ds[0]
    print(f"Sample 0 MFCC shape: {tuple(mfcc.shape)}  label index: {label}")
