"""
splits.py
=========
Honest train / validation / test splits.

WHY THIS EXISTS
---------------
The original split (torch.random_split over clips) had two leaks that made
test accuracy look better than it really is:

  1. DUPLICATES. data/raw contained byte-identical copies of the same recording
     (e.g. 7_manit_0.wav and 7_manit_0_dup0.wav ... _dup5.wav). A random split
     can put one copy in train and another in test, so the model is "tested" on
     audio it already trained on.
  2. SPEAKER LEAKAGE. Clips were split one by one, so every speaker appeared in
     train AND test. The model gets credit for recognising voices it has
     already heard, but the real system must work for NEW speakers.

This module fixes both:
  - removes exact duplicates (same file bytes) and '_dupN' copies,
  - splits by SPEAKER: all clips of a speaker go to exactly one of
    train / val / test (a "grouped split"),
  - keeps DERIVED clips (laser replays, simulated laser, synthetic laser --
    re-recordings or transformations of other speakers' audio) in TRAIN only,
    so a test speaker's voice can never leak in through a derived copy.

Speaker ids come from the filenames already used in this project:
    7_jackson_3.wav             FSDD                       -> jackson
    7_sc1b4c9b89_415.wav        Speech Commands digit      -> 1b4c9b89
    down_1b4c9b89_2.wav         Speech Commands 'unknown'  -> 1b4c9b89 (same speaker!)
    audiomnist/01/7_01_3.wav    AudioMNIST                 -> am01
    9_michael_std_0.wav         team recording (DAQ)       -> michael
    7_laserplay_laser_12.wav    replay_record_laser.py     -> derived (train only)
    7_physlaser_5.wav           physics_laser_sim.py       -> derived (train only)
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

DERIVED_SPEAKERS = ("laserplay", "physlaser", "lasersynth")
_DUP_RE = re.compile(r"_dup\d+$")
_SC_RE = re.compile(r"^sc([0-9a-f]{8})$")


def speaker_of(path: Path) -> str:
    """Best-effort speaker id from a clip's path (see module docstring)."""
    path = Path(path)
    parts = _DUP_RE.sub("", path.stem).split("_")
    spk = parts[1] if len(parts) >= 2 else parts[0]
    for d in DERIVED_SPEAKERS:
        if spk.startswith(d):
            return "DERIVED:" + d
    if "audiomnist" in path.parts:
        return "am" + spk
    m = _SC_RE.match(spk)
    return m.group(1) if m else spk


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def dedupe_indices(paths, cache_file: Path | None = None) -> list[int]:
    """Indices of `paths` to keep: drops '_dupN' copies and byte-identical files.

    The first occurrence (in the given order) of each unique recording is kept.
    Hashes are cached in `cache_file` (keyed by path, size and mtime) so repeated
    runs on the same data are fast.
    """
    cache = {}
    if cache_file is not None and Path(cache_file).exists():
        try:
            cache = json.loads(Path(cache_file).read_text())
        except Exception:  # noqa: BLE001 - corrupt cache: just rebuild
            cache = {}
    seen, keep, updated = set(), [], False
    for i, p in enumerate(paths):
        p = Path(p)
        if _DUP_RE.search(p.stem):
            continue
        st = p.stat()
        key = f"{p}|{st.st_size}|{int(st.st_mtime)}"
        h = cache.get(key)
        if h is None:
            h = _file_hash(p)
            cache[key] = h
            updated = True
        if h in seen:
            continue
        seen.add(h)
        keep.append(i)
    if cache_file is not None and updated:
        try:
            Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_file).write_text(json.dumps(cache))
        except OSError:
            pass  # read-only location (e.g. Kaggle input) -- caching is optional
    return keep


def grouped_split(paths, keep, val_frac: float, test_frac: float, seed: int):
    """Split the kept indices into (train, val, test) index lists BY SPEAKER.

    Speakers are shuffled with `seed`, then assigned to test until it holds
    >= test_frac of the (non-derived) clips, then to val until >= val_frac;
    the rest go to train. Derived speakers always go to train.
    """
    by_spk: dict[str, list[int]] = {}
    for i in keep:
        by_spk.setdefault(speaker_of(paths[i]), []).append(i)
    derived = [s for s in by_spk if s.startswith("DERIVED:")]
    real = sorted(s for s in by_spk if not s.startswith("DERIVED:"))
    random.Random(seed).shuffle(real)

    n_real = sum(len(by_spk[s]) for s in real)
    test, val, train = [], [], []
    for s in real:
        if len(test) < test_frac * n_real:
            test += by_spk[s]
        elif len(val) < val_frac * n_real:
            val += by_spk[s]
        else:
            train += by_spk[s]
    for s in derived:
        train += by_spk[s]
    return sorted(train), sorted(val), sorted(test)


def split_summary(paths, train, val, test) -> str:
    def spk(ix):
        return {speaker_of(paths[i]) for i in ix}
    s_tr, s_va, s_te = spk(train), spk(val), spk(test)
    overlap = (s_tr & s_va) | (s_tr & s_te) | (s_va & s_te)
    return (f"grouped split: train {len(train)} clips / {len(s_tr)} speakers | "
            f"val {len(val)} / {len(s_va)} | test {len(test)} / {len(s_te)} | "
            f"speakers shared between splits: {len(overlap)}")


def kfold_split(paths, keep, k_folds: int, fold: int, val_frac: float, seed: int):
    """Speaker-grouped K-fold: fold `fold` (0-based) of `k_folds` is the test set.

    Speakers are spread over K folds so each fold holds ~1/K of the clips (largest
    speakers placed first, each into the currently smallest fold). Every speaker is
    in exactly one fold, so across the K runs every clip is tested exactly once, by
    a model that never heard that speaker. From the remaining speakers, ~val_frac of
    the clips go to validation (for early stopping / calibration); the rest train.
    Derived clips always train.
    """
    if not 0 <= fold < k_folds:
        raise ValueError(f"fold must be in 0..{k_folds - 1}, got {fold}")
    by_spk: dict[str, list[int]] = {}
    for i in keep:
        by_spk.setdefault(speaker_of(paths[i]), []).append(i)
    derived = [s for s in by_spk if s.startswith("DERIVED:")]
    real = sorted(s for s in by_spk if not s.startswith("DERIVED:"))
    rng = random.Random(seed)
    rng.shuffle(real)
    order = sorted(real, key=lambda s: -len(by_spk[s]))     # stable: ties keep shuffled order
    folds = [[] for _ in range(k_folds)]
    sizes = [0] * k_folds
    for s in order:
        j = min(range(k_folds), key=lambda f: sizes[f])
        folds[j].append(s)
        sizes[j] += len(by_spk[s])

    n_real = sum(sizes)
    test = [i for s in folds[fold] for i in by_spk[s]]
    rest = [s for f in range(k_folds) if f != fold for s in folds[f]]
    random.Random(seed + 1000 + fold).shuffle(rest)
    val, train = [], []
    for s in rest:
        if len(val) < val_frac * n_real:
            val += by_spk[s]
        else:
            train += by_spk[s]
    for s in derived:
        train += by_spk[s]
    return sorted(train), sorted(val), sorted(test)
