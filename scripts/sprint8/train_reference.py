"""
train_reference.py  (Sprint 8, Agent B)
=======================================
Train the Sprint 8 REFERENCE CNN from scratch on the manifest's `split=train,
kind=clean` clips, validating on `split=val, kind=clean`.

This file also hosts the manifest-driven Dataset and helpers that
finetune_sim.py and evaluate_sprint8.py import, so all three scripts load audio
in exactly the same way:

  * clean clips : preprocess_file(path)                                  (dataset.py path)
  * sim clips   : load_audio -> band-pass 200-3800 Hz (Butterworth 4, zero
                  phase) -> enhance_waveform -> preprocess_waveform
                  = the dashboard's LASER channel (signal_backend._LiveClassifier
                  with LMML_LASER_BANDPASS=200,3800 and ENABLE_ENHANCE=True), the
                  same as scripts/finetune.py's _LivePathDataset.

Nothing here re-splits anything: the split column of the manifest is final.

Recipe (the project's "improved" recipe, docs/EXPERIMENTS.md Exp 4/5):
  CNN on log-mel, dropout 0.4, Adam 1e-3 (+ weight decay 1e-5), ReduceLROnPlateau
  (factor 0.5, patience 2), waveform augmentation with noise down to 0 dB SNR,
  SpecAugment, label smoothing 0.1, batch 32, max 40 epochs, early stopping
  patience 6 on validation loss (min delta 1e-3). Seed 42.

Run (from the project root, env activated, LMML_OUTPUT_DIR set):
  python scripts/sprint8/train_reference.py --manifest results/sprint8/data/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from config import DEVICE, DIGIT_LABELS, SAMPLE_RATE, UNKNOWN_LABEL, WEIGHT_DECAY  # noqa: E402
from features import extract_features  # noqa: E402
from model import build_model  # noqa: E402
from preprocess import load_audio, preprocess_file, preprocess_waveform  # noqa: E402
from train import diagnose_fit, plot_curves, run_epoch  # noqa: E402

LABELS = list(DIGIT_LABELS) + [UNKNOWN_LABEL]          # 11 classes, 'unknown' last
LABEL_TO_INDEX = {lbl: i for i, lbl in enumerate(LABELS)}
UNKNOWN_IDX = LABEL_TO_INDEX[UNKNOWN_LABEL]

LIVE_BANDPASS = (200.0, 3800.0)      # dashboard laser channel default (LMML_LASER_BANDPASS)
LIVE_BANDPASS_ORDER = 4              # signal_backend._LiveClassifier._filter uses butter(4, ...)

DEFAULT_MANIFEST = ROOT / "results/sprint8/data/manifest.csv"
MODELS_DIR = ROOT / "results/sprint8/models"
TRAIN_DIR = ROOT / "results/sprint8/train"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return {"seed": seed, "python_random": seed, "numpy": seed, "torch": seed}


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def read_manifest(path: str | Path) -> list[dict]:
    """Read results/sprint8/data/manifest.csv (schema in results/sprint8/PLAN.md)."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    required = {"file", "label", "speaker", "split", "kind", "dataset", "synthetic_noise", "condition"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"manifest {path} is missing columns {sorted(missing)}")
    for r in rows:
        r["label"] = r["label"].strip()
        if r["label"] not in LABEL_TO_INDEX:
            raise ValueError(f"manifest row has unexpected label {r['label']!r}: {r['file']}")
        r["synthetic_noise"] = int(r.get("synthetic_noise") or 0)
    return rows


def select_rows(rows, split: str | None = None, kind: str | None = None,
                condition: str | None = None, label: str | None = None,
                speaker: str | None = None) -> list[dict]:
    out = rows
    if split is not None:
        out = [r for r in out if r["split"] == split]
    if kind is not None:
        out = [r for r in out if r["kind"] == kind]
    if condition is not None:
        out = [r for r in out if r["condition"] == condition]
    if label is not None:
        out = [r for r in out if r["label"] == label]
    if speaker is not None:
        out = [r for r in out if r["speaker"] == speaker]
    return out


def label_counts(rows) -> dict:
    counts = {lbl: 0 for lbl in LABELS}
    for r in rows:
        counts[r["label"]] += 1
    return counts


# ---------------------------------------------------------------------------
# Audio loading: clean path vs the dashboard's laser path
# ---------------------------------------------------------------------------
_SOS = None


def _bandpass_sos():
    global _SOS
    if _SOS is None:
        from scipy.signal import butter
        low, high = LIVE_BANDPASS
        high = min(float(high), SAMPLE_RATE / 2 * 0.95)
        _SOS = butter(LIVE_BANDPASS_ORDER, [float(low), high], btype="band", fs=SAMPLE_RATE, output="sos")
    return _SOS


def load_live_path(path: Path) -> torch.Tensor:
    """What the dashboard's laser channel feeds the model: band-pass -> enhance -> preprocess."""
    from scipy.signal import sosfiltfilt
    from enhance import enhance_waveform
    w = load_audio(path)
    if w.numel() > 64:
        w = torch.from_numpy(np.ascontiguousarray(sosfiltfilt(_bandpass_sos(), w.numpy()), dtype=np.float32))
    w = enhance_waveform(w)
    return preprocess_waveform(w)


def load_clip(path: Path, live_path: bool) -> torch.Tensor:
    """Fixed-length (MAX_AUDIO_SAMPLES,) waveform, loaded the way the model will hear it."""
    return load_live_path(path) if live_path else preprocess_file(path)


class ManifestDataset(Dataset):
    """(mel feature, label index) pairs for a list of manifest rows.

    live_path_for(row) decides, per clip, whether to use the laser path (default:
    kind == 'sim'). Preprocessed waveforms are cached in RAM (8000 floats each);
    when `augment` is False the final features are cached too, so evaluation and
    validation never recompute anything.
    """

    def __init__(self, rows, feature: str = "mel", augment: bool = False,
                 specaugment: bool = False, live_path_for=None, noise_min_snr: float | None = None):
        self.rows = list(rows)
        self.feature = feature
        self.augment = augment
        self.specaugment = specaugment and augment
        self.live_path_for = live_path_for or (lambda r: r["kind"] == "sim")
        self.noise_min_snr = noise_min_snr
        self.paths = [ROOT / r["file"] for r in self.rows]
        self.labels = [LABEL_TO_INDEX[r["label"]] for r in self.rows]
        self.live = [bool(self.live_path_for(r)) for r in self.rows]
        self._waves: dict[int, torch.Tensor] = {}
        self._feats: dict[int, torch.Tensor] = {}

    def __len__(self):
        return len(self.rows)

    def waveform(self, idx: int) -> torch.Tensor:
        w = self._waves.get(idx)
        if w is None:
            w = load_clip(self.paths[idx], self.live[idx])
            self._waves[idx] = w
        return w

    def preload(self, log_every: int = 2000, label: str = "") -> float:
        """Load every waveform once (main process) so epochs are compute-bound."""
        t0 = time.time()
        for i in range(len(self)):
            self.waveform(i)
            if log_every and (i + 1) % log_every == 0:
                print(f"    [{label}] preloaded {i + 1}/{len(self)} clips ({time.time() - t0:.0f}s)", flush=True)
        return time.time() - t0

    def __getitem__(self, idx: int):
        if not self.augment and idx in self._feats:
            return self._feats[idx], self.labels[idx]
        wave = self.waveform(idx)
        if self.augment:
            import augment as _aug
            if self.noise_min_snr is not None:
                _aug._SNR_MIN_DB = float(self.noise_min_snr)
            wave = _aug.augment_waveform(wave)
        feat = extract_features(wave, self.feature)
        if self.specaugment:
            from augment import spec_augment
            feat = spec_augment(feat, self.feature)
        item = feat.float()
        if not self.augment:
            self._feats[idx] = item
        return item, self.labels[idx]


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, seed: int | None = None) -> DataLoader:
    gen = None
    if shuffle:
        gen = torch.Generator()
        gen.manual_seed(seed if seed is not None else 0)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=gen)


# ---------------------------------------------------------------------------
# Misc helpers shared by the Sprint 8 scripts
# ---------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def env_info() -> dict:
    cpu = ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"python": sys.version.split()[0], "torch": str(torch.__version__),
            "platform": platform.platform(), "cpu": cpu, "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(), "device": str(DEVICE),
            "sample_rate": SAMPLE_RATE, "output_dir_env": os.environ.get("LMML_OUTPUT_DIR")}


def check_loadable(path: Path) -> None:
    """A project checkpoint must load with torch.load's default weights_only=True
    (predict.py / signal_backend use plain torch.load), so no exotic objects allowed."""
    torch.load(path, map_location="cpu")


# ---------------------------------------------------------------------------
# Reference training
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Sprint 8 reference CNN (clean train split only).")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--out-ckpt", default=str(MODELS_DIR / "reference_cnn.pt"))
    ap.add_argument("--out-dir", default=str(TRAIN_DIR / "reference"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--min-delta", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--noise-min-snr", type=float, default=0.0, help="augmentation noise floor (dB SNR)")
    ap.add_argument("--no-specaugment", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="debug: cap clips per split")
    args = ap.parse_args()

    t_start = time.time()
    torch.set_num_threads(args.threads)
    seeds = seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(args.manifest)
    train_rows = select_rows(rows, split=args.train_split, kind="clean")
    val_rows = select_rows(rows, split=args.val_split, kind="clean")
    if args.limit:
        train_rows, val_rows = train_rows[:args.limit], val_rows[:max(8, args.limit // 4)]
    if not train_rows or not val_rows:
        print(f"ERROR: train rows {len(train_rows)}, val rows {len(val_rows)}")
        return 2
    spk_train = {r["speaker"] for r in train_rows}
    spk_val = {r["speaker"] for r in val_rows}
    print(f"Manifest: {args.manifest}  ({len(rows)} rows)")
    print(f"train clean: {len(train_rows)} clips / {len(spk_train)} speakers  {label_counts(train_rows)}")
    print(f"val   clean: {len(val_rows)} clips / {len(spk_val)} speakers  {label_counts(val_rows)}")
    print(f"speakers shared train/val: {len(spk_train & spk_val)}")
    specaugment = not args.no_specaugment
    print(f"Recipe: CNN mel | dropout {args.dropout} | Adam {args.lr} wd {WEIGHT_DECAY} | plateau | "
          f"SpecAugment {specaugment} | noise SNR {args.noise_min_snr:g}-30 dB | LS {args.label_smoothing} | "
          f"batch {args.batch_size} | max {args.epochs} ep | patience {args.patience} | seed {args.seed} | "
          f"device {DEVICE} | threads {torch.get_num_threads()}", flush=True)

    train_ds = ManifestDataset(train_rows, augment=True, specaugment=specaugment,
                               noise_min_snr=args.noise_min_snr)
    val_ds = ManifestDataset(val_rows, augment=False)
    print("Preloading waveforms ...", flush=True)
    t_pre = train_ds.preload(label="train") + val_ds.preload(label="val")
    print(f"  preload done in {t_pre:.0f}s", flush=True)

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True, seed=args.seed)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False)

    model = build_model("cnn", num_classes=len(LABELS), dropout=args.dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [], "epoch_seconds": []}
    best_val_loss, best_epoch, bad = float("inf"), 0, 0
    config = {
        "stage": "reference", "model_type": "cnn", "feature": "mel", "labels": LABELS,
        "manifest": os.path.relpath(args.manifest, ROOT), "train_split": args.train_split,
        "val_split": args.val_split, "train_kind": "clean", "val_kind": "clean",
        "n_train": len(train_rows), "n_val": len(val_rows), "train_label_counts": label_counts(train_rows),
        "val_label_counts": label_counts(val_rows), "dropout": args.dropout, "optimizer": "Adam",
        "lr": args.lr, "weight_decay": WEIGHT_DECAY, "scheduler": "ReduceLROnPlateau(factor=0.5,patience=2,min_lr=1e-5)",
        "specaugment": specaugment, "waveform_augment": True, "noise_min_snr_db": args.noise_min_snr,
        "noise_max_snr_db": 30.0, "label_smoothing": args.label_smoothing, "batch_size": args.batch_size,
        "max_epochs": args.epochs, "early_stop_patience": args.patience, "early_stop_min_delta": args.min_delta,
        "early_stop_metric": "val_loss (clean val clips)", "seeds": seeds, "num_workers": 0,
        "live_path_applied_to": "sim clips only (none in this stage)", "sample_rate": SAMPLE_RATE,
        "environment": env_info(),
    }

    print("Training ...", flush=True)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion)
        lr_now = optimizer.param_groups[0]["lr"]
        sched.step(va_loss)
        dt = time.time() - t0
        for k, v in (("train_loss", tr_loss), ("train_acc", tr_acc), ("val_loss", va_loss),
                     ("val_acc", va_acc), ("lr", lr_now), ("epoch_seconds", dt)):
            history[k].append(v)
        improved = va_loss < best_val_loss - args.min_delta
        note = ""
        if improved:
            best_val_loss, best_epoch, bad = va_loss, epoch, 0
            note = "  <- best (saved)"
            torch.save({
                "model_state": model.state_dict(), "model_type": "cnn", "feature": "mel",
                "labels": LABELS, "val_acc": va_acc, "val_loss": va_loss, "epoch": epoch,
                "dropout": args.dropout, "split": "sprint8-manifest", "scheduler": "plateau",
                "specaugment": specaugment, "noise_min_snr": args.noise_min_snr,
                "label_smoothing": args.label_smoothing, "sample_rate": SAMPLE_RATE,
                "sprint8": config,
            }, out_ckpt)
            check_loadable(out_ckpt)
        else:
            bad += 1
            note = f"  (no improvement {bad}/{args.patience})"
        print(f"Epoch {epoch:3d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} | lr {lr_now:.1e} | {dt:.0f}s{note}", flush=True)
        if dt > 90:
            print(f"  WARNING: epoch took {dt:.0f}s (> 90 s budget)", flush=True)
        save_json(history, out_dir / "training_history.json")
        if bad >= args.patience:
            print(f"Early stopping at epoch {epoch} (best {best_val_loss:.4f} at epoch {best_epoch}).")
            break

    fit = diagnose_fit(history, best_epoch, augment=True)
    plot_curves(history, best_epoch, f"Sprint 8 reference CNN (clean train)  |  {fit['verdict']}",
                out_dir / "training_curves_reference.png")
    summary = {"best_epoch": best_epoch, "epochs_run": len(history["val_loss"]),
               "best_val_loss": history["val_loss"][best_epoch - 1], "best_val_acc": history["val_acc"][best_epoch - 1],
               "fit": fit, "wall_seconds_total": time.time() - t_start, "wall_seconds_preload": t_pre,
               "mean_epoch_seconds": float(np.mean(history["epoch_seconds"])),
               "checkpoint": os.path.relpath(out_ckpt, ROOT), "checkpoint_sha256": sha256_of(out_ckpt),
               "config": config}
    save_json(summary, out_dir / "summary.json")
    print(f"\nBest epoch {best_epoch}: val loss {summary['best_val_loss']:.4f}, val acc {summary['best_val_acc']:.4f}")
    print(f"Fit: {fit['verdict']}")
    print(f"Saved -> {out_ckpt}  (sha256 {summary['checkpoint_sha256'][:12]}...)")
    print(f"Total wall time {summary['wall_seconds_total']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
