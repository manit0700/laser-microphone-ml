"""
finetune.py
===========
Fine-tune the EXISTING trained models (the Kaggle LSTM + CNN) on a small set of
your own recordings -- team voices through the standard mic, or real laser
clips -- instead of retraining from scratch.

WHY FINE-TUNE
-------------
The shipped models learned from ~45K clips on a Kaggle GPU. Retraining from
scratch on the Jetson means re-downloading most of that data and hours of
training. Fine-tuning starts from those weights and nudges them toward your
voices / your hardware with a low learning rate for a few epochs -- minutes,
not hours.

HOW IT AVOIDS "FORGETTING"
--------------------------
Training only on your clips can make the model forget everyone else and lose
its 'unknown' reject class. So each run mixes in a REPLAY sample of the base
data already in data/raw/ (FSDD, Speech Commands, unknown words), and the
script only saves an epoch if base-data accuracy stays within --max-drop of
where it started. Your new clips are split 80/20 so you also see accuracy on
YOUR clips that the model never trained on.

The previous checkpoint is backed up to models/backups/ before anything is
overwritten, and nothing is saved at all if no epoch beats the starting point.

USAGE (from the project root)
-----------------------------
  1. Record clips (label = first token of the filename):
       python3 scripts/record_daq_wav.py --all --speaker manit --takes 10 --channel std
       python3 scripts/record_daq_wav.py --all --speaker lasermanit --takes 10 --channel laser
  2. (Recommended) have some base data for replay -- FSDD is small:
       python3 scripts/download_fsdd.py
  3. Fine-tune both models on those speakers:
       python3 scripts/finetune.py --speakers manit,brooke,john
       python3 scripts/finetune.py --speakers lasermanit --epochs 15

  Only one model:     --model lstm   (or cnn)
  Dry run (no save):  --no-save

Undo: copy the backup printed at the start back over models/best_model*.pt.
"""

import argparse
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from config import (  # noqa: E402
    BATCH_SIZE, DEVICE, DIGIT_LABELS, FEATURE_FOR_MODEL, MODELS_DIR, NUM_WORKERS,
    RAW_DATA_DIR, SEED, UNKNOWN_LABEL, WEIGHT_DECAY, model_checkpoint,
)
from dataset import SpokenDigitDataset  # noqa: E402
from model import build_model  # noqa: E402
from train import run_epoch  # noqa: E402
from utils import set_seed  # noqa: E402


def _is_speaker_clip(path: Path, speakers) -> bool:
    """FSDD-style name '<digit>_<speaker>_...' -> does <speaker> match?"""
    parts = path.stem.split("_")
    return len(parts) >= 2 and parts[1] in speakers


def _subset(template: SpokenDigitDataset, samples, augment: bool, feature: str):
    """A dataset with the same label mapping as `template` but only `samples`."""
    ds = SpokenDigitDataset.__new__(SpokenDigitDataset)
    ds.data_dir = template.data_dir
    ds.include_unknown = template.include_unknown
    ds.feature = feature
    ds.augment = augment
    ds.specaugment = False
    ds.cache_in_memory = not augment
    ds.label_to_index = template.label_to_index
    ds.samples = list(samples)
    ds._cache = {}
    return ds


def _counts(samples, label_to_index):
    inv = {v: k for k, v in label_to_index.items()}
    out = {k: 0 for k in label_to_index}
    for _, idx in samples:
        out[inv[idx]] += 1
    return out


def finetune_one(model_type: str, args, speakers) -> None:
    ckpt_path = Path(model_checkpoint(model_type))
    print("\n" + "=" * 70)
    print(f"  FINE-TUNE {model_type.upper()}   ({ckpt_path.name})")
    print("=" * 70)
    if not ckpt_path.exists():
        print(f"  SKIP: no checkpoint at {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    labels = ckpt.get("labels", DIGIT_LABELS)
    feature = ckpt.get("feature", FEATURE_FOR_MODEL[model_type])
    include_unknown = UNKNOWN_LABEL in labels

    model = build_model(ckpt.get("model_type", model_type), num_classes=len(labels)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])

    # Scan data/raw once with the SAME label order the checkpoint was trained on.
    full = SpokenDigitDataset(data_dir=args.data_dir, include_unknown=include_unknown,
                              cache_in_memory=False, feature=feature)
    ckpt_order = {lbl: i for i, lbl in enumerate(labels)}
    if ckpt_order != full.label_to_index:
        print(f"  SKIP: checkpoint labels {labels} don't match dataset labels "
              f"{list(full.label_to_index)}")
        return

    # Drop '_dupN' copies and byte-identical duplicates first, so the same recording
    # can never be in both the training part and the held-out part of the check.
    from splits import dedupe_indices
    before = len(full.samples)
    keep = dedupe_indices([p for p, _ in full.samples])
    full.samples = [full.samples[i] for i in keep]
    if before != len(full.samples):
        print(f"  removed {before - len(full.samples)} duplicate clips")

    rng = random.Random(SEED)
    new = [s for s in full.samples if _is_speaker_clip(s[0], speakers)]
    base = [s for s in full.samples if not _is_speaker_clip(s[0], speakers)]
    if len(new) < 10:
        print(f"  SKIP: only {len(new)} clips found for speakers {sorted(speakers)} in "
              f"{args.data_dir}. Record more with scripts/record_daq_wav.py first.")
        return

    # Split YOUR clips 80/20 (stratified by label) so accuracy on them is honest.
    by_label = {}
    for s in new:
        by_label.setdefault(s[1], []).append(s)
    new_train, new_val = [], []
    for items in by_label.values():
        rng.shuffle(items)
        k = max(1, round(len(items) * 0.2)) if len(items) >= 3 else 0
        new_val += items[:k]
        new_train += items[k:]

    # Replay: a class-balanced sample of the base data, split into train/val.
    base_by_label = {}
    for s in base:
        base_by_label.setdefault(s[1], []).append(s)
    replay_train, base_val = [], []
    for items in base_by_label.values():
        rng.shuffle(items)
        base_val += items[:args.base_val_per_class]
        replay_train += items[args.base_val_per_class:args.base_val_per_class + args.replay_per_class]

    print(f"  your clips : {len(new)} total -> train {len(new_train)}, held-out {len(new_val)}")
    print(f"               per label: {_counts(new, full.label_to_index)}")
    print(f"  replay     : {len(replay_train)} base clips mixed into training, "
          f"{len(base_val)} held out to watch for forgetting")
    if not replay_train:
        print("  WARNING: no base data in data/raw/ for replay. The model may forget other")
        print("           speakers and the 'unknown' class. Run scripts/download_fsdd.py first,")
        print("           or keep --epochs low.")

    train_ds = _subset(full, new_train * args.oversample + replay_train, augment=True, feature=feature)
    new_val_ds = _subset(full, new_val, augment=False, feature=feature)
    base_val_ds = _subset(full, base_val, augment=False, feature=feature)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    new_val_loader = DataLoader(new_val_ds, batch_size=BATCH_SIZE, shuffle=False) if new_val else None
    base_val_loader = DataLoader(base_val_ds, batch_size=BATCH_SIZE, shuffle=False) if base_val else None

    criterion = nn.CrossEntropyLoss()

    def evaluate():
        n_acc = run_epoch(model, new_val_loader, criterion)[1] if new_val_loader else float("nan")
        b_acc = run_epoch(model, base_val_loader, criterion)[1] if base_val_loader else float("nan")
        return n_acc, b_acc

    start_new, start_base = evaluate()
    print(f"\n  BEFORE fine-tune: your held-out clips {start_new:.3f} | base data {start_base:.3f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    best = None  # (new_acc, base_acc, epoch, state)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        n_acc, b_acc = evaluate()
        base_ok = (b_acc != b_acc) or (start_base != start_base) or (b_acc >= start_base - args.max_drop)
        score = n_acc if n_acc == n_acc else tr_acc
        flag = ""
        if base_ok and (best is None or score > best[0]):
            best = (score, b_acc, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            flag = "  <- best"
        elif not base_ok:
            flag = "  (base data dropped too much, not kept)"
        print(f"  epoch {epoch:2d}/{args.epochs} | train acc {tr_acc:.3f} | your clips {n_acc:.3f} "
              f"| base {b_acc:.3f} | {time.time() - t0:.0f}s{flag}")

    if best is None:
        print("\n  No epoch kept base-data accuracy within --max-drop. Model NOT changed.")
        return
    start_score = start_new if start_new == start_new else 0.0
    if best[0] <= start_score:
        print(f"\n  Fine-tuning did not beat the starting accuracy on your clips "
              f"({best[0]:.3f} <= {start_score:.3f}). Model NOT changed.")
        return
    print(f"\n  AFTER (epoch {best[2]}): your held-out clips {best[0]:.3f} "
          f"(was {start_new:.3f}) | base data {best[1]:.3f} (was {start_base:.3f})")

    if args.no_save:
        print("  --no-save: leaving the checkpoint unchanged.")
        return

    backup_dir = Path(MODELS_DIR) / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"{ckpt_path.stem}_before_finetune_{time.strftime('%Y%m%d_%H%M%S')}.pt"
    shutil.copy2(ckpt_path, backup)
    ckpt = dict(ckpt)
    ckpt["model_state"] = best[3]
    ckpt["finetuned_on"] = sorted(speakers)
    ckpt["finetune_epoch"] = best[2]
    torch.save(ckpt, ckpt_path)
    print(f"  Saved -> {ckpt_path}")
    print(f"  Backup of the previous model -> {backup}")


def main() -> int:
    p = argparse.ArgumentParser(description="Fine-tune the trained models on your own recordings.")
    p.add_argument("--speakers", required=True,
                   help="comma-separated speaker names from the filenames, e.g. manit,brooke "
                        "(matches '<digit>_<speaker>_*.wav')")
    p.add_argument("--model", choices=["lstm", "cnn", "both"], default="both",
                   help="which model(s) to fine-tune (default both -- the dashboard uses both)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4, help="learning rate (low on purpose)")
    p.add_argument("--replay-per-class", type=int, default=150,
                   help="base clips per label mixed into training to prevent forgetting")
    p.add_argument("--base-val-per-class", type=int, default=40,
                   help="base clips per label held out to measure forgetting")
    p.add_argument("--oversample", type=int, default=3,
                   help="repeat your clips this many times per epoch (augmentation varies each copy)")
    p.add_argument("--max-drop", type=float, default=0.03,
                   help="max allowed drop in base-data accuracy for an epoch to be kept")
    p.add_argument("--data-dir", default=str(RAW_DATA_DIR))
    p.add_argument("--no-save", action="store_true", help="dry run: report accuracy, don't save")
    args = p.parse_args()

    set_seed(SEED)
    speakers = {s.strip() for s in args.speakers.split(",") if s.strip()}
    print(f"Device: {DEVICE} | speakers: {sorted(speakers)} | data: {args.data_dir}")
    for m in (["lstm", "cnn"] if args.model == "both" else [args.model]):
        finetune_one(m, args, speakers)
    print("\nDone. Run the dashboard to try it:  python3 signal_dashboard2.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
