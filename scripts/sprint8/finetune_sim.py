"""
finetune_sim.py  (Sprint 8, Agent B)
====================================
Fine-tune a COPY of the Sprint 8 reference CNN on the manifest's `split=train,
kind=sim` clips plus a class-balanced replay sample of `split=train, kind=clean`
clips. Validation = `split=val` sim + clean clips (the manifest's split column is
final; nothing is re-split here).

LASER PATH DECISION (documented here, applied identically in evaluate_sprint8.py)
----------------------------------------------------------------------------------
The dashboard's laser channel (signal_backend._LiveClassifier, LMML_LASER_BANDPASS=
200,3800, ENABLE_ENHANCE=True) feeds the model: raw -> Butterworth-4 band-pass
200-3800 Hz (zero phase) -> enhance_waveform (DC removal, spectral gate, AGC) ->
preprocess_waveform -> mel. To train on what the model will hear live, every SIM
clip goes through exactly that chain in BOTH training and evaluation. CLEAN clips
go through the plain dataset.py path (preprocess_file), because the standard mic
channel in the dashboard does not band-pass (ENABLE_FILTER=False) -- enhance IS
applied live to the std channel, but on clean clips enhance is close to a no-op
(enhance.py docstring), so this is the one known, small mismatch for clean audio.
The checkpoint records this under 'sprint8.live_path'.

Recipe: Adam lr 1e-4 (weight decay 1e-5), batch 32, up to 15 epochs, label
smoothing 0.1 (same loss as the reference so val losses are comparable), waveform
augmentation (gain/noise 8-30 dB/shift/speed, as scripts/finetune.py) without
SpecAugment, early stopping patience 4 on combined val loss. The selected epoch
is the lowest combined val loss among epochs whose CLEAN val accuracy stays within
--max-clean-drop (3 points) of the reference's clean val accuracy.

Run (project root, env activated, LMML_OUTPUT_DIR set):
  python scripts/sprint8/finetune_sim.py --manifest results/sprint8/data/manifest.csv
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from config import DEVICE, SAMPLE_RATE, WEIGHT_DECAY  # noqa: E402
from model import model_from_checkpoint  # noqa: E402
from train import run_epoch  # noqa: E402

from train_reference import (  # noqa: E402
    LABELS, MODELS_DIR, TRAIN_DIR, DEFAULT_MANIFEST, ManifestDataset, env_info, label_counts,
    make_loader, read_manifest, save_json, seed_everything, select_rows, sha256_of, check_loadable,
    LIVE_BANDPASS, LIVE_BANDPASS_ORDER,
)
from plots import plot_finetune_curves  # noqa: E402


def balanced_replay(clean_rows, sim_rows, ratio: float, rng: random.Random) -> list[dict]:
    """Per label, sample round(ratio * n_sim_label) clean clips (all of them if fewer)."""
    by_label: dict[str, list] = {}
    for r in clean_rows:
        by_label.setdefault(r["label"], []).append(r)
    sim_counts = label_counts(sim_rows)
    out = []
    for lbl, items in sorted(by_label.items()):
        want = int(round(ratio * sim_counts.get(lbl, 0)))
        items = sorted(items, key=lambda r: r["file"])
        rng.shuffle(items)
        out += items[:min(want, len(items))]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Sprint 8: fine-tune the reference CNN on train-split sim clips.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--base", default=str(MODELS_DIR / "reference_cnn.pt"))
    ap.add_argument("--out-ckpt", default=str(MODELS_DIR / "finetuned_cnn.pt"))
    ap.add_argument("--out-dir", default=str(TRAIN_DIR / "finetune"))
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--min-delta", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--replay-ratio", type=float, default=1.0, help="clean replay clips per sim clip, per label")
    ap.add_argument("--max-clean-drop", type=float, default=0.03,
                    help="an epoch is only selectable if clean val acc >= reference clean val acc - this")
    ap.add_argument("--noise-min-snr", type=float, default=8.0, help="augmentation noise floor for fine-tune")
    ap.add_argument("--specaugment", action="store_true")
    ap.add_argument("--no-live-path", action="store_true", help="feed sim clips WITHOUT band-pass+enhance")
    ap.add_argument("--no-sim", action="store_true",
                    help="CONTROL: train on the same clean replay sample only (no sim clips); validation unchanged")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="debug: cap sim clips")
    args = ap.parse_args()

    t_start = time.time()
    torch.set_num_threads(args.threads)
    seeds = seed_everything(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = Path(args.out_ckpt); out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    live = not args.no_live_path
    live_for = (lambda r: r["kind"] == "sim") if live else (lambda r: False)

    base_path = Path(args.base)
    base = torch.load(base_path, map_location=DEVICE)
    if base.get("labels") != LABELS:
        print(f"ERROR: base checkpoint labels {base.get('labels')} != {LABELS}"); return 2
    model = model_from_checkpoint(base).to(DEVICE)
    print(f"Base: {base_path} (epoch {base.get('epoch')}, val acc {base.get('val_acc', float('nan')):.4f}, "
          f"sha256 {sha256_of(base_path)[:12]}...)")

    rows = read_manifest(args.manifest)
    rng = random.Random(args.seed)
    sim_train = select_rows(rows, split=args.train_split, kind="sim")
    clean_train = select_rows(rows, split=args.train_split, kind="clean")
    sim_val = select_rows(rows, split=args.val_split, kind="sim")
    clean_val = select_rows(rows, split=args.val_split, kind="clean")
    if args.limit:
        sim_train = sim_train[:args.limit]; sim_val = sim_val[:max(8, args.limit // 4)]
        clean_val = clean_val[:max(8, args.limit // 4)]
    if not sim_train or not sim_val:
        print(f"ERROR: sim train {len(sim_train)}, sim val {len(sim_val)}"); return 2
    replay = balanced_replay(clean_train, sim_train, args.replay_ratio, rng)
    if args.no_sim:
        print("CONTROL RUN: sim clips removed from training; clean replay sample kept identical.")
        sim_train = []
    train_rows = sim_train + replay
    spk = lambda rs: {r["speaker"] for r in rs}  # noqa: E731
    print(f"train: {len(sim_train)} sim {label_counts(sim_train)}")
    print(f"       + {len(replay)} clean replay (ratio {args.replay_ratio}) {label_counts(replay)}")
    print(f"val  : {len(sim_val)} sim + {len(clean_val)} clean; speakers shared train/val: "
          f"{len(spk(train_rows) & spk(sim_val + clean_val))}")
    print(f"Laser path on sim clips: {live} (band-pass {LIVE_BANDPASS} order {LIVE_BANDPASS_ORDER} + enhance); "
          f"clean clips: plain path", flush=True)

    train_ds = ManifestDataset(train_rows, augment=True, specaugment=args.specaugment,
                               live_path_for=live_for, noise_min_snr=args.noise_min_snr)
    val_sim_ds = ManifestDataset(sim_val, augment=False, live_path_for=live_for)
    val_clean_ds = ManifestDataset(clean_val, augment=False, live_path_for=live_for)
    print("Preloading waveforms ...", flush=True)
    t_pre = train_ds.preload(label="train") + val_sim_ds.preload(label="val sim") + val_clean_ds.preload(label="val clean")
    print(f"  preload done in {t_pre:.0f}s", flush=True)

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True, seed=args.seed)
    val_sim_loader = make_loader(val_sim_ds, args.batch_size, shuffle=False)
    val_clean_loader = make_loader(val_clean_ds, args.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def validate():
        ls, as_ = run_epoch(model, val_sim_loader, criterion)
        lc, ac = run_epoch(model, val_clean_loader, criterion)
        n_s, n_c = len(val_sim_ds), len(val_clean_ds)
        return {"val_sim_loss": ls, "val_sim_acc": as_, "val_clean_loss": lc, "val_clean_acc": ac,
                "val_loss": (ls * n_s + lc * n_c) / (n_s + n_c), "val_acc": (as_ * n_s + ac * n_c) / (n_s + n_c)}

    start = validate()
    clean_floor = start["val_clean_acc"] - args.max_clean_drop
    print(f"BEFORE fine-tune (reference): val sim loss {start['val_sim_loss']:.4f} acc {start['val_sim_acc']:.4f} | "
          f"val clean loss {start['val_clean_loss']:.4f} acc {start['val_clean_acc']:.4f} | "
          f"clean floor {clean_floor:.4f}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "val_sim_loss", "val_sim_acc",
                               "val_clean_loss", "val_clean_acc", "lr", "epoch_seconds", "selectable")}
    history["before"] = start
    history["clean_acc_floor"] = clean_floor
    best_loss, best_epoch, best_state, best_val, bad = float("inf"), 0, None, None, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        v = validate()
        dt = time.time() - t0
        selectable = v["val_clean_acc"] >= clean_floor
        for k, val in (("train_loss", tr_loss), ("train_acc", tr_acc), ("lr", optimizer.param_groups[0]["lr"]),
                       ("epoch_seconds", dt), ("selectable", selectable), *v.items()):
            history[k].append(val)
        improved = selectable and v["val_loss"] < best_loss - args.min_delta
        note = ""
        if improved:
            best_loss, best_epoch, bad = v["val_loss"], epoch, 0
            best_state = copy.deepcopy({k: t.detach().cpu().clone() for k, t in model.state_dict().items()})
            best_val = dict(v)
            note = "  <- best"
        else:
            bad += 1
            note = "  (clean acc below floor, not selectable)" if not selectable else f"  (no improvement {bad}/{args.patience})"
        print(f"Epoch {epoch:2d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {v['val_loss']:.4f} acc {v['val_acc']:.3f} | sim acc {v['val_sim_acc']:.3f} | "
              f"clean acc {v['val_clean_acc']:.3f} | {dt:.0f}s{note}", flush=True)
        save_json(history, out_dir / "training_history.json")
        if bad >= args.patience:
            print(f"Early stopping at epoch {epoch} (best combined val loss {best_loss:.4f} at epoch {best_epoch}).")
            break

    config = {
        "stage": "finetune", "base_checkpoint": os.path.relpath(base_path, ROOT), "base_sha256": sha256_of(base_path),
        "manifest": os.path.relpath(args.manifest, ROOT), "train_split": args.train_split, "val_split": args.val_split,
        "n_train_sim": len(sim_train), "control_no_sim": bool(args.no_sim), "n_train_clean_replay": len(replay), "replay_ratio": args.replay_ratio,
        "train_sim_label_counts": label_counts(sim_train), "replay_label_counts": label_counts(replay),
        "n_val_sim": len(sim_val), "n_val_clean": len(clean_val), "optimizer": "Adam", "lr": args.lr,
        "weight_decay": WEIGHT_DECAY, "scheduler": "none", "batch_size": args.batch_size,
        "label_smoothing": args.label_smoothing, "waveform_augment": True, "noise_min_snr_db": args.noise_min_snr,
        "noise_max_snr_db": 30.0, "specaugment": args.specaugment, "max_epochs": args.epochs,
        "early_stop_patience": args.patience, "early_stop_min_delta": args.min_delta,
        "early_stop_metric": "combined val loss (val sim + val clean), clip-weighted",
        "selection_rule": f"lowest combined val loss among epochs with clean val acc >= reference clean val acc - {args.max_clean_drop}",
        "clean_acc_floor": clean_floor, "live_path": {
            "applied_to": "sim clips (train and eval)" if live else "NOT applied",
            "bandpass_hz": list(LIVE_BANDPASS) if live else None, "bandpass_order": LIVE_BANDPASS_ORDER if live else None,
            "enhance": live, "clean_clips": "plain preprocess_file path (no band-pass, no enhance)",
            "known_mismatch": "dashboard std channel applies enhance_waveform live; clean training clips do not (near no-op on clean audio)",
        }, "seeds": seeds, "num_workers": 0, "sample_rate": SAMPLE_RATE, "environment": env_info(),
        "before_finetune": start,
    }
    if best_state is None:
        print("\nNEGATIVE RESULT: no epoch improved combined val loss while keeping clean val acc within the floor. "
              "No fine-tuned checkpoint saved.")
        save_json({"best_epoch": 0, "saved": False, "config": config, "history": history,
                   "wall_seconds_total": time.time() - t_start}, out_dir / "summary.json")
        return 3

    ckpt = dict(base)
    ckpt["model_state"] = best_state
    ckpt["epoch"] = best_epoch
    ckpt["val_acc"] = best_val["val_acc"]
    ckpt["val_loss"] = best_val["val_loss"]
    ckpt["finetuned_on"] = (f"sprint8 CONTROL: {args.train_split}-split clean replay only (no sim)" if args.no_sim
                            else f"sprint8 {args.train_split}-split sim clips + clean replay")
    ckpt["finetune_epoch"] = best_epoch
    ckpt["live_path"] = {"bandpass": list(LIVE_BANDPASS) if live else None, "enhance": live}
    for k in ("temperature", "threshold", "val_ece_before", "val_ece_after", "val_threshold_metrics", "ensemble_threshold"):
        ckpt.pop(k, None)             # re-fitted on validation by evaluate_sprint8.py calibrate
    ckpt["sprint8"] = {**base.get("sprint8", {}), "finetune": config}
    torch.save(ckpt, out_ckpt)
    check_loadable(out_ckpt)

    plot_finetune_curves(history, best_epoch, "Sprint 8 fine-tune on sim (train split) + clean replay",
                         out_dir / "training_curves_finetune.png")
    summary = {"best_epoch": best_epoch, "epochs_run": len(history["val_loss"]), "saved": True,
               "best": best_val, "before": start,
               "delta_val_sim_acc": best_val["val_sim_acc"] - start["val_sim_acc"],
               "delta_val_clean_acc": best_val["val_clean_acc"] - start["val_clean_acc"],
               "wall_seconds_total": time.time() - t_start, "wall_seconds_preload": t_pre,
               "mean_epoch_seconds": float(np.mean(history["epoch_seconds"])),
               "checkpoint": os.path.relpath(out_ckpt, ROOT), "checkpoint_sha256": sha256_of(out_ckpt), "config": config}
    save_json(summary, out_dir / "summary.json")
    print(f"\nSelected epoch {best_epoch}: val sim acc {start['val_sim_acc']:.4f} -> {best_val['val_sim_acc']:.4f}, "
          f"val clean acc {start['val_clean_acc']:.4f} -> {best_val['val_clean_acc']:.4f}")
    print(f"Saved -> {out_ckpt}  (sha256 {summary['checkpoint_sha256'][:12]}...)  total {summary['wall_seconds_total']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
