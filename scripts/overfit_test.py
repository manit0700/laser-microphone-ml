"""
overfit_test.py
===============
Sanity check: CAN this CNN memorise the laser training clips at all?

It trains on the laser training clips ONLY (the same 641 that finetune.py trains
on), with every regulariser switched off:
  - no augmentation (no added noise, gain, shift or speed change)
  - no dropout
  - no normal-mic replay clips
  - no weight decay, higher learning rate

A healthy model/feature setup should reach ~90%+ on clips it is allowed to
memorise. If it does, the normal recipe (augmentation, dropout, replay, low LR)
is what holds training accuracy down. If it cannot, the limit is the model size
or the features/signal, not the recipe.

This is a DIAGNOSTIC. No model file is written or changed. The held-out number
it prints is for information only: a model trained this way is overfit on
purpose and is not a candidate for the dashboard.

USAGE (on the Jetson)
---------------------
    python3 scripts/overfit_test.py
    python3 scripts/overfit_test.py --epochs 200 --lr 0.003
"""

import argparse
import json
import random
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DEFAULT_SPEAKERS = "ali,brooke,brooke2,john,michael,laserali,laserjohn,lasermichaeltrim"


def main() -> int:
    p = argparse.ArgumentParser(description="Overfit test on the laser training clips (saves no model).")
    p.add_argument("--repo", default=str(Path.home() / "laser_microphone_ml"))
    p.add_argument("--ckpt", default=None,
                   help="starting model (default <repo>/models/backups/best_model_cnn_laser_oct13.pt)")
    p.add_argument("--speakers", default=DEFAULT_SPEAKERS)
    p.add_argument("--only-tag", default="laser")
    p.add_argument("--bandpass", default="200,3800", help="'off' to disable")
    p.add_argument("--no-enhance", action="store_true")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--keep-dropout", action="store_true", help="leave dropout on (default: off)")
    p.add_argument("--out-dir", default=str(HERE / "results" / "jetson_overfit"))
    args = p.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    for sub in ("src", "scripts"):
        if not (repo / sub).is_dir():
            print(f"ERROR: {repo / sub} not found. Pass --repo.")
            return 2
        sys.path.insert(0, str(repo / sub))

    import torch
    from torch import nn
    import finetune as ft
    from config import DEVICE, DIGIT_LABELS, FEATURE_FOR_MODEL, RAW_DATA_DIR, SEED, UNKNOWN_LABEL
    from dataset import SpokenDigitDataset
    from model import model_from_checkpoint
    from splits import dedupe_indices
    from utils import set_seed

    ckpt_path = Path(args.ckpt).expanduser() if args.ckpt else repo / "models" / "backups" / "best_model_cnn_laser_oct13.pt"
    if not ckpt_path.exists():
        print(f"ERROR: no checkpoint at {ckpt_path}. Pass --ckpt.")
        return 3
    if args.bandpass.strip().lower() not in ("off", "none", ""):
        ft._LIVE_PATH["bandpass"] = tuple(float(v) for v in args.bandpass.split(","))
    ft._LIVE_PATH["enhance"] = not args.no_enhance
    speakers = {s.strip() for s in args.speakers.split(",") if s.strip()}

    set_seed(SEED)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    labels = list(ckpt.get("labels", DIGIT_LABELS))
    feature = ckpt.get("feature", FEATURE_FOR_MODEL["cnn"])
    full = SpokenDigitDataset(data_dir=RAW_DATA_DIR, include_unknown=UNKNOWN_LABEL in labels,
                              cache_in_memory=False, feature=feature)
    if {l: i for i, l in enumerate(labels)} != full.label_to_index:
        print("ERROR: checkpoint labels do not match the dataset labels")
        return 3

    # Same 80/20 split as finetune.finetune_one (same code path, same seed).
    keep = dedupe_indices([pth for pth, _ in full.samples])
    samples = [full.samples[i] for i in keep]
    rng = random.Random(SEED)
    new = [s for s in samples if ft._is_speaker_clip(s[0], speakers)
           and (not args.only_tag or f"_{args.only_tag}_" in Path(s[0]).name)]
    by_label, train, val = {}, [], []
    for s in new:
        by_label.setdefault(s[1], []).append(s)
    for items in by_label.values():
        rng.shuffle(items)
        k = max(1, round(len(items) * 0.2)) if len(items) >= 3 else 0
        val += items[:k]
        train += items[k:]
    print(f"Host: {socket.gethostname()} ({DEVICE}) | start: {ckpt_path.name}")
    print(f"laser clips: {len(new)} total -> train {len(train)}, held-out {len(val)}")
    if len(train) < 10:
        print("ERROR: too few training clips")
        return 4

    def tensors(items):
        ds = ft._subset(full, items, augment=False, feature=feature)   # clean: no augmentation
        xs, ys = zip(*[ds[i] for i in range(len(ds))])
        return torch.stack(xs).to(DEVICE), torch.tensor(ys).to(DEVICE)

    print("loading clips once (clean, same band-pass + enhance as the dashboard laser channel)...")
    t0 = time.time()
    xtr, ytr = tensors(train)
    xva, yva = tensors(val) if val else (None, None)
    print(f"  done in {time.time() - t0:.0f}s | feature shape {tuple(xtr.shape[1:])}")

    model = model_from_checkpoint(ckpt).to(DEVICE)
    n_drop = 0
    if not args.keep_dropout:
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                m.p = 0.0
                n_drop += 1
    n_params = sum(q.numel() for q in model.parameters())
    print(f"model: {n_params} parameters | dropout layers switched off: {n_drop} | lr {args.lr} | "
          f"no augmentation, no replay, no weight decay")

    def accuracy(x, y):
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(x[i:i + 256]).argmax(1) for i in range(0, len(x), 256)])
        return float((pred == y).float().mean())

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()
    start_tr, start_va = accuracy(xtr, ytr), (accuracy(xva, yva) if val else float("nan"))
    print(f"\nBEFORE: train clips {start_tr:.3f} | held-out {start_va:.3f}")
    log, best_tr, reached = [], start_tr, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr), device=DEVICE)
        for i in range(0, len(xtr), args.batch):
            idx = perm[i:i + args.batch]
            if len(idx) < 2:
                continue                       # BatchNorm needs more than one clip
            opt.zero_grad()
            loss = loss_fn(model(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        tr = accuracy(xtr, ytr)
        va = accuracy(xva, yva) if val else float("nan")
        log.append({"epoch": epoch, "train_acc": tr, "heldout_acc": va, "last_batch_loss": loss.item()})
        best_tr = max(best_tr, tr)
        if reached is None and tr >= 0.90:
            reached = epoch
        if epoch <= 5 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:3d}/{args.epochs} | train clips {tr:.3f} | held-out {va:.3f} | loss {loss.item():.3f}")

    print(f"\nRESULT: best accuracy on the {len(train)} training clips = {best_tr:.3f}")
    if reached:
        print(f"  Reached 90% at epoch {reached}. The CNN CAN memorise these clips, so the normal recipe")
        print("  (augmentation, dropout, replay, low learning rate) is what keeps training accuracy low.")
    else:
        print("  Did NOT reach 90%. Even with every regulariser off the CNN cannot memorise these clips:")
        print("  the limit is model size or the features/signal, not the training recipe.")
    print("  (held-out numbers above are from an on-purpose overfit model; not a candidate for the dashboard)")

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {"host": socket.gethostname(), "device": str(DEVICE), "date": time.strftime("%Y-%m-%d %H:%M:%S"),
           "start_checkpoint": ckpt_path.name, "n_train": len(train), "n_heldout": len(val),
           "epochs": args.epochs, "lr": args.lr, "batch": args.batch, "dropout_off": not args.keep_dropout,
           "parameters": n_params, "before": {"train_acc": start_tr, "heldout_acc": start_va},
           "best_train_acc": best_tr, "reached_90_at_epoch": reached, "log": log}
    out = out_dir / f"overfit_test_{ckpt_path.stem}.json"
    out.write_text(json.dumps(rec, indent=2))
    print(f"log -> {out}   (no model saved)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
