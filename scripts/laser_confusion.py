"""
laser_confusion.py
==================
Confusion matrix of a laser-channel CNN on REAL laser recordings.

It scores the held-out 20% of the real laser clips -- the same clips
scripts/finetune.py kept out of training -- through the same audio path the
dashboard's laser channel uses (band-pass 200-3800 Hz + enhance). The split is
rebuilt with finetune.py's own code and seed, so with the same --speakers the
"held-out" clips here are exactly the ones the fine-tune never trained on.

Nothing is trained and no model file is written or changed. Outputs (PNG, JSON,
CSV) go to --out-dir.

USAGE (on the Jetson, from the clone that holds this script)
------------------------------------------------------------
    python3 scripts/laser_confusion.py

Defaults: data and code from ~/laser_microphone_ml, the 8 laser speaker sets
recorded up to Nov 11, the team model plus any other laser model found, and
results/jetson_nov11/ next to this script for the output.

    python3 scripts/laser_confusion.py --repo PATH --ckpt A.pt B.pt --out-dir DIR
    python3 scripts/laser_confusion.py --split all      # every laser clip (NOT held-out)
"""

import argparse
import hashlib
import json
import random
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DEFAULT_SPEAKERS = "ali,brooke,brooke2,john,michael,laserali,laserjohn,lasermichaeltrim"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_checkpoints(repo: Path) -> list:
    names = ["models/best_model_cnn_laser_team.pt", "models/best_model_cnn_laser.pt",
             "models/backups/best_model_cnn_laser_oct13.pt"]
    return [repo / n for n in names if (repo / n).exists()]


def _split(ft, full, speakers, only_tag, seed):
    """Same 80/20 split as finetune.finetune_one (same code path, same seed)."""
    from splits import dedupe_indices
    keep = dedupe_indices([p for p, _ in full.samples])
    samples = [full.samples[i] for i in keep]
    rng = random.Random(seed)
    new = [s for s in samples if ft._is_speaker_clip(s[0], speakers)
           and (not only_tag or f"_{only_tag}_" in Path(s[0]).name)]
    by_label = {}
    for s in new:
        by_label.setdefault(s[1], []).append(s)
    train, val = [], []
    for items in by_label.values():
        rng.shuffle(items)
        k = max(1, round(len(items) * 0.2)) if len(items) >= 3 else 0
        val += items[:k]
        train += items[k:]
    return new, train, val


def _plot(cm, labels, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, label="clips")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45 if len(max(labels, key=len)) > 2 else 0)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Spoken (true label)")
    ax.set_title(title, fontsize=10)
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Confusion matrix on real laser clips (no training).")
    p.add_argument("--repo", default=str(Path.home() / "laser_microphone_ml"),
                   help="project copy that holds src/, scripts/finetune.py and data/raw")
    p.add_argument("--ckpt", nargs="*", default=None, help="checkpoint(s) to score")
    p.add_argument("--speakers", default=DEFAULT_SPEAKERS)
    p.add_argument("--only-tag", default="laser", help="only files containing _<tag>_ ('' = all)")
    p.add_argument("--bandpass", default="200,3800", help="'off' to disable")
    p.add_argument("--no-enhance", action="store_true")
    p.add_argument("--split", choices=["heldout", "all"], default="heldout")
    p.add_argument("--out-dir", default=str(HERE / "results" / "jetson_nov11"))
    args = p.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    for sub in ("src", "scripts"):
        if not (repo / sub).is_dir():
            print(f"ERROR: {repo / sub} not found. Pass --repo.")
            return 2
        sys.path.insert(0, str(repo / sub))

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import finetune as ft
    from config import BATCH_SIZE, DEVICE, DIGIT_LABELS, FEATURE_FOR_MODEL, RAW_DATA_DIR, SEED, UNKNOWN_LABEL
    from dataset import SpokenDigitDataset
    from model import model_from_checkpoint

    ckpts = [Path(c).expanduser() for c in args.ckpt] if args.ckpt else _default_checkpoints(repo)
    ckpts = [c if c.is_absolute() else (Path.cwd() / c) for c in ckpts]
    if not ckpts:
        print("ERROR: no checkpoint found. Pass --ckpt.")
        return 3
    if args.bandpass.strip().lower() not in ("off", "none", ""):
        ft._LIVE_PATH["bandpass"] = tuple(float(v) for v in args.bandpass.split(","))
    ft._LIVE_PATH["enhance"] = not args.no_enhance
    speakers = {s.strip() for s in args.speakers.split(",") if s.strip()}
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    device_note = f"{socket.gethostname()} ({DEVICE})"
    print(f"Device: {device_note} | data: {RAW_DATA_DIR} | split: {args.split}")

    seen, summary = set(), []
    for ckpt_path in ckpts:
        if not ckpt_path.exists():
            print(f"SKIP: no checkpoint at {ckpt_path}")
            continue
        sha = _sha256(ckpt_path)
        if sha in seen:
            print(f"SKIP: {ckpt_path.name} is byte-identical to a model already scored")
            continue
        seen.add(sha)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        labels = list(ckpt.get("labels", DIGIT_LABELS))
        feature = ckpt.get("feature", FEATURE_FOR_MODEL["cnn"])
        full = SpokenDigitDataset(data_dir=RAW_DATA_DIR, include_unknown=UNKNOWN_LABEL in labels,
                                  cache_in_memory=False, feature=feature)
        if {l: i for i, l in enumerate(labels)} != full.label_to_index:
            print(f"SKIP: {ckpt_path.name} labels do not match the dataset labels")
            continue
        new, train, val = _split(ft, full, speakers, args.only_tag, SEED)
        chosen = val if args.split == "heldout" else new
        print(f"\n{ckpt_path.name}  sha256 {sha[:12]}")
        print(f"  laser clips: {len(new)} total -> train {len(train)}, held-out {len(val)} | scoring {len(chosen)}")
        if not chosen:
            print("  SKIP: no clips to score")
            continue

        model = model_from_checkpoint(ckpt).to(DEVICE).eval()
        ds = ft._subset(full, chosen, augment=False, feature=feature)
        preds, true = [], []
        with torch.no_grad():
            for x, y in DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False):
                preds += model(x.to(DEVICE)).argmax(dim=1).cpu().tolist()
                true += y.tolist()
        n = len(labels)
        cm = np.zeros((n, n), dtype=int)
        for t, pr in zip(true, preds):
            cm[t, pr] += 1
        acc = float(np.trace(cm)) / max(1, cm.sum())
        per_class = {labels[i]: (float(cm[i, i]) / cm[i].sum() if cm[i].sum() else None) for i in range(n)}
        # Drop an all-zero 'unknown' row AND column so the figure shows only what occurred.
        show = [i for i in range(n) if labels[i] != UNKNOWN_LABEL or cm[i].sum() or cm[:, i].sum()]
        cm_show = cm[np.ix_(show, show)]
        show_labels = [labels[i] for i in show]

        stem = f"confusion_real_laser_{ckpt_path.stem}_{args.split}"
        what = "held-out clips (not used in fine-tuning)" if args.split == "heldout" else "ALL clips (includes training clips)"
        title = (f"Real laser recordings, {what}\n{ckpt_path.name}: {acc:.1%} of {int(cm.sum())} clips\n"
                 f"same speakers as training | scored on {socket.gethostname().split('.')[0]}")
        _plot(cm_show, show_labels, title, out_dir / f"{stem}.png")
        record = {
            "checkpoint": ckpt_path.name, "checkpoint_sha256": sha,
            "finetuned_on": ckpt.get("finetuned_on"), "finetune_epoch": ckpt.get("finetune_epoch"),
            "split": args.split, "seed": SEED, "speakers": sorted(speakers), "only_tag": args.only_tag,
            "audio_path": {"bandpass": ft._LIVE_PATH["bandpass"], "enhance": ft._LIVE_PATH["enhance"]},
            "decision": "argmax over model classes, no confidence threshold (same as finetune.py 'your clips')",
            "device": device_note, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_laser_total": len(new), "n_train": len(train), "n_heldout": len(val), "n_scored": int(cm.sum()),
            "accuracy": acc, "per_class_accuracy": per_class, "labels": labels,
            "confusion_matrix_rows_true_cols_pred": cm.tolist(),
            "clips": [{"file": Path(pth).name, "true": labels[t], "pred": labels[pr]}
                      for (pth, _), t, pr in zip(chosen, true, preds)],
        }
        (out_dir / f"{stem}.json").write_text(json.dumps(record, indent=2))
        with open(out_dir / f"{stem}.csv", "w") as fh:
            fh.write("true\\pred," + ",".join(labels) + "\n")
            for i in range(n):
                fh.write(labels[i] + "," + ",".join(str(int(v)) for v in cm[i]) + "\n")
        print(f"  accuracy {acc:.3f} ({int(np.trace(cm))}/{int(cm.sum())})")
        print(f"  saved -> {out_dir / (stem + '.png')}")
        summary.append((ckpt_path.name, acc, int(cm.sum())))

    print("\nSummary (real laser clips, " + args.split + "):")
    for name, acc, total in summary:
        print(f"  {name:40s} {acc:.3f}  n={total}")
    return 0 if summary else 4


if __name__ == "__main__":
    sys.exit(main())
