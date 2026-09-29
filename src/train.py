"""
train.py
========
Trains the LSTM digit classifier and saves the best model + training history.

PIPELINE STAGE
--------------
dataset (MFCC, label)  ->  [THIS FILE: train/validate loop]  ->  models/best_model.pt

WHAT IT DOES
------------
1. Loads the dataset (dataset.py) and splits it into train / validation / test.
2. Trains the LSTM (model.py), printing loss and validation accuracy each epoch.
3. Keeps the model with the LOWEST validation loss (early stopping) and saves it.
4. Saves training history (loss/accuracy per epoch) to results/ for plotting.
5. Saves the test-split indices so evaluate.py scores the SAME held-out data.

RUN
---
    python src/train.py

OUTPUT
------
    models/best_model.pt           (model weights + label mapping)
    results/training_history.json  (per-epoch metrics)

LASER NOTE
----------
Nothing here is audio-specific. Once laser data is loadable by dataset.py, this
script trains on it unchanged.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

import argparse

from config import (
    BATCH_SIZE,
    DEVICE,
    DROPOUT,
    EARLY_STOP_MIN_DELTA,
    EARLY_STOP_PATIENCE,
    LR_SCHEDULER,
    SPECAUGMENT,
    SPLIT_METHOD,
    DIGIT_LABELS,
    FEATURE_FOR_MODEL,
    LEARNING_RATE,
    MODEL_TYPE,
    NUM_EPOCHS,
    NUM_WORKERS,
    SEED,
    TEST_SPLIT,
    TRAIN_SPLIT,
    TRAINING_HISTORY_PATH,
    VAL_SPLIT,
    WEIGHT_DECAY,
    model_checkpoint,
)
from dataset import SpokenDigitDataset
from model import build_model
from utils import ensure_dirs, save_json, set_seed

# Where we remember which samples belong to the test split (used by evaluate.py).
# The split depends only on the seed, so LSTM and CNN share the same test set.
TEST_INDICES_PATH = model_checkpoint("lstm").parent / "test_indices.json"


def split_dataset(dataset, method: str | None = None, verbose: bool = False):
    """Split the dataset into train/val/test subsets reproducibly.

    method "grouped" (default, config.SPLIT_METHOD): drop duplicate recordings and
    keep each speaker in exactly one split, so test = unseen speakers (src/splits.py).
    method "random": the old clip-by-clip random split (speakers and duplicate
    clips leak across splits -- only for reproducing earlier numbers).
    """
    method = method or SPLIT_METHOD
    if method == "grouped":
        from config import RESULTS_DIR
        from splits import dedupe_indices, grouped_split, split_summary
        paths = [p for p, _ in dataset.samples]
        keep = dedupe_indices(paths, cache_file=Path(RESULTS_DIR) / "hash_cache.json")
        train_ix, val_ix, test_ix = grouped_split(paths, keep, VAL_SPLIT, TEST_SPLIT, SEED)
        if verbose:
            print(f"Removed {len(paths) - len(keep)} duplicate clips. "
                  + split_summary(paths, train_ix, val_ix, test_ix))
        return Subset(dataset, train_ix), Subset(dataset, val_ix), Subset(dataset, test_ix)

    n = len(dataset)
    n_train = int(TRAIN_SPLIT * n)
    n_val = int(VAL_SPLIT * n)
    n_test = n - n_train - n_val  # remainder avoids rounding gaps

    generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )
    return train_ds, val_ds, test_ds


def run_epoch(model, loader, criterion, optimizer=None):
    """Run one pass over `loader`. If optimizer is given, train; else evaluate.

    Returns (average_loss, accuracy).
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(is_train)

    for mfcc, labels in loader:
        mfcc = mfcc.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(mfcc)
        loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    torch.set_grad_enabled(True)
    return total_loss / total, correct / total


def diagnose_fit(history: dict, best_epoch: int, augment: bool) -> dict:
    """Look at the train/val curves and say whether the model over- or under-fits.

    Rules of thumb used here:
      - OVERFITTING  : after the best epoch, validation loss went back UP by >5%
                       while training loss kept going DOWN (memorising the
                       training clips instead of learning general patterns).
      - STILL IMPROVING / UNDERFITTING : the best epoch was (one of) the last ones,
                       so more epochs or less regularization could help.
      - GOOD FIT     : validation loss bottomed out and stayed flat.
    With --augment the training metrics are measured on randomly distorted clips,
    so training accuracy is naturally a bit LOWER than validation accuracy; a
    train-val accuracy gap only signals overfitting once it is clearly positive.
    """
    tl, vl = history["train_loss"], history["val_loss"]
    ta, va = history["train_acc"], history["val_acc"]
    n = len(vl)
    b = best_epoch - 1
    val_rise = (vl[-1] - vl[b]) / max(vl[b], 1e-9)
    train_drop = (tl[b] - tl[-1]) / max(tl[b], 1e-9)
    gap = ta[b] - va[b]
    after = vl[b + 1:]
    sustained = bool(after) and min(after) > vl[b] * 1.05   # EVERY later epoch clearly worse
    if len(after) >= 3 and sustained and train_drop > 0:
        verdict = "OVERFITTING: val loss rose {:.0%} after epoch {} while train loss kept falling".format(
            val_rise, best_epoch)
    elif len(after) >= 2 and sustained and train_drop > 0:
        verdict = ("POSSIBLE OVERFITTING: val loss rose {:.0%} in the {} epoch(s) after epoch {} "
                   "(use a larger --patience to confirm)").format(val_rise, len(after), best_epoch)
    elif best_epoch >= n - 1:
        verdict = "STILL IMPROVING / possible underfitting: best epoch was at the end - more epochs or lower dropout may help"
    elif gap > 0.05:
        verdict = "MILD OVERFITTING: train accuracy {:.1%} above val at the best epoch".format(gap)
    else:
        verdict = "GOOD FIT: val loss bottomed out at epoch {} and stayed flat".format(best_epoch)
    return {
        "best_epoch": best_epoch, "epochs_run": n,
        "best_val_loss": round(vl[b], 4), "best_val_acc": round(va[b], 4),
        "train_loss_at_best": round(tl[b], 4), "train_acc_at_best": round(ta[b], 4),
        "train_minus_val_acc": round(gap, 4), "val_loss_rise_after_best": round(val_rise, 4),
        "augmented_train": augment, "verdict": verdict,
    }


def plot_curves(history: dict, best_epoch: int, title: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = range(1, len(history["val_loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    a1.plot(ep, history["train_loss"], label="train loss")
    a1.plot(ep, history["val_loss"], label="val loss")
    a2.plot(ep, history["train_acc"], label="train acc")
    a2.plot(ep, history["val_acc"], label="val acc")
    for ax, name in ((a1, "loss"), (a2, "accuracy")):
        ax.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"best (epoch {best_epoch})")
        ax.set_xlabel("epoch"); ax.set_ylabel(name); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(model_type: str = MODEL_TYPE, augment: bool = False, unknown: bool = False,
         dropout: float | None = None, patience: int | None = None,
         epochs: int | None = None, min_delta: float | None = None,
         specaugment: bool | None = None, scheduler: str | None = None,
         noise_min_snr: float | None = None, split: str | None = None) -> dict:
    set_seed(SEED)
    ensure_dirs()

    dropout = DROPOUT if dropout is None else float(dropout)
    patience = EARLY_STOP_PATIENCE if patience is None else int(patience)
    epochs = NUM_EPOCHS if epochs is None else int(epochs)
    min_delta = EARLY_STOP_MIN_DELTA if min_delta is None else float(min_delta)
    specaugment = SPECAUGMENT if specaugment is None else bool(specaugment)
    scheduler = LR_SCHEDULER if scheduler is None else scheduler
    split = SPLIT_METHOD if split is None else split
    if noise_min_snr is not None:
        import augment as _aug
        _aug._SNR_MIN_DB = float(noise_min_snr)
    import augment as _aug
    noise_min_snr = _aug._SNR_MIN_DB

    feature = FEATURE_FOR_MODEL[model_type]
    checkpoint_path = model_checkpoint(model_type)
    print(f"Model: {model_type}  |  feature: {feature}  |  device: {DEVICE}"
          f"  |  augment: {augment}  |  unknown-class: {unknown}")
    print(f"Dropout: {dropout}  |  max epochs: {epochs}  |  early stopping: "
          + (f"patience {patience} on val loss (min delta {min_delta})" if patience > 0 else "off"))
    print(f"Split: {split}  |  LR schedule: {scheduler}  |  SpecAugment: {specaugment}"
          f"  |  augment noise SNR: {noise_min_snr:g}-30 dB")
    print("Loading dataset...")
    # Clean dataset supplies val/test. When augmenting, a second (augmented)
    # instance supplies train. Both are split with the SAME seed, so the index
    # partition is identical -> train stays disjoint from val/test, and only the
    # training clips get randomized. Without --augment, both point to one dataset.
    dataset = SpokenDigitDataset(feature=feature, include_unknown=unknown)
    print(f"Total samples: {len(dataset)} | class counts: {dataset.class_counts()}")

    # Ordered label list (0..9, plus 'unknown' when enabled) -> class count.
    labels = [lbl for lbl, _ in sorted(dataset.label_to_index.items(), key=lambda kv: kv[1])]
    num_classes = len(labels)

    _, val_ds, test_ds = split_dataset(dataset, split, verbose=True)
    if augment:
        train_source = SpokenDigitDataset(feature=feature, augment=True,
                                          include_unknown=unknown, specaugment=specaugment)
        train_ds, _, _ = split_dataset(train_source, split)
    else:
        train_ds, _, _ = split_dataset(dataset, split)
    print(f"Split -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS)

    model = build_model(model_type, num_classes=num_classes, dropout=dropout).to(DEVICE)

    # Use BOTH GPUs when available ("2x GPU"). DataParallel splits each batch
    # across cards. For this small LSTM the win is modest, but it honors the
    # 2x-GPU setup and scales if the model/data grow.
    from config import GPU_COUNT
    if GPU_COUNT > 1:
        print(f"Using {GPU_COUNT} GPUs via DataParallel.")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)

    if scheduler == "plateau":
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5)
    elif scheduler == "cosine":
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    elif scheduler == "none":
        lr_sched = None
    else:
        raise ValueError(f"unknown scheduler {scheduler!r} (plateau, cosine or none)")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0

    print("Starting training...\n")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        lr_now = optimizer.param_groups[0]["lr"]
        history["lr"].append(lr_now)
        if lr_sched is not None:
            lr_sched.step(val_loss) if scheduler == "plateau" else lr_sched.step()

        improved = val_loss < best_val_loss - min_delta
        note = ""
        # Keep the checkpoint with the LOWEST validation loss (the early-stopping target).
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            note = "  <- best (saved)"
            # Unwrap DataParallel so the checkpoint loads into a plain model
            # later (evaluate.py / predict.py build the bare DigitLSTM).
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(
                {
                    "model_state": state,
                    "model_type": model_type,   # so evaluate/predict rebuild the right net
                    "feature": feature,         # and extract the matching feature
                    "labels": labels,           # 0..9 (+ 'unknown' if trained with it)
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "epoch": epoch,
                    "dropout": dropout,
                    "split": split,
                    "scheduler": scheduler,
                    "specaugment": specaugment,
                    "noise_min_snr": noise_min_snr,
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1
            if patience > 0:
                note = f"  (no val-loss improvement {bad_epochs}/{patience})"

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f} | lr {lr_now:.1e}{note}")

        if patience > 0 and bad_epochs >= patience:
            print(f"\nEarly stopping: val loss has not improved for {patience} epochs "
                  f"(best {best_val_loss:.4f} at epoch {best_epoch}).")
            break

    # Persist history and the test indices for evaluate.py.
    save_json(history, TRAINING_HISTORY_PATH)
    save_json({"test_indices": list(test_ds.indices)}, TEST_INDICES_PATH)

    # --- Overfitting check ---------------------------------------------------
    fit = diagnose_fit(history, best_epoch, augment)
    run = {"model": model_type, "dropout": dropout, "patience": patience,
           "max_epochs": epochs, "split": split, "scheduler": scheduler,
           "specaugment": specaugment, "noise_min_snr": noise_min_snr,
           "n_samples": len(dataset), "n_train": len(train_ds), "n_val": len(val_ds),
           "n_test": len(test_ds), **fit}
    from config import PLOTS_DIR, REPORTS_DIR
    tag = f"{model_type}_d{dropout:g}"
    plot_path = Path(PLOTS_DIR) / f"training_curves_{tag}.png"
    plot_curves(history, best_epoch, f"{model_type.upper()}  dropout={dropout:g}  |  {fit['verdict']}",
                plot_path)
    save_json(run, Path(REPORTS_DIR) / f"training_run_{tag}.json")
    import csv
    runs_csv = Path(REPORTS_DIR) / "training_runs.csv"
    new_file = not runs_csv.exists()
    with open(runs_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(run.keys()))
        if new_file:
            w.writeheader()
        w.writerow(run)

    print(f"\nBest epoch {best_epoch}: val loss {fit['best_val_loss']}, val acc {fit['best_val_acc']}"
          f" (train acc {fit['train_acc_at_best']})")
    print(f"Fit check: {fit['verdict']}")
    print(f"Saved best model       -> {checkpoint_path}")
    print(f"Saved training history -> {TRAINING_HISTORY_PATH}")
    print(f"Saved curves plot      -> {plot_path}")
    print(f"Run summary appended   -> {runs_csv}")
    print(f"Saved test indices     -> {TEST_INDICES_PATH}")
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a spoken-digit classifier.")
    parser.add_argument(
        "--model", choices=["lstm", "cnn"], default=MODEL_TYPE,
        help="which network to train: 'lstm' (MFCC) or 'cnn' (mel spectrogram)",
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="apply training-time data augmentation (noise/gain/time-shift) for "
             "robustness to real mic/laser noise",
    )
    parser.add_argument(
        "--unknown-class", action="store_true", dest="unknown",
        help="train a dedicated 'unknown' class from non-digit clips in "
             "data/raw/unknown/ (11 classes) so non-digits are actively rejected",
    )
    parser.add_argument("--dropout", type=float, default=None,
                        help="dropout probability for this run (default config.DROPOUT=0.3; try 0.2-0.5)")
    parser.add_argument("--patience", type=int, default=None,
                        help="early stopping: epochs without val-loss improvement before stopping "
                             "(default config.EARLY_STOP_PATIENCE=6; 0 disables)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="maximum epochs (default config.NUM_EPOCHS=40)")
    parser.add_argument("--specaugment", action="store_true",
                        help="mask random frequency bands / time steps of training features")
    parser.add_argument("--scheduler", choices=["plateau", "cosine", "none"], default=None,
                        help="learning-rate schedule (default config.LR_SCHEDULER='plateau')")
    parser.add_argument("--noise-min-snr", type=float, default=None,
                        help="lowest SNR (dB) for augmentation noise (default 8; try 0 for laser)")
    parser.add_argument("--split", choices=["grouped", "random"], default=None,
                        help="grouped = by speaker, duplicates removed (default); random = old split")
    args = parser.parse_args()
    main(model_type=args.model, augment=args.augment, unknown=args.unknown,
         dropout=args.dropout, patience=args.patience, epochs=args.epochs,
         specaugment=True if args.specaugment else None, scheduler=args.scheduler,
         noise_min_snr=args.noise_min_snr, split=args.split)
