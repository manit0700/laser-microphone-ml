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
3. Keeps the model with the best validation accuracy and saves it to models/.
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


def split_dataset(dataset):
    """Split the dataset into train/val/test subsets reproducibly."""
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


def main(model_type: str = MODEL_TYPE, augment: bool = False):
    set_seed(SEED)
    ensure_dirs()

    feature = FEATURE_FOR_MODEL[model_type]
    checkpoint_path = model_checkpoint(model_type)
    print(f"Model: {model_type}  |  feature: {feature}  |  device: {DEVICE}"
          f"  |  augment: {augment}")
    print("Loading dataset...")
    # Clean dataset supplies val/test. When augmenting, a second (augmented)
    # instance supplies train. Both are split with the SAME seed, so the index
    # partition is identical -> train stays disjoint from val/test, and only the
    # training clips get randomized. Without --augment, both point to one dataset.
    dataset = SpokenDigitDataset(feature=feature)
    print(f"Total samples: {len(dataset)} | class counts: {dataset.class_counts()}")

    _, val_ds, test_ds = split_dataset(dataset)
    if augment:
        train_source = SpokenDigitDataset(feature=feature, augment=True)
        train_ds, _, _ = split_dataset(train_source)
    else:
        train_ds, _, _ = split_dataset(dataset)
    print(f"Split -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS)

    model = build_model(model_type).to(DEVICE)

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

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    print("Starting training...\n")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}")

        # Save the best model so far (by validation accuracy).
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Unwrap DataParallel so the checkpoint loads into a plain model
            # later (evaluate.py / predict.py build the bare DigitLSTM).
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(
                {
                    "model_state": state,
                    "model_type": model_type,   # so evaluate/predict rebuild the right net
                    "feature": feature,         # and extract the matching feature
                    "labels": DIGIT_LABELS,
                    "val_acc": best_val_acc,
                    "epoch": epoch,
                },
                checkpoint_path,
            )

    # Persist history and the test indices for evaluate.py.
    save_json(history, TRAINING_HISTORY_PATH)
    save_json({"test_indices": list(test_ds.indices)}, TEST_INDICES_PATH)

    print(f"\nBest validation accuracy: {best_val_acc:.3f}")
    print(f"Saved best model      -> {checkpoint_path}")
    print(f"Saved training history -> {TRAINING_HISTORY_PATH}")
    print(f"Saved test indices    -> {TEST_INDICES_PATH}")


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
    args = parser.parse_args()
    main(model_type=args.model, augment=args.augment)
