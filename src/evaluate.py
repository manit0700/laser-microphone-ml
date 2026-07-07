"""
evaluate.py
===========
Loads the trained model and measures how well it does on the held-out test set.

PIPELINE STAGE
--------------
models/best_model.pt  ->  [THIS FILE]  ->  accuracy, confusion matrix, report

WHAT IT DOES
------------
1. Rebuilds the dataset and re-creates the SAME test split used in training
   (using the indices train.py saved).
2. Runs the model on the test set.
3. Reports overall accuracy and per-class accuracy.
4. Saves a confusion-matrix plot to results/plots/.
5. Writes a short evaluation report (JSON + text) to results/reports/.

RUN
---
    python src/evaluate.py     (run train.py first)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend so it works on headless machines
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader, Subset

from config import (
    BATCH_SIZE,
    DEVICE,
    DIGIT_LABELS,
    FEATURE_FOR_MODEL,
    MODEL_TYPE,
    NUM_WORKERS,
    PLOTS_DIR,
    REPORTS_DIR,
    SEED,
    model_checkpoint,
)
from dataset import SpokenDigitDataset
from model import build_model
from train import TEST_INDICES_PATH, split_dataset
from utils import ensure_dirs, load_json, save_json, set_seed


def load_test_subset(dataset):
    """Recreate the test split: prefer saved indices, else re-split with the seed."""
    if Path(TEST_INDICES_PATH).exists():
        indices = load_json(TEST_INDICES_PATH)["test_indices"]
        return Subset(dataset, indices)
    # Fallback (e.g. indices file missing): reproduce the split deterministically.
    _, _, test_ds = split_dataset(dataset)
    return test_ds


def plot_confusion_matrix(cm, labels, out_path):
    """Save a labeled confusion-matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title("Confusion Matrix (test set)")

    # Write the count inside each cell for readability.
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _predictions_single(model_type):
    """Run one trained model over the test split. Returns (loaded_type, preds, true)."""
    checkpoint = torch.load(model_checkpoint(model_type), map_location=DEVICE)
    loaded_type = checkpoint.get("model_type", model_type)
    feature = checkpoint.get("feature", FEATURE_FOR_MODEL[loaded_type])
    print(f"Evaluating model '{loaded_type}' (feature: {feature})")

    dataset = SpokenDigitDataset(feature=feature)
    test_ds = load_test_subset(dataset)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS)

    model = build_model(loaded_type).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_preds, all_true = [], []
    with torch.no_grad():
        for feats, labels in test_loader:
            logits = model(feats.to(DEVICE))
            all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            all_true.extend(labels.numpy().tolist())
    return loaded_type, np.array(all_preds), np.array(all_true)


def _predictions_ensemble():
    """Run the LSTM+CNN ensemble over the test split. Returns ('ensemble', preds, true)."""
    import predict as predict_mod
    from preprocess import load_audio

    # We need the raw files for the test split, so we can feed both models.
    dataset = SpokenDigitDataset(cache_in_memory=False)
    indices = load_json(TEST_INDICES_PATH)["test_indices"]
    label_index = {label: i for i, label in enumerate(DIGIT_LABELS)}

    all_preds, all_true = [], []
    for i in indices:
        path, true_idx = dataset.samples[i]
        _, probs = predict_mod.infer_ensemble(load_audio(path))
        pred_label = max(probs, key=probs.get)       # argmax (ignore threshold here)
        all_preds.append(label_index[pred_label])
        all_true.append(true_idx)
    print(f"Evaluating ENSEMBLE (lstm + cnn) on {len(all_true)} test samples")
    return "ensemble", np.array(all_preds), np.array(all_true)


def main(model_type: str = MODEL_TYPE):
    set_seed(SEED)
    ensure_dirs()

    if model_type == "ensemble":
        loaded_type, all_preds, all_true = _predictions_ensemble()
    else:
        ckpt_path = model_checkpoint(model_type)
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(
                f"No trained model at {ckpt_path}. Run `python src/train.py "
                f"--model {model_type}` first."
            )
        loaded_type, all_preds, all_true = _predictions_single(model_type)

    overall_acc = accuracy_score(all_true, all_preds)
    cm = confusion_matrix(all_true, all_preds,
                          labels=list(range(len(DIGIT_LABELS))))

    # Per-class accuracy = correct / total for each digit (diagonal / row sum).
    per_class_acc = {}
    for i, label in enumerate(DIGIT_LABELS):
        row_total = cm[i].sum()
        per_class_acc[label] = float(cm[i, i] / row_total) if row_total > 0 else 0.0

    report_text = classification_report(
        all_true, all_preds,
        labels=list(range(len(DIGIT_LABELS))),
        target_names=DIGIT_LABELS,
        zero_division=0,
    )

    # ---- Save outputs (filenames tagged by model so LSTM/CNN don't clash) ----
    cm_path = PLOTS_DIR / f"confusion_matrix_{loaded_type}.png"
    plot_confusion_matrix(cm, DIGIT_LABELS, cm_path)

    report = {
        "model_type": loaded_type,
        "overall_test_accuracy": float(overall_acc),
        "per_class_accuracy": per_class_acc,
        "num_test_samples": int(len(all_true)),
        "confusion_matrix": cm.tolist(),
        "labels": DIGIT_LABELS,
    }
    save_json(report, REPORTS_DIR / f"evaluation_report_{loaded_type}.json")
    with open(REPORTS_DIR / f"evaluation_report_{loaded_type}.txt", "w") as f:
        f.write(f"Laser Microphone - Digit Classifier Evaluation\n")
        f.write(f"==============================================\n\n")
        f.write(f"Overall test accuracy: {overall_acc:.4f} "
                f"on {len(all_true)} samples\n\n")
        f.write("Per-class accuracy:\n")
        for label, acc in per_class_acc.items():
            f.write(f"  digit {label}: {acc:.4f}\n")
        f.write("\nFull classification report:\n")
        f.write(report_text)

    # ---- Print summary ----
    print(f"Overall test accuracy: {overall_acc:.4f} on {len(all_true)} samples\n")
    print("Per-class accuracy:")
    for label, acc in per_class_acc.items():
        print(f"  digit {label}: {acc:.4f}")
    print(f"\nConfusion matrix plot -> {cm_path}")
    print(f"Reports               -> {REPORTS_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a trained digit model.")
    parser.add_argument(
        "--model", choices=["lstm", "cnn", "ensemble"], default=MODEL_TYPE,
        help="which model to evaluate ('ensemble' = LSTM + CNN averaged)",
    )
    args = parser.parse_args()
    main(model_type=args.model)
