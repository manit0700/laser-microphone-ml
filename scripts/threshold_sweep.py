"""
threshold_sweep.py
==================
Calibrate the confidence threshold that decides digit vs. "unknown".

WHY
---
predict.py reports "unknown" when the model's top probability is below
CONFIDENCE_THRESHOLD (config.py, currently 0.60). That number was a reasonable
first guess. This script measures, on the held-out test set, exactly what each
threshold buys you, so the choice is data-driven instead of a guess.

For every candidate threshold it reports:
    coverage  : fraction of clips the model is confident enough to ANSWER
                (the rest become "unknown")
    accuracy  : accuracy on the clips it DID answer
So a high threshold answers less often but is right more often when it does.

FSDD note: the public test set is all real digits (no true noise/unknown clips),
so here "unknown" = a rejected real digit. Once we have real mic/laser noise and
silence clips, re-run this and the same table tells you the operating point that
best separates digits from junk. That's the Sprint 3->4 calibration handoff.

RUN
---
    python scripts/threshold_sweep.py --model lstm
    python scripts/threshold_sweep.py --model ensemble
Outputs a table, a CSV (results/threshold_sweep_<model>.csv), and a PNG plot.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import (  # noqa: E402
    BATCH_SIZE, DEVICE, DIGIT_LABELS, FEATURE_FOR_MODEL, MODEL_TYPE,
    NUM_WORKERS, RESULTS_DIR, model_checkpoint,
)
from dataset import SpokenDigitDataset  # noqa: E402
from model import build_model  # noqa: E402
from utils import load_json  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

TEST_INDICES_PATH = model_checkpoint("lstm").parent / "test_indices.json"


def _conf_correct_single(model_type):
    """Return (confidence, correct) arrays over the test split for one model."""
    checkpoint = torch.load(model_checkpoint(model_type), map_location=DEVICE)
    loaded_type = checkpoint.get("model_type", model_type)
    feature = checkpoint.get("feature", FEATURE_FOR_MODEL[loaded_type])

    dataset = SpokenDigitDataset(feature=feature)
    indices = load_json(TEST_INDICES_PATH)["test_indices"]
    loader = DataLoader(Subset(dataset, indices), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS)

    model = build_model(loaded_type).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    confs, correct = [], []
    with torch.no_grad():
        for feats, labels in loader:
            probs = F.softmax(model(feats.to(DEVICE)), dim=1)
            conf, pred = probs.max(dim=1)
            confs.extend(conf.cpu().numpy().tolist())
            correct.extend((pred.cpu() == labels).numpy().tolist())
    return np.array(confs), np.array(correct, dtype=bool)


def _conf_correct_ensemble():
    """Same, for the LSTM+CNN ensemble (averaged probabilities)."""
    import predict as predict_mod
    from preprocess import load_audio

    dataset = SpokenDigitDataset(cache_in_memory=False)
    indices = load_json(TEST_INDICES_PATH)["test_indices"]
    label_index = {label: i for i, label in enumerate(DIGIT_LABELS)}

    confs, correct = [], []
    for i in indices:
        path, true_idx = dataset.samples[i]
        _, probs = predict_mod.infer_ensemble(load_audio(path))
        pred_label = max(probs, key=probs.get)
        confs.append(probs[pred_label])
        correct.append(label_index[pred_label] == true_idx)
    return np.array(confs), np.array(correct, dtype=bool)


def sweep(confs, correct, thresholds):
    """For each threshold, compute coverage and accuracy-on-answered."""
    rows = []
    for t in thresholds:
        answered = confs >= t
        coverage = float(answered.mean())
        acc = float(correct[answered].mean()) if answered.any() else float("nan")
        rows.append((float(t), coverage, acc))
    return rows


def main(model_type):
    print(f"Sweeping confidence thresholds for '{model_type}' on the test split...")
    if model_type == "ensemble":
        confs, correct = _conf_correct_ensemble()
    else:
        confs, correct = _conf_correct_single(model_type)

    thresholds = np.round(np.arange(0.0, 0.96, 0.05), 2)
    rows = sweep(confs, correct, thresholds)

    print(f"\n{'threshold':>10} {'coverage':>10} {'accuracy':>10}")
    for t, cov, acc in rows:
        print(f"{t:>10.2f} {cov:>10.1%} {acc:>10.1%}")

    # Suggest a threshold: highest answered-accuracy while still answering >=90%.
    ok = [(t, cov, acc) for (t, cov, acc) in rows if cov >= 0.90 and acc == acc]
    if ok:
        best = max(ok, key=lambda r: r[2])
        print(f"\nSuggested threshold: {best[0]:.2f} "
              f"(answers {best[1]:.0%} of clips at {best[2]:.1%} accuracy)")
        print("Raise it once real noise/silence clips exist to reject more junk.")

    # Save CSV.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"threshold_sweep_{model_type}.csv"
    with open(csv_path, "w") as f:
        f.write("threshold,coverage,accuracy_on_answered\n")
        for t, cov, acc in rows:
            f.write(f"{t:.2f},{cov:.4f},{acc:.4f}\n")
    print(f"Saved {csv_path}")

    # Save plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ts = [r[0] for r in rows]
        plt.figure(figsize=(7, 4))
        plt.plot(ts, [r[1] for r in rows], "-o", label="coverage (fraction answered)")
        plt.plot(ts, [r[2] for r in rows], "-o", label="accuracy on answered")
        plt.xlabel("confidence threshold")
        plt.ylabel("rate")
        plt.title(f"Confidence threshold sweep - {model_type}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        png_path = RESULTS_DIR / f"threshold_sweep_{model_type}.png"
        plt.savefig(png_path, dpi=120)
        print(f"Saved {png_path}")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calibrate the confidence threshold.")
    parser.add_argument("--model", choices=["lstm", "cnn", "ensemble"],
                        default=MODEL_TYPE)
    args = parser.parse_args()
    main(args.model)
