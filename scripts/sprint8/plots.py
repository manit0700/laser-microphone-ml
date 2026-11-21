"""
plots.py  (Sprint 8, Agent B)
=============================
Plot helpers for the Sprint 8 evaluation: 11x11 confusion matrices, reliability
diagrams (calibration), and 'unknown'-threshold sweeps. All figures are written
with the Agg backend; nothing here needs a display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_confusion_matrix(cm, labels, title: str, out_path: Path, normalize_rows: bool = False) -> None:
    """Heat-map of counts with every cell annotated. Rows = true, cols = predicted."""
    cm = np.asarray(cm)
    shown = cm.astype(float)
    if normalize_rows:
        shown = shown / np.maximum(shown.sum(axis=1, keepdims=True), 1)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    im = ax.imshow(shown, cmap="Blues", vmin=0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="row fraction" if normalize_rows else "count")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=10, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted (after temperature + threshold)", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(title, fontsize=11)
    thresh = shown.max() / 2 if shown.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            txt = f"{cm[i, j]:d}" if not normalize_rows else f"{shown[i, j]:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if shown[i, j] > thresh else "black")
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#dddddd", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def reliability_bins(conf, correct, n_bins: int = 15):
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    centers, accs, confs, counts = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        centers.append((lo + hi) / 2)
        counts.append(int(m.sum()))
        accs.append(float(correct[m].mean()) if m.any() else np.nan)
        confs.append(float(conf[m].mean()) if m.any() else np.nan)
    return np.array(centers), np.array(accs), np.array(confs), np.array(counts), edges


def plot_reliability(conf_before, correct_before, conf_after, correct_after, ece_before: float,
                     ece_after: float, temperature: float, title: str, out_path: Path, n_bins: int = 15) -> None:
    """Two reliability diagrams side by side: raw softmax vs temperature-scaled."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (conf, corr, ece, name) in zip(axes, [
        (conf_before, correct_before, ece_before, "before (T = 1.0)"),
        (conf_after, correct_after, ece_after, f"after (T = {temperature:.3f})"),
    ]):
        centers, accs, confs, counts, edges = reliability_bins(conf, corr, n_bins)
        width = edges[1] - edges[0]
        ax.bar(centers, np.nan_to_num(accs), width=width * 0.95, color="#4c78a8", alpha=0.85, label="accuracy in bin")
        gaps = np.nan_to_num(confs) - np.nan_to_num(accs)
        ax.bar(centers, gaps, bottom=np.nan_to_num(accs), width=width * 0.95, color="#e45756", alpha=0.35,
               label="gap to confidence")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("confidence (max softmax)")
        ax.set_ylabel("accuracy")
        ax.set_title(f"{name}   ECE = {ece:.4f}", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax2 = ax.twinx()
        ax2.plot(centers, counts, color="gray", lw=1, marker=".", ms=3)
        ax2.set_ylabel("clips per bin", color="gray", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_threshold_sweep(sweep_rows, chosen: float, title: str, out_path: Path) -> None:
    """sweep_rows: list of dicts with threshold, balanced_acc, digit_acc, digit_rejected, unknown_rejection."""
    t = [r["threshold"] for r in sweep_rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for key, lab, style in (("balanced_acc", "balanced accuracy (11 classes)", "-"),
                            ("digit_acc", "digits correct", "--"),
                            ("digit_rejected", "digits wrongly rejected", ":"),
                            ("unknown_rejection", "unknown rejected", "-.")):
        ax.plot(t, [r[key] for r in sweep_rows], style, lw=1.6, label=lab)
    ax.axvline(chosen, color="gray", ls="--", lw=1, label=f"chosen threshold {chosen:.2f}")
    ax.axvline(0.60, color="#bbbbbb", ls=":", lw=1, label="old fixed 0.60")
    ax.set_xlabel("'unknown' confidence threshold")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_finetune_curves(history: dict, best_epoch: int, title: str, out_path: Path) -> None:
    """Loss + accuracy curves for the fine-tune stage, with sim/clean val shown separately."""
    ep = range(1, len(history["val_loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4))
    a1.plot(ep, history["train_loss"], label="train loss (sim + clean replay, augmented)")
    a1.plot(ep, history["val_loss"], label="val loss (sim + clean)")
    a1.plot(ep, history["val_sim_loss"], "--", label="val loss (sim)")
    a1.plot(ep, history["val_clean_loss"], ":", label="val loss (clean)")
    a2.plot(ep, history["train_acc"], label="train acc")
    a2.plot(ep, history["val_acc"], label="val acc (sim + clean)")
    a2.plot(ep, history["val_sim_acc"], "--", label="val acc (sim)")
    a2.plot(ep, history["val_clean_acc"], ":", label="val acc (clean)")
    if "clean_acc_floor" in history:
        a2.axhline(history["clean_acc_floor"], color="#e45756", lw=1, ls="--", label="clean floor (start - 3 pts)")
    for ax, name in ((a1, "loss"), (a2, "accuracy")):
        if best_epoch:
            ax.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"selected epoch {best_epoch}")
        ax.set_xlabel("epoch"); ax.set_ylabel(name); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
