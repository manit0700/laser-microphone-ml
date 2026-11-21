"""
common.py  (Sprint 8 follow-up)
Shared helpers for the follow-up audit. Reuses the Sprint 8 loader, dataset and
model code from scripts/sprint8 so every follow-up score uses exactly the same
preprocessing as the frozen evaluation (eval mode, no augmentation, T from the
calibration record).

Test-split rule: nothing in this package reads split == 'test' except
`--allow-test` paths, which are not used by the diagnostic or calibration
scripts. Unknown-rejection tuning uses validation only.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "sprint8"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from train_reference import (ManifestDataset, LABELS, UNKNOWN_IDX,  # noqa: E402
                              make_loader, read_manifest, select_rows)
from evaluate_sprint8 import load_model  # noqa: E402

OUT = ROOT / "results" / "sprint8_followup"
CACHE = OUT / "cache"
MANIFEST = ROOT / "results" / "sprint8" / "data" / "manifest_v2.csv"
MODELS = {
    "reference": ROOT / "results" / "sprint8" / "models" / "reference_cnn.pt",
    "continued_clean": ROOT / "results" / "sprint8" / "models" / "continued_clean_cnn.pt",
    "finetuned_v2": ROOT / "results" / "sprint8" / "models" / "finetuned_v2_cnn.pt",
}
# Temperatures and thresholds exactly as frozen in eval/FROZEN_SETTINGS.json (rounds.v2).
FROZEN_T = {"reference": 0.5811861753463745, "continued_clean": 0.7711358666419983,
            "finetuned_v2": 0.5721844434738159}
FROZEN_THR = {"reference": 0.26, "continued_clean": 0.22, "finetuned_v2": 0.26}


def manifest_rows(split: str, kind: str | None = None):
    rows = read_manifest(MANIFEST)
    return select_rows(rows, split=split, kind=kind)


def raw_logits(model_name: str, split: str, kind: str):
    """Eval-mode logits for one (model, split, kind) subset, cached on disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / f"{model_name}__{split}__{kind}.pt"
    if cache_file.exists():
        d = torch.load(cache_file, weights_only=False)
        return d["logits"], d["labels"], d["rows"]
    rows = manifest_rows(split, kind)
    ckpt, model = load_model(MODELS[model_name])
    ds = ManifestDataset(rows, feature="mel", augment=False)
    logits, labels = [], []
    loader = make_loader(ds, 64, shuffle=False)
    with torch.no_grad():
        for feats, y in loader:
            logits.append(model(feats).float().cpu())
            labels.append(y.cpu())
    logits = torch.cat(logits)
    labels = torch.cat(labels)
    torch.save({"logits": logits, "labels": labels, "rows": rows}, cache_file)
    return logits, labels, rows


def probs_from(logits, temperature: float):
    return F.softmax(logits / temperature, dim=1)


def decide(probs, threshold: float):
    conf, pred = probs.max(dim=1)
    pred = pred.clone()
    pred[conf < threshold] = UNKNOWN_IDX
    return pred, conf


def ece(probs, labels, n_bins: int = 15):
    """Top-label ECE with equal-width confidence bins (15 bins by default)."""
    conf, pred = probs.max(dim=1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += float(m.float().mean()) * abs(float(correct[m].mean()) - float(conf[m].mean()))
    return total


def brier(probs, labels):
    onehot = F.one_hot(labels, num_classes=probs.shape[1]).float()
    return float(((probs - onehot) ** 2).sum(dim=1).mean())


def nll(probs, labels):
    return float(F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels))
