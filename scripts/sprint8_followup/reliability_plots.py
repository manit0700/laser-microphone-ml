"""Reliability diagrams from out-of-fold validation predictions (speaker-group 2-fold cross-fit).
Validation only. 15 equal-width bins, top-label confidence. Writes figures/ and a JSON of bin counts."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from calibration_crossfit import fold_of, fit_T, gather

def oof(name, kinds):
    logits, labels, rows = gather(name, kinds)
    folds = torch.tensor([fold_of(r["speaker"]) for r in rows])
    probs = torch.zeros_like(logits)
    for held in [0, 1]:
        fit = folds != held; ho = folds == held
        T = fit_T(logits[fit], labels[fit])
        probs[ho] = C.probs_from(logits[ho], T)
    return probs, labels

def bins(probs, labels, n=15):
    conf, pred = probs.max(dim=1); corr = (pred == labels).float()
    edges = torch.linspace(0, 1, n + 1); out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                    "mean_conf": float(conf[m].mean()) if m.any() else None,
                    "acc": float(corr[m].mean()) if m.any() else None})
    return out

def main():
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    report = {}
    for i, proto in enumerate([["clean"], ["clean", "sim"]]):
        for j, name in enumerate(["reference", "continued_clean", "finetuned_v2"]):
            p, y = oof(name, proto)
            b = bins(p, y); ax = axes[i, j]
            xs = [(bb["lo"] + bb["hi"]) / 2 for bb in b if bb["n"] > 0]
            ys = [bb["acc"] for bb in b if bb["n"] > 0]
            ws = [bb["n"] for bb in b if bb["n"] > 0]
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.bar(xs, ys, width=1 / 15 * 0.9, alpha=0.6, edgecolor="k")
            ax.set_title(f"{name}\n{'clean' if len(proto)==1 else 'clean+sim'} validation, out-of-fold", fontsize=9)
            ax.set_xlabel("confidence (bin centre)")
            if j == 0: ax.set_ylabel("accuracy in bin")
            ax.text(0.02, 0.92, f"n per bin shown in JSON\nN={len(y)}", transform=ax.transAxes, fontsize=7, va="top")
            report[f"{name}__{'clean' if len(proto)==1 else 'clean_sim'}"] = {"bins": b, "N": int(len(y))}
    fig.suptitle("Reliability (out-of-fold, validation only, 15 bins). Dashed = perfect calibration.", fontsize=10)
    fig.tight_layout()
    out = C.OUT / "figures"; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "reliability_oof_val.png", dpi=150)
    (C.OUT / "calibration" / "reliability_bins_oof_val.json").write_text(json.dumps(report, indent=2))
    print("wrote", out / "reliability_oof_val.png")

if __name__ == "__main__":
    main()
