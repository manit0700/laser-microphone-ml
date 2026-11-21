"""
diagnose_overfit.py  (Sprint 8 follow-up, Agent 2 diagnostics)
Train vs validation diagnostics for frozen checkpoints. Eval mode, no dropout,
no augmentation, identical preprocessing, frozen T and threshold. Uses train and
val only. Writes results/sprint8_followup/diagnostics/overfit_<model>.json.

Usage:  python scripts/sprint8_followup/diagnose_overfit.py finetuned_v2 reference
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def subset_stats(name, split, kind):
    logits, labels, rows = C.raw_logits(name, split, kind)
    T, thr = C.FROZEN_T[name], C.FROZEN_THR[name]
    probs = C.probs_from(logits, T)
    raw_pred = probs.argmax(dim=1)
    dec_pred, _ = C.decide(probs, thr)
    is_unk = labels == C.UNKNOWN_IDX
    dig = ~is_unk
    out = {
        "n": int(len(labels)), "n_digit": int(dig.sum()), "n_unknown": int(is_unk.sum()),
        "digit_acc_argmax_no_threshold": float((raw_pred[dig] == labels[dig]).float().mean()),
        "digit_acc_decision": float((dec_pred[dig] == labels[dig]).float().mean()),
        "digit_rejected_decision": float((dec_pred[dig] == C.UNKNOWN_IDX).float().mean()),
        "unknown_rejection_decision": float((dec_pred[is_unk] == C.UNKNOWN_IDX).float().mean()) if is_unk.any() else None,
        "overall_acc_11_decision": float((dec_pred == labels).float().mean()),
        "nll_calibrated": C.nll(probs, labels),
        "brier_calibrated": C.brier(probs, labels),
        "ece_15bin_calibrated": C.ece(probs, labels),
    }
    # per-class recall (decision rule), digits and unknown
    per_class = {}
    for c, lbl in enumerate(C.LABELS):
        m = labels == c
        if m.any():
            per_class[lbl] = {"n": int(m.sum()), "recall": float((dec_pred[m] == c).float().mean())}
    out["per_class_recall"] = per_class
    # equal-weight speaker digit accuracy (speaker groups, digits only)
    by_spk = defaultdict(list)
    for i, r in enumerate(rows):
        if dig[i]:
            by_spk[r["speaker"]].append(bool(dec_pred[i] == labels[i]))
    spk_acc = {s: float(np.mean(v)) for s, v in by_spk.items()}
    out["speaker_groups"] = len(spk_acc)
    out["speaker_equal_digit_acc"] = float(np.mean(list(spk_acc.values())))
    out["speaker_digit_acc_min"] = float(min(spk_acc.values()))
    out["speaker_digit_acc_max"] = float(max(spk_acc.values()))
    out["top_speaker_share_of_digits"] = float(max(len(v) for v in by_spk.values()) / max(1, int(dig.sum())))
    # per-source (sim clips share a source with a clean clip; count by dataset)
    by_ds = defaultdict(list)
    for i, r in enumerate(rows):
        if dig[i]:
            by_ds[r["dataset"]].append(bool(dec_pred[i] == labels[i]))
    out["digit_acc_by_dataset"] = {k: {"n": len(v), "acc": float(np.mean(v))} for k, v in by_ds.items()}
    return out


def main():
    names = sys.argv[1:] or ["finetuned_v2"]
    C.OUT.joinpath("diagnostics").mkdir(parents=True, exist_ok=True)
    for name in names:
        res = {"model": name, "checkpoint": str(C.MODELS[name].relative_to(C.ROOT)),
               "T": C.FROZEN_T[name], "threshold": C.FROZEN_THR[name],
               "protocol": "eval mode, no dropout/augmentation, identical preprocessing, frozen T/threshold; train+val only"}
        for split in ["train", "val"]:
            for kind in ["clean", "sim"]:
                res[f"{split}_{kind}"] = subset_stats(name, split, kind)
                s = res[f"{split}_{kind}"]
                print(name, split, kind, "digit_acc", round(s["digit_acc_decision"], 4),
                      "unk_rej", s["unknown_rejection_decision"] and round(s["unknown_rejection_decision"], 4),
                      "nll", round(s["nll_calibrated"], 4), flush=True)
        path = C.OUT / "diagnostics" / f"overfit_{name}.json"
        path.write_text(json.dumps(res, indent=2))
        print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
