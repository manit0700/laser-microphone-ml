"""
calibration_crossfit.py  (Sprint 8 follow-up, Agent 1/2 calibration)
Independent calibration check using VALIDATION data only, cross-fitted over
speaker groups (2 folds). For each protocol the temperature T is fitted by NLL
on one fold and evaluated on the other fold (never on the data it was fitted on).

Protocols (applied identically to every model, as the audit requested):
  P_clean : T fitted on validation clean clips only
  P_mix   : T fitted on validation clean + simulated clips (the Round 2 protocol)
Metrics reported on the held-out fold: NLL, Brier, 15-bin ECE, digit and
unknown breakdown at the frozen threshold, and the uncalibrated (T=1) baseline.
Nothing here touches the test split.

Usage: python scripts/sprint8_followup/calibration_crossfit.py reference continued_clean finetuned_v2
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def fold_of(speaker: str) -> int:
    """Deterministic speaker-group fold (no randomness, stable across runs)."""
    return int(hashlib.sha256(speaker.encode()).hexdigest(), 16) % 2


def fit_T(logits, labels):
    """1-D search for the NLL-minimising temperature on a log grid (no test data)."""
    best_T, best = 1.0, float("inf")
    for T in torch.logspace(-1, 1, 201).tolist():
        v = float(F.cross_entropy(logits / T, labels))
        if v < best:
            best, best_T = v, T
    return best_T


def gather(name, kinds):
    L, Y, R = [], [], []
    for kind in kinds:
        l, y, r = C.raw_logits(name, "val", kind)
        L.append(l); Y.append(y); R += r
    return torch.cat(L), torch.cat(Y), R


def crossfit(name, kinds):
    logits, labels, rows = gather(name, kinds)
    folds = torch.tensor([fold_of(r["speaker"]) for r in rows])
    # both folds are scored once as held-out
    results = {}
    for held in [0, 1]:
        fit_m = folds != held
        ho_m = folds == held
        T = fit_T(logits[fit_m], labels[fit_m])
        thr = C.FROZEN_THR[name]
        probs = C.probs_from(logits[ho_m], T)
        pred, _ = C.decide(probs, thr)
        y = labels[ho_m]
        unk = y == C.UNKNOWN_IDX
        dig = ~unk
        results[f"holdout_fold_{held}"] = {
            "T_fit_on_other_fold": T,
            "n_heldout": int(ho_m.sum()),
            "nll_T_fit": C.nll(probs, y),
            "brier_T_fit": C.brier(probs, y),
            "ece15_T_fit": C.ece(probs, y),
            "nll_uncalibrated_T1": C.nll(C.probs_from(logits[ho_m], 1.0), y),
            "ece15_uncalibrated_T1": C.ece(C.probs_from(logits[ho_m], 1.0), y),
            "digit_acc_at_frozen_thr": float((pred[dig] == y[dig]).float().mean()),
            "digit_rejected_at_frozen_thr": float((pred[dig] == C.UNKNOWN_IDX).float().mean()),
            "unknown_rejection_at_frozen_thr": float((pred[unk] == C.UNKNOWN_IDX).float().mean()) if unk.any() else None,
        }
    # pooled over both held-out folds (each point scored once, out-of-fold)
    oof_probs = torch.zeros_like(logits)
    for held in [0, 1]:
        fit_m = folds != held; ho_m = folds == held
        T = fit_T(logits[fit_m], labels[fit_m])
        oof_probs[ho_m] = C.probs_from(logits[ho_m], T)
    results["out_of_fold_pooled"] = {
        "nll": C.nll(oof_probs, labels), "brier": C.brier(oof_probs, labels),
        "ece15": C.ece(oof_probs, labels), "n": int(len(labels)),
    }
    return results


def main():
    names = sys.argv[1:] or ["reference", "continued_clean", "finetuned_v2"]
    out = {"protocol": "validation only; 2-fold speaker-group cross-fit; T by NLL on log grid; "
                       "threshold = frozen value; ECE = 15 equal-width bins, top-label",
           "fold_rule": "sha256(speaker) mod 2", "models": {}}
    for name in names:
        out["models"][name] = {
            "P_clean": crossfit(name, ["clean"]),
            "P_mix": crossfit(name, ["clean", "sim"]),
        }
        for proto in ["P_clean", "P_mix"]:
            oof = out["models"][name][proto]["out_of_fold_pooled"]
            print(name, proto, "OOF nll", round(oof["nll"], 4), "brier", round(oof["brier"], 4),
                  "ece15", round(oof["ece15"], 4), flush=True)
    path = C.OUT / "calibration" / "calibration_crossfit_val.json"
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
