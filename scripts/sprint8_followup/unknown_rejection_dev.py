"""
unknown_rejection_dev.py  (Sprint 8 follow-up, Agent 3 unknown rejection)
Development-only (VALIDATION split) comparison of rejection rules against the
existing unknown-class + max-probability baseline.

PRE-DECLARED OPERATING REQUIREMENT (written before any validation number was inspected):
  * valid-digit rejection on validation (clean + sim, digits only) <= 5 %
  * among rules meeting that, choose the one with the highest REAL non-digit-speech
    rejection (unknown clips with synthetic_noise == 0)
  * the synthetic-noise rejection is reported separately and is NOT used for selection
The chosen rule is a validation-selected candidate. It is NOT a test result: the
Sprint 8 test split was already used, and no fresh confirmatory data exists here.

Rules (all use the frozen temperature T of the model being tested):
  maxprob : reject if max_p < tau                       (baseline, frozen rule)
  margin  : reject if (p1 - p2) < tau
  entropy : reject if H(p)/log(10 or 11) > tau  (normalised entropy)

Usage: python scripts/sprint8_followup/unknown_rejection_dev.py finetuned_v2
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def score_rule(probs, rule, tau):
    """True where the clip is rejected. The model's own 'unknown' output (argmax == UNKNOWN_IDX)
    always counts as a rejection, exactly as in the frozen decision rule (evaluate_sprint8 /
    calibration.apply_threshold, where pred = argmax over 11 classes)."""
    own_unknown = probs.argmax(dim=1) == C.UNKNOWN_IDX
    return own_unknown | _rule_reject(probs, rule, tau)


def _rule_reject(probs, rule, tau):
    if rule == "maxprob":
        conf = probs.max(dim=1).values
        return conf < tau
    if rule == "margin":
        top2 = probs.topk(2, dim=1).values
        return (top2[:, 0] - top2[:, 1]) < tau
    if rule == "entropy":
        p = probs.clamp_min(1e-12)
        H = -(p * p.log()).sum(dim=1) / np.log(probs.shape[1])
        return H > tau
    raise ValueError(rule)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "finetuned_v2"
    T = C.FROZEN_T[name]
    logits, labels, rows = [], [], []
    for kind in ["clean", "sim"]:
        l, y, r = C.raw_logits(name, "val", kind)
        logits.append(l); labels.append(y); rows += r
    logits = torch.cat(logits); labels = torch.cat(labels)
    probs = C.probs_from(logits, T)
    unk = labels == C.UNKNOWN_IDX
    dig = ~unk
    is_noise = torch.tensor([r.get("synthetic_noise", "0") in ("1", 1, "1.0", True) for r in rows])
    is_speech_unk = unk & ~is_noise
    is_noise_unk = unk & is_noise
    counts = {"digit_clips": int(dig.sum()), "unknown_real_speech": int(is_speech_unk.sum()),
              "unknown_synthetic_noise": int(is_noise_unk.sum())}
    print("validation composition", counts, flush=True)

    taus = {"maxprob": np.round(np.arange(0.20, 0.96, 0.01), 2),
            "margin": np.round(np.arange(0.00, 0.95, 0.01), 2),
            "entropy": np.round(np.arange(0.05, 1.00, 0.01), 2)}
    sweep = {}
    for rule, grid in taus.items():
        rows_out = []
        for tau in grid:
            rej = score_rule(probs, rule, float(tau))
            digit_rej = float(rej[dig].float().mean())
            speech_rej = float(rej[is_speech_unk].float().mean()) if is_speech_unk.any() else float("nan")
            noise_rej = float(rej[is_noise_unk].float().mean()) if is_noise_unk.any() else float("nan")
            rows_out.append({"tau": float(tau), "digit_rejected": digit_rej,
                             "real_speech_unknown_rejected": speech_rej,
                             "synthetic_noise_rejected": noise_rej})
        sweep[rule] = rows_out

    # Operating-requirement selection (validation only, pre-declared rule)
    LIMIT = 0.05
    choice = {}
    for rule, rows_out in sweep.items():
        ok = [r for r in rows_out if r["digit_rejected"] <= LIMIT]
        if ok:
            best = max(ok, key=lambda r: (r["real_speech_unknown_rejected"], -r["digit_rejected"]))
            choice[rule] = best
        else:
            choice[rule] = None
    # frozen baseline operating point (0.26, maxprob) for reference
    frozen = next(r for r in sweep["maxprob"] if abs(r["tau"] - C.FROZEN_THR[name]) < 1e-9)
    # cross-check: the sweep at the frozen threshold must equal the frozen decision rule
    frozen_pred, _ = C.decide(probs, C.FROZEN_THR[name])
    frozen_dig_rej = float((frozen_pred[dig] == C.UNKNOWN_IDX).float().mean())
    assert abs(frozen_dig_rej - frozen["digit_rejected"]) < 1e-12, (frozen_dig_rej, frozen["digit_rejected"])
    frozen["cross_check_frozen_decide_digit_rejected"] = frozen_dig_rej
    out = {"model": name, "T": T, "composition_val": counts,
           "requirement": "digit_rejected <= 0.05 on validation; maximise real_speech_unknown_rejected; "
                          "synthetic noise reported, not used for selection",
           "frozen_baseline_maxprob_0.26": frozen,
           "selected_on_validation": choice, "sweeps": sweep,
           "test_split_touched": False}
    path = C.OUT / "unknown" / f"unknown_rejection_dev_{name}.json"
    path.write_text(json.dumps(out, indent=2))
    print("frozen baseline (maxprob 0.26):", frozen, flush=True)
    for rule, r in choice.items():
        print(rule, "selected:", r, flush=True)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
