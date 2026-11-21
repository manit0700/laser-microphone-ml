"""
seed_comparison_val.py (Sprint 8 follow-up) - VALIDATION ONLY.
Arms: fine-tune (sim + clean replay, 30 epochs) vs update-matched clean-only control (63 epochs, no sim).
Seeds 42 (existing finetuned_v2 / new control), 43, 44.
Calibration: temperature fitted by speaker-group 2-fold cross-fit on the validation set (same protocol for all
checkpoints, P_mix = clean+sim); frozen threshold 0.26. Out-of-fold predictions are scored.
Equal-weight speaker metrics + speaker-clustered bootstrap (1000 resamples over speaker groups).
Paired comparison: fine-tune minus control, same seed number, same validation speakers.
Selection caveat: both arms chose their checkpoints using validation loss, so validation metrics are
in-sample for selection; this applies equally to both arms but inflates both.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from calibration_crossfit import fold_of, fit_T
import numpy as np, torch
from collections import defaultdict
from train_reference import make_loader, ManifestDataset
from evaluate_sprint8 import load_model

ROOT = C.ROOT; F = C.OUT
CKPTS = {
  ("finetune", 42): C.ROOT/"results/sprint8/models/finetuned_v2_cnn.pt",
  ("finetune", 43): F/"models/ft_s43.pt",
  ("finetune", 44): F/"models/ft_s44.pt",
  ("control", 42): F/"models/control_matched_s42.pt",
  ("control", 43): F/"models/control_matched_s43.pt",
  ("control", 44): F/"models/control_matched_s44.pt",
}

def logits_for(path, rows):
    ckpt, model = load_model(path)
    ds = ManifestDataset(rows, feature="mel", augment=False)
    L = []
    with torch.no_grad():
        for feats, _ in make_loader(ds, 64, shuffle=False):
            L.append(model(feats).float().cpu())
    return torch.cat(L)

def main():
    rows = C.manifest_rows("val", "clean") + C.manifest_rows("val", "sim")
    labels = torch.tensor([C.LABELS.index(r["label"]) for r in rows])
    speakers = np.array([r["speaker"] for r in rows])
    noise = np.array([str(r["synthetic_noise"]) == "1" for r in rows])
    unk = (labels == C.UNKNOWN_IDX).numpy()
    dig = ~unk
    folds = torch.tensor([fold_of(s) for s in speakers])
    out = {"protocol": "val only; OOF speaker 2-fold T (P_mix); frozen thr 0.26; bootstrap over speaker groups 1000",
           "n_val": len(rows), "models": {}}
    per_clip = {}
    for key, path in CKPTS.items():
        cache = F/"cache"/f"valseed__{key[0]}_{key[1]}.pt"
        if cache.exists():
            logits = torch.load(cache, weights_only=False)
        else:
            logits = logits_for(path, rows); torch.save(logits, cache)
        probs = torch.zeros_like(logits)
        for held in [0, 1]:
            fit = folds != held; ho = folds == held
            T = fit_T(logits[fit], labels[fit])
            probs[ho] = C.probs_from(logits[ho], T)
        pred, _ = C.decide(probs, C.FROZEN_THR["finetuned_v2"])
        pred = pred.numpy(); y = labels.numpy()
        correct = (pred == y).astype(float)
        rej_dig = (pred == C.UNKNOWN_IDX)[dig].astype(float)
        per_clip[key] = {"correct": correct, "dig": dig, "unk": unk, "noise": noise, "speakers": speakers, "pred": pred, "y": y}
        # speaker-equal digit accuracy (equal weight over speaker groups)
        def spk_eq(mask_w):
            by = defaultdict(list)
            for i in np.where(dig)[0]:
                by[speakers[i]].append(correct[i])
            return float(np.mean([np.mean(v) for v in by.values()]))
        real_speech = unk & ~noise
        synth = unk & noise
        out["models"][f"{key[0]}_s{key[1]}"] = {
            "digit_acc_pooled": float(correct[dig].mean()),
            "digit_acc_speaker_equal": spk_eq(None),
            "digit_rejected": float(rej_dig.mean()),
            "real_nondigit_rejected": float((pred[real_speech] == C.UNKNOWN_IDX).mean()),
            "synthetic_noise_rejected": float((pred[synth] == C.UNKNOWN_IDX).mean()),
            "nll_oof_P_mix": C.nll(probs, labels),
            "ece15_oof_P_mix": C.ece(probs, labels),
            "n_speakers_digit": len(set(speakers[dig])),
        }
        print(key, {k: round(v, 4) for k, v in out["models"][f"{key[0]}_s{key[1]}"].items() if isinstance(v, float)}, flush=True)
    # speaker-clustered bootstrap over speaker groups for speaker-equal accuracy, and paired diffs
    rng = np.random.default_rng(2026)
    uniq = np.array(sorted(set(speakers[dig])))
    def boot_spk_eq(key):
        d = per_clip[key]; corr = d["correct"]; sp = d["speakers"]; dg = d["dig"]
        per_spk = {s: corr[(sp == s) & dg].mean() for s in uniq}
        vals = np.array([per_spk[s] for s in uniq])
        return vals
    vals = {k: boot_spk_eq(k) for k in CKPTS}
    B = 1000; idx = [rng.integers(0, len(uniq), len(uniq)) for _ in range(B)]
    out["bootstrap_speaker_equal_ci95"] = {}
    for k in CKPTS:
        bs = np.array([vals[k][ii].mean() for ii in idx])
        out["bootstrap_speaker_equal_ci95"][f"{k[0]}_s{k[1]}"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    out["paired_diff_finetune_minus_control_speaker_equal"] = {}
    for seed in [42, 43, 44]:
        f_, c_ = vals[("finetune", seed)], vals[("control", seed)]
        diffs = np.array([f_[ii].mean() - c_[ii].mean() for ii in idx])
        out["paired_diff_finetune_minus_control_speaker_equal"][f"seed{seed}"] = {
            "point": float(f_.mean() - c_.mean()),
            "ci95_speaker_bootstrap": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    # mean over seeds per arm
    for arm in ["finetune", "control"]:
        m = [out["models"][f"{arm}_s{s}"]["digit_acc_speaker_equal"] for s in [42, 43, 44]]
        out.setdefault("arm_means_speaker_equal", {})[arm] = {"mean": float(np.mean(m)), "sd_over_3_seeds": float(np.std(m, ddof=1))}
    (F/"diagnostics"/"seed_comparison_val.json").write_text(json.dumps(out, indent=2, default=float))
    print("wrote", F/"diagnostics"/"seed_comparison_val.json")

if __name__ == "__main__":
    main()
