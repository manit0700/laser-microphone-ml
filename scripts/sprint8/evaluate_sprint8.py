"""
evaluate_sprint8.py  (Sprint 8, Agent B)
========================================
Calibration + 'unknown' threshold on the VALIDATION split, then a single test run.

  calibrate   temperature (src/calibration.py) and threshold sweep (balanced
              accuracy over 11 classes) on split=val for reference_cnn.pt and
              finetuned_cnn.pt; writes temperature/threshold into the checkpoints,
              results/sprint8/eval/calibration_val.json, reliability diagrams,
              threshold sweep CSV/PNG, FROZEN_SETTINGS.json (incl. acceptance
              criteria), models/checksums.txt and models/MODELS_READY.
  test        reads FROZEN_SETTINGS.json and scores split=test ONCE for the
              reference, fine-tuned and deployed (models/best_model_cnn.pt,
              read-only, information only) checkpoints on the conditions clean,
              sim_good, sim_medium, sim_poor, sim_all, plus speaker 'manit' clean.
  environment writes results/sprint8/train/ENVIRONMENT.md.

Decision rule (identical to predict.py / calibration.apply_threshold):
  probs = softmax(logits / T); pred = argmax; if max prob < threshold -> 'unknown'.
Sim clips are loaded through the dashboard's laser path (band-pass 200-3800 Hz +
enhance), clean clips through the plain path -- see finetune_sim.py's docstring.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from calibration import (apply_threshold, collect_logits, decision_metrics,  # noqa: E402
                         expected_calibration_error, fit_temperature)
from config import DEVICE  # noqa: E402
from model import model_from_checkpoint  # noqa: E402

from train_reference import (  # noqa: E402
    LABELS, UNKNOWN_IDX, MODELS_DIR, TRAIN_DIR, DEFAULT_MANIFEST, ManifestDataset, env_info,
    label_counts, make_loader, read_manifest, save_json, seed_everything, select_rows, sha256_of, check_loadable,
    LIVE_BANDPASS, LIVE_BANDPASS_ORDER,
)
from plots import plot_confusion_matrix, plot_reliability, plot_threshold_sweep  # noqa: E402

EVAL_DIR = ROOT / "results/sprint8/eval"
PLOTS_DIR = EVAL_DIR / "plots"
DEPLOYED = ROOT / "models/best_model_cnn.pt"
DEPLOYED_NOTE = ("deployed checkpoint - training provenance not established as speaker-disjoint from this "
                 "test set (trained on a Kaggle fold of a superset of these sources), information only")
CONDITIONS = ["clean", "sim_good", "sim_medium", "sim_poor", "sim_all"]
TAG = ""          # "" for round 1; "_v2" etc. for later rounds (suffixes calibration/test outputs, not FROZEN_SETTINGS.json)


# ---------------------------------------------------------------------------
# Shared scoring helpers
# ---------------------------------------------------------------------------
def load_model(path: Path):
    ckpt = torch.load(path, map_location=DEVICE)
    model = model_from_checkpoint(ckpt).to(DEVICE)
    model.eval()
    return ckpt, model


def logits_for(model, ds: ManifestDataset, batch_size: int = 64):
    if len(ds) == 0:
        return torch.zeros(0, len(LABELS)), torch.zeros(0, dtype=torch.long)
    return collect_logits(model, make_loader(ds, batch_size, shuffle=False), DEVICE)


def sweep_threshold(probs, labels, lo=0.20, hi=0.95, step=0.01):
    """Same grid/selection as calibration.tune_threshold, but returns the whole curve."""
    grid = [round(lo + i * step, 4) for i in range(int(round((hi - lo) / step)) + 1)]
    rows = []
    for t in grid:
        m = decision_metrics(apply_threshold(probs, t, UNKNOWN_IDX), labels, UNKNOWN_IDX)
        rows.append({"threshold": t, **m})
    best = max(r["balanced_acc"] for r in rows)
    near = [r["threshold"] for r in rows if r["balanced_acc"] >= best - 0.002]
    chosen = near[len(near) // 2]
    return chosen, rows


def full_metrics(pred: np.ndarray, true: np.ndarray, rows: list[dict]) -> dict:
    """Everything the test table needs for one model x condition."""
    pred, true = np.asarray(pred), np.asarray(true)
    n = len(true)
    is_unk = true == UNKNOWN_IDX
    synth = np.array([r["synthetic_noise"] == 1 for r in rows], dtype=bool) if rows else np.zeros(0, bool)
    unk_synth, unk_speech = is_unk & synth, is_unk & ~synth
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    per_class = {}
    recalls = []
    for i, lbl in enumerate(LABELS):
        tot = int(cm[i].sum())
        acc = float(cm[i, i] / tot) if tot else None
        per_class[lbl] = {"n": tot, "acc": acc}
        if tot:
            recalls.append(acc)

    def rate(mask, cond):
        return float(cond[mask].mean()) if mask.any() else None

    return {
        "n_clips": int(n), "n_digit": int((~is_unk).sum()), "n_unknown": int(is_unk.sum()),
        "n_unknown_synthetic_noise": int(unk_synth.sum()), "n_unknown_real_speech": int(unk_speech.sum()),
        "digit_acc": rate(~is_unk, pred == true),
        "digit_rejection_rate": rate(~is_unk, pred == UNKNOWN_IDX),
        "digit_wrong_digit_rate": rate(~is_unk, (pred != true) & (pred != UNKNOWN_IDX)),
        "unknown_rejection_rate": rate(is_unk, pred == UNKNOWN_IDX),
        "unknown_rejection_rate_synthetic_noise": rate(unk_synth, pred == UNKNOWN_IDX),
        "unknown_rejection_rate_real_speech": rate(unk_speech, pred == UNKNOWN_IDX),
        "overall_acc_11": float((pred == true).mean()) if n else None,
        "balanced_acc": float(np.mean(recalls)) if recalls else None,
        "n_classes_present": len(recalls),
        "per_class_acc": per_class, "confusion_matrix": cm.tolist(), "labels": LABELS,
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------
def cmd_calibrate(args) -> int:
    t_start = time.time()
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    rows = read_manifest(args.manifest)
    val_clean = select_rows(rows, split=args.val_split, kind="clean")
    val_sim = select_rows(rows, split=args.val_split, kind="sim")
    if args.limit:
        val_clean, val_sim = val_clean[:args.limit], val_sim[:args.limit]
    print(f"val clean {len(val_clean)} {label_counts(val_clean)}")
    print(f"val sim   {len(val_sim)} {label_counts(val_sim)}", flush=True)
    ds_clean = ManifestDataset(val_clean, augment=False)
    ds_sim = ManifestDataset(val_sim, augment=False)          # live path for kind == sim
    print("Preloading val waveforms ...", flush=True)
    ds_clean.preload(label="val clean"); ds_sim.preload(label="val sim")

    extra = dict(e.split("=", 1) for e in (args.extra or []))
    models = {k: Path(v) for k, v in extra.items()} if extra else {"reference": Path(args.reference), "finetuned": Path(args.finetuned)}
    prev_report = json.loads((EVAL_DIR / f"calibration_val{TAG}.json").read_text()) if extra and (EVAL_DIR / f"calibration_val{TAG}.json").exists() else None
    prev_frozen = json.loads((EVAL_DIR / "FROZEN_SETTINGS.json").read_text()) if extra and (EVAL_DIR / "FROZEN_SETTINGS.json").exists() else None
    report = {"val_split": args.val_split, "n_val_clean": len(val_clean), "n_val_sim": len(val_sim),
              "manifest": os.path.relpath(args.manifest, ROOT), "manifest_sha256": sha256_of(Path(args.manifest)),
              "threshold_grid": "0.20..0.95 step 0.01; middle of the thresholds within 0.002 of the best balanced accuracy",
              "temperature_fit": "LBFGS on NLL of validation logits (calibration.fit_temperature); kept only if ECE does not get worse",
              "models": {}}
    frozen_models = dict(prev_frozen["models"]) if prev_frozen else {}
    if TAG:
        # New round: start from the carried-over models only (their settings stay exactly as frozen in round 1).
        frozen_models = {n: {**prev_frozen["models"][n], "settings_from": "round 1 freeze (not re-fitted)"} for n in (args.carry or [])}
        prev_report = None
    if prev_report:
        report["models"] = dict(prev_report["models"])
    for name, path in models.items():
        if not path.exists():
            print(f"  {name}: {path} missing; skipped")
            report["models"][name] = {"missing": True}
            continue
        ckpt, model = load_model(path)
        lg_clean, y_clean = logits_for(model, ds_clean)
        lg_sim, y_sim = logits_for(model, ds_sim)
        lg_all, y_all = torch.cat([lg_clean, lg_sim]), torch.cat([y_clean, y_sim])
        # Calibration/threshold set: clean only for the reference, clean + sim for the fine-tuned.
        fit_logits, fit_labels, fit_set = (lg_clean, y_clean, "val clean") if name == "reference" else (lg_all, y_all, "val clean + val sim")
        t_fit = fit_temperature(fit_logits, fit_labels)
        ece_b = expected_calibration_error(fit_logits, fit_labels)
        ece_a = expected_calibration_error(fit_logits, fit_labels, t_fit)
        temperature = t_fit if ece_a <= ece_b else 1.0
        if temperature == 1.0:
            ece_a = ece_b
        probs_fit = F.softmax(fit_logits / temperature, dim=1)
        threshold, sweep = sweep_threshold(probs_fit, fit_labels)
        m_thr = dict((r["threshold"], r) for r in sweep)[threshold]
        m_060 = decision_metrics(apply_threshold(probs_fit, 0.60, UNKNOWN_IDX), fit_labels, UNKNOWN_IDX)
        m_argmax = decision_metrics(apply_threshold(probs_fit, 0.0, UNKNOWN_IDX), fit_labels, UNKNOWN_IDX)

        def subset_report(lg, y, label):
            if len(y) == 0:
                return None
            p = F.softmax(lg / temperature, dim=1)
            out = {"n": int(len(y)), "ece_before": expected_calibration_error(lg, y),
                   "ece_after": expected_calibration_error(lg, y, temperature),
                   "at_chosen_threshold": decision_metrics(apply_threshold(p, threshold, UNKNOWN_IDX), y, UNKNOWN_IDX)}
            own_t, _ = sweep_threshold(p, y)
            out["threshold_this_subset_alone_would_pick"] = own_t
            out["at_own_threshold"] = decision_metrics(apply_threshold(p, own_t, UNKNOWN_IDX), y, UNKNOWN_IDX)
            return out

        entry = {
            "checkpoint": os.path.relpath(path, ROOT), "checkpoint_sha256_before": sha256_of(path),
            "epoch": ckpt.get("epoch"), "fit_set": fit_set, "n_fit": int(len(fit_labels)),
            "temperature_fitted": t_fit, "temperature": temperature,
            "ece_before": ece_b, "ece_after": ece_a, "threshold": threshold,
            "val_metrics_at_threshold": m_thr, "val_metrics_at_0.60": m_060, "val_metrics_argmax_only": m_argmax,
            "val_clean": subset_report(lg_clean, y_clean, "clean"), "val_sim": subset_report(lg_sim, y_sim, "sim"),
        }
        # plots + csv
        conf_b, pred_b = F.softmax(fit_logits, dim=1).max(dim=1)
        conf_a, pred_a = probs_fit.max(dim=1)
        plot_reliability(conf_b.numpy(), (pred_b == fit_labels).numpy(), conf_a.numpy(), (pred_a == fit_labels).numpy(),
                         ece_b, ece_a, temperature, f"Reliability - {name} CNN on {fit_set} (n={len(fit_labels)})",
                         PLOTS_DIR / f"reliability_{name}.png")
        write_csv(sweep, EVAL_DIR / f"threshold_sweep_{name}{TAG}.csv")
        plot_threshold_sweep(sweep, threshold, f"'unknown' threshold sweep - {name} CNN on {fit_set} (T={temperature:.3f})",
                             PLOTS_DIR / f"threshold_sweep_{name}.png")
        if name == "reference" and len(y_sim):
            # Information: how the reference's threshold curve looks on sim val.
            _, sweep_sim = sweep_threshold(F.softmax(lg_sim / temperature, dim=1), y_sim)
            write_csv(sweep_sim, EVAL_DIR / "threshold_sweep_reference_on_val_sim.csv")
            plot_threshold_sweep(sweep_sim, threshold, "reference CNN threshold curve on val SIM (info; threshold chosen on clean)",
                                 PLOTS_DIR / "threshold_sweep_reference_on_val_sim.png")
        # Store in checkpoint
        ckpt["temperature"] = float(temperature)
        ckpt["threshold"] = float(threshold)
        ckpt["val_ece_before"], ckpt["val_ece_after"] = float(ece_b), float(ece_a)
        ckpt["val_threshold_metrics"] = m_thr
        ckpt.setdefault("sprint8", {})["calibration"] = {
            "fit_set": fit_set, "n_fit": int(len(fit_labels)), "temperature": float(temperature),
            "threshold": float(threshold), "ece_before": float(ece_b), "ece_after": float(ece_a)}
        torch.save(ckpt, path)
        check_loadable(path)
        entry["checkpoint_sha256"] = sha256_of(path)
        report["models"][name] = entry
        frozen_models[name] = {"checkpoint": os.path.relpath(path, ROOT), "sha256": entry["checkpoint_sha256"],
                               "temperature": float(temperature), "threshold": float(threshold),
                               "epoch": ckpt.get("epoch"), "calibrated_on": fit_set}
        print(f"  {name:10s}: T {temperature:.3f} (fit {t_fit:.3f}) ECE {ece_b:.4f} -> {ece_a:.4f} | threshold {threshold:.2f} | "
              f"val balanced {m_thr['balanced_acc']:.4f} digits {m_thr['digit_acc']:.4f} unk-rej {m_thr['unknown_rejection']:.4f}"
              f" | clean@thr digits {entry['val_clean']['at_chosen_threshold']['digit_acc']:.4f}"
              + (f" | sim@thr digits {entry['val_sim']['at_chosen_threshold']['digit_acc']:.4f}" if entry["val_sim"] else ""),
              flush=True)

    # Deployed checkpoint: information only, keep its own stored T / threshold.
    if DEPLOYED.exists() and not extra:
        dck = torch.load(DEPLOYED, map_location="cpu")
        frozen_models["deployed"] = {"checkpoint": os.path.relpath(DEPLOYED, ROOT), "sha256": sha256_of(DEPLOYED),
                                     "temperature": float(dck.get("temperature", 1.0)), "threshold": float(dck.get("threshold", 0.60)),
                                     "epoch": dck.get("epoch"), "calibrated_on": "its own Kaggle validation fold (stored in checkpoint, not re-fitted)",
                                     "note": DEPLOYED_NOTE, "read_only": True}
        report["models"]["deployed"] = {"info_only": True, **frozen_models["deployed"]}

    report["wall_seconds"] = time.time() - t_start
    if prev_report:
        report = {**prev_report, "models": report["models"],
                  "additions": prev_report.get("additions", []) + [{"added_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                                                                     "models": list(models), "wall_seconds": report["wall_seconds"]}]}
    save_json(report, EVAL_DIR / f"calibration_val{TAG}.json")

    # Freeze everything used for test, including acceptance criteria.
    ref_v = report["models"].get("reference", {})
    ft_v = report["models"].get("finetuned", {})
    frozen = {
        "frozen_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "rule": "Test split is read once, after this file is written. No setting below may change afterwards.",
        "manifest": report["manifest"], "manifest_sha256": report["manifest_sha256"],
        "test_split": "test", "val_split": args.val_split,
        "decision_rule": "probs = softmax(logits / T); pred = argmax; if max(probs) < threshold -> 'unknown' (calibration.apply_threshold)",
        "labels": LABELS, "unknown_index": UNKNOWN_IDX,
        "audio_paths": {"clean": "preprocess_file (dataset.py path, no band-pass, no enhance)",
                        "sim": f"load_audio -> Butterworth-{LIVE_BANDPASS_ORDER} band-pass {list(LIVE_BANDPASS)} Hz (sosfiltfilt) -> enhance_waveform -> preprocess_waveform (dashboard laser channel)"},
        "conditions": {"clean": "kind == clean", "sim_good": "condition == sim_good (snr >= 20 dB)",
                       "sim_medium": "condition == sim_medium (10 <= snr < 20 dB)", "sim_poor": "condition == sim_poor (snr < 10 dB)",
                       "sim_all": "kind == sim", "manit_clean": "kind == clean and speaker == manit"},
        "metrics": ["n_digit", "n_unknown (synthetic noise / real non-digit speech)", "digit_acc", "digit_rejection_rate",
                    "unknown_rejection_rate", "overall_acc_11", "balanced_acc (mean recall over classes present)",
                    "per_class_acc", "confusion_matrix 11x11 (0-9, unknown)"],
        "models": frozen_models,
        "seeds": {"seed": args.seed, "note": "evaluation is deterministic; seed only fixes torch/numpy/random state"},
        "feature": "mel (40 log-mel bands, standardised), 8 kHz, 1.0 s clips",
        "batch_size_eval": 64, "torch_threads": args.threads,
        "acceptance_criteria": [
            {"id": "AC1", "statement": "fine-tuned test sim_all digit_acc >= reference test sim_all digit_acc + 0.05",
             "rationale": "the fine-tune must buy a clear gain on simulated laser audio (Exp 6 saw +12 pts)"},
            {"id": "AC2", "statement": "fine-tuned test clean digit_acc >= reference test clean digit_acc - 0.03",
             "rationale": "clean-audio accuracy may not be sacrificed by more than 3 points"},
            {"id": "AC3", "statement": "fine-tuned test sim_all balanced_acc >= reference test sim_all balanced_acc",
             "rationale": "unknown rejection on sim must not collapse while digits improve"},
            {"id": "AC4", "statement": "both models: test clean digit_rejection_rate <= 0.05",
             "rationale": "the tuned threshold must not reject real digits"},
        ],
        "val_preview_for_criteria": {
            "reference_val_clean_digit_acc": (ref_v.get("val_clean") or {}).get("at_chosen_threshold", {}).get("digit_acc"),
            "reference_val_sim_digit_acc": (ref_v.get("val_sim") or {}).get("at_chosen_threshold", {}).get("digit_acc"),
            "finetuned_val_clean_digit_acc": (ft_v.get("val_clean") or {}).get("at_chosen_threshold", {}).get("digit_acc"),
            "finetuned_val_sim_digit_acc": (ft_v.get("val_sim") or {}).get("at_chosen_threshold", {}).get("digit_acc"),
        },
    }
    if prev_frozen and TAG:
        frozen = dict(prev_frozen)
        frozen.setdefault("rounds", {})[TAG.lstrip("_")] = {
            "frozen_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "manifest": report["manifest"], "manifest_sha256": report["manifest_sha256"], "test_split": "test",
            "models": frozen_models, "note": args.note or "",
            "outputs": f"calibration_val{TAG}.json, test_metrics{TAG}.json/.md, plots{TAG}/",
            "acceptance_criteria": frozen["acceptance_criteria"],
            "criteria_note": "same metrics/conditions/decision rule as round 1; AC1-AC4 evaluated with the round's fine-tuned model vs reference"}
    elif prev_frozen:
        frozen = dict(prev_frozen)
        frozen["models"] = frozen_models
        frozen.setdefault("additions", []).append({
            "frozen_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "models": {n: frozen_models[n] for n in models},
            "note": args.note or "added after the first test run; settings fixed on validation before these models' test scores were seen",
            "criteria": "scored under the same conditions and metrics; original models' test rows are not recomputed"})
    save_json(frozen, EVAL_DIR / "FROZEN_SETTINGS.json")

    # checksums + MODELS_READY
    lines = []
    for name, info in frozen_models.items():
        lines.append(f"{info['sha256']}  {info['checkpoint']}")
    models_dir = Path(args.reference).parent
    if not TAG:
        (models_dir / "checksums.txt").write_text("\n".join(lines) + "\n")
    if TAG:
        with open(models_dir / "MODELS_READY", "a") as fh:
            for n in models:
                i = frozen_models[n]
                fh.write(f"{n} (round {TAG.lstrip('_')}, {_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')}): "
                         f"{i['checkpoint']} T={i['temperature']:.4f} thr={i['threshold']:.2f} sha256={i['sha256']}\n")
        with open(models_dir / "checksums.txt", "a") as fh:
            for n in models:
                fh.write(f"{frozen_models[n]['sha256']}  {frozen_models[n]['checkpoint']}\n")
        print("MODELS_READY / checksums.txt appended.")
    elif "reference" in frozen_models and "finetuned" in frozen_models:
        (models_dir / "MODELS_READY").write_text(
            f"written {frozen['frozen_at_utc']} by scripts/sprint8/evaluate_sprint8.py calibrate\n"
            + "\n".join(f"{n}: {i['checkpoint']} T={i['temperature']:.4f} thr={i['threshold']:.2f} sha256={i['sha256']}"
                        for n, i in frozen_models.items() if n != "deployed") + "\n")
        print("MODELS_READY written.")
    print(f"calibrate done in {report['wall_seconds']:.0f}s -> {EVAL_DIR / f'calibration_val{TAG}.json'}, FROZEN_SETTINGS.json")
    return 0


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------
def cmd_test(args) -> int:
    t_start = time.time()
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    frozen_path = EVAL_DIR / "FROZEN_SETTINGS.json"
    if not frozen_path.exists():
        print("ERROR: FROZEN_SETTINGS.json missing - run `calibrate` first."); return 2
    frozen = json.loads(frozen_path.read_text())
    if TAG:
        rnd = frozen["rounds"][TAG.lstrip("_")]
        frozen = {**frozen, "models": rnd["models"], "manifest": rnd["manifest"], "manifest_sha256": rnd["manifest_sha256"],
                  "test_split": rnd["test_split"], "round": TAG.lstrip("_")}
    prev = None
    if (EVAL_DIR / f"test_metrics{TAG}.json").exists():
        prev = json.loads((EVAL_DIR / f"test_metrics{TAG}.json").read_text())
        if not args.models and not args.force:
            print("ERROR: test_metrics.json already exists. The test split is scored once; pass --force only to redo a failed run.")
            return 2
        if args.models:
            dup = [m for m in args.models if m in prev["models"] and not prev["models"][m].get("missing")]
            if dup and not args.force:
                print(f"ERROR: {dup} already have test scores; refusing to re-score."); return 2
    manifest_path = Path(args.manifest)
    man_sha = sha256_of(manifest_path)
    if man_sha != frozen["manifest_sha256"]:
        print(f"ERROR: manifest sha256 {man_sha[:12]} != frozen {frozen['manifest_sha256'][:12]}"); return 2
    rows = read_manifest(manifest_path)
    test_clean = select_rows(rows, split=frozen["test_split"], kind="clean")
    test_sim = select_rows(rows, split=frozen["test_split"], kind="sim")
    if args.limit:
        test_clean, test_sim = test_clean[:args.limit], test_sim[:args.limit]
    print(f"test clean {len(test_clean)} {label_counts(test_clean)}")
    print(f"test sim   {len(test_sim)} {label_counts(test_sim)}", flush=True)
    ds_clean = ManifestDataset(test_clean, augment=False)
    ds_sim = ManifestDataset(test_sim, augment=False)
    print("Preloading test waveforms ...", flush=True)
    t_pre = ds_clean.preload(label="test clean") + ds_sim.preload(label="test sim")

    # Condition masks over the concatenated [clean..., sim...] order.
    all_rows = test_clean + test_sim
    idx_clean = np.arange(len(test_clean))
    idx_sim = np.arange(len(test_clean), len(all_rows))
    cond_idx = {"clean": idx_clean, "sim_all": idx_sim}
    for c in ("sim_good", "sim_medium", "sim_poor"):
        cond_idx[c] = np.array([i for i in idx_sim if all_rows[i]["condition"] == c], dtype=int)
    cond_idx["manit_clean"] = np.array([i for i in idx_clean if all_rows[i]["speaker"] == "manit"], dtype=int)

    results = {"frozen_settings": frozen, "manifest_sha256": man_sha, "n_test_clean": len(test_clean),
               "n_test_sim": len(test_sim), "test_speakers": sorted({r["speaker"] for r in all_rows}),
               "conditions": {c: int(len(ix)) for c, ix in cond_idx.items()}, "models": {}}
    for name, info in frozen["models"].items():
        if args.models and name not in args.models:
            continue
        path = ROOT / info["checkpoint"]
        if not path.exists():
            results["models"][name] = {"missing": True}; continue
        sha = sha256_of(path)
        if sha != info["sha256"]:
            print(f"ERROR: {name} checkpoint sha256 changed since freeze ({sha[:12]} != {info['sha256'][:12]})"); return 2
        ckpt, model = load_model(path)
        T, thr = float(info["temperature"]), float(info["threshold"])
        lg_c, y_c = logits_for(model, ds_clean)
        lg_s, y_s = logits_for(model, ds_sim)
        logits, y = torch.cat([lg_c, lg_s]), torch.cat([y_c, y_s])
        probs = F.softmax(logits / T, dim=1)
        pred = apply_threshold(probs, thr, UNKNOWN_IDX).numpy()
        y = y.numpy()
        entry = {"checkpoint": info["checkpoint"], "sha256": sha, "temperature": T, "threshold": thr,
                 "epoch": ckpt.get("epoch"), "note": info.get("note"), "conditions": {}}
        for cond, ix in cond_idx.items():
            if len(ix) == 0:
                entry["conditions"][cond] = {"n_clips": 0, "note": "no clips in this condition"}; continue
            m = full_metrics(pred[ix], y[ix], [all_rows[i] for i in ix])
            entry["conditions"][cond] = m
            title = (f"{name} CNN - {cond} (test, n={m['n_clips']})  digits {pct(m['digit_acc'])}%  "
                     f"unk-rej {pct(m['unknown_rejection_rate'])}%  bal {pct(m['balanced_acc'])}%  T={T:.2f} thr={thr:.2f}")
            plot_confusion_matrix(m["confusion_matrix"], LABELS, title, PLOTS_DIR / f"confusion_{name}_{cond}.png")
            print(f"  {name:10s} {cond:11s} n={m['n_clips']:5d} digits {pct(m['digit_acc']):>5}% rej {pct(m['digit_rejection_rate']):>5}% "
                  f"unk-rej {pct(m['unknown_rejection_rate']):>5}% overall {pct(m['overall_acc_11']):>5}% bal {pct(m['balanced_acc']):>5}%", flush=True)
        results["models"][name] = entry

    if prev and args.models:
        merged = dict(prev["models"]); merged.update(results["models"])
        results["models"] = merged
        results["additions"] = prev.get("additions", []) + [{"scored_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                                                             "models": list(args.models)}]
    # Acceptance criteria
    def g(model, cond, key):
        try:
            return results["models"][model]["conditions"][cond][key]
        except KeyError:
            return None
    ac = []
    FT = args.ft_name
    r_sim, f_sim = g("reference", "sim_all", "digit_acc"), g(FT, "sim_all", "digit_acc")
    r_cl, f_cl = g("reference", "clean", "digit_acc"), g(FT, "clean", "digit_acc")
    r_bal, f_bal = g("reference", "sim_all", "balanced_acc"), g(FT, "sim_all", "balanced_acc")
    r_rej, f_rej = g("reference", "clean", "digit_rejection_rate"), g(FT, "clean", "digit_rejection_rate")
    results["acceptance_finetuned_model"] = FT
    ok = lambda cond: None if cond is None else bool(cond)  # noqa: E731
    ac.append({"id": "AC1", "pass": ok(None if None in (r_sim, f_sim) else f_sim >= r_sim + 0.05),
               "values": {"reference_sim_all_digit_acc": r_sim, "finetuned_sim_all_digit_acc": f_sim,
                          "delta": None if None in (r_sim, f_sim) else f_sim - r_sim}})
    ac.append({"id": "AC2", "pass": ok(None if None in (r_cl, f_cl) else f_cl >= r_cl - 0.03),
               "values": {"reference_clean_digit_acc": r_cl, "finetuned_clean_digit_acc": f_cl,
                          "delta": None if None in (r_cl, f_cl) else f_cl - r_cl}})
    ac.append({"id": "AC3", "pass": ok(None if None in (r_bal, f_bal) else f_bal >= r_bal),
               "values": {"reference_sim_all_balanced_acc": r_bal, "finetuned_sim_all_balanced_acc": f_bal}})
    ac.append({"id": "AC4", "pass": ok(None if None in (r_rej, f_rej) else (r_rej <= 0.05 and f_rej <= 0.05)),
               "values": {"reference_clean_digit_rejection": r_rej, "finetuned_clean_digit_rejection": f_rej}})
    results["acceptance"] = {"criteria": frozen["acceptance_criteria"], "results": ac,
                             "all_pass": all(a["pass"] for a in ac if a["pass"] is not None) and all(a["pass"] is not None for a in ac)}
    results["wall_seconds"] = time.time() - t_start
    results["wall_seconds_preload"] = t_pre
    results["environment"] = env_info()
    save_json(results, EVAL_DIR / f"test_metrics{TAG}.json")
    write_markdown(results, EVAL_DIR / f"test_metrics{TAG}.md")
    print(f"\nAcceptance: " + ", ".join(f"{a['id']}={'PASS' if a['pass'] else ('n/a' if a['pass'] is None else 'FAIL')}" for a in ac))
    print(f"test done in {results['wall_seconds']:.0f}s -> {EVAL_DIR / f'test_metrics{TAG}.json'}, test_metrics.md, plots/")
    return 0


def write_markdown(results: dict, path: Path) -> None:
    L = []
    L.append(f"# Sprint 8 test results (split=test, scored once){' - round ' + results['frozen_settings']['round'] if results['frozen_settings'].get('round') else ''}\n")
    if results.get("acceptance_finetuned_model"):
        L.append(f"Acceptance criteria evaluated with `{results['acceptance_finetuned_model']}` as the fine-tuned model.\n")
    L.append(f"Manifest sha256 `{results['manifest_sha256'][:16]}…`; test clean clips {results['n_test_clean']}, "
             f"test sim clips {results['n_test_sim']}; {len(results['test_speakers'])} test speakers. "
             f"Compute: cloud dev container (2 vCPU, CPU-only PyTorch 2.15). Wall time {results['wall_seconds']:.0f} s.\n")
    L.append("Decision rule: softmax(logits / T), argmax, `unknown` if max prob < threshold. Sim clips are fed through the "
             "dashboard laser path (band-pass 200-3800 Hz + enhance); clean clips through the plain path.\n")
    if results.get("additions"):
        L.append("Later additions (pre-registered on validation, then scored once): " +
                 "; ".join(f"{a['scored_utc']}: {', '.join(a['models'])}" for a in results["additions"]) + "\n")
    L.append("## Models\n")
    L.append("| model | checkpoint | epoch | T | threshold | note |")
    L.append("|---|---|---|---|---|---|")
    for name, e in results["models"].items():
        if e.get("missing"):
            L.append(f"| {name} | missing | | | | |"); continue
        L.append(f"| {name} | `{e['checkpoint']}` | {e['epoch']} | {e['temperature']:.3f} | {e['threshold']:.2f} | {e.get('note') or ''} |")
    L.append("")
    L.append("## Headline table (percent)\n")
    L.append("| model | condition | n digit | n unknown (synth / speech) | digit acc | digits rejected | unknown rejected (synth / speech) | overall 11-class | balanced acc |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for name, e in results["models"].items():
        if e.get("missing"):
            continue
        for cond in CONDITIONS + ["manit_clean"]:
            m = e["conditions"].get(cond)
            if not m or m.get("n_clips", 0) == 0:
                L.append(f"| {name} | {cond} | 0 | - | - | - | - | - | - |"); continue
            L.append(f"| {name} | {cond} | {m['n_digit']} | {m['n_unknown']} ({m['n_unknown_synthetic_noise']} / {m['n_unknown_real_speech']}) | "
                     f"{pct(m['digit_acc'])} | {pct(m['digit_rejection_rate'])} | {pct(m['unknown_rejection_rate'])} "
                     f"({pct(m['unknown_rejection_rate_synthetic_noise'])} / {pct(m['unknown_rejection_rate_real_speech'])}) | "
                     f"{pct(m['overall_acc_11'])} | {pct(m['balanced_acc'])} |")
    L.append("")
    L.append("## Acceptance criteria (defined in FROZEN_SETTINGS.json before testing)\n")
    L.append("| id | statement | result | values |")
    L.append("|---|---|---|---|")
    crit = {c["id"]: c["statement"] for c in results["acceptance"]["criteria"]}
    for a in results["acceptance"]["results"]:
        verdict = "PASS" if a["pass"] else ("n/a" if a["pass"] is None else "FAIL")
        vals = ", ".join(f"{k}={'-' if v is None else f'{v:.4f}'}" for k, v in a["values"].items())
        L.append(f"| {a['id']} | {crit[a['id']]} | **{verdict}** | {vals} |")
    L.append("")
    L.append("## Per-class accuracy (percent; digits = correct digit after threshold, unknown = rejected)\n")
    for name, e in results["models"].items():
        if e.get("missing"):
            continue
        L.append(f"### {name}\n")
        L.append("| condition | " + " | ".join(LABELS) + " |")
        L.append("|---|" + "---|" * len(LABELS))
        for cond in CONDITIONS + ["manit_clean"]:
            m = e["conditions"].get(cond)
            if not m or m.get("n_clips", 0) == 0:
                continue
            L.append(f"| {cond} | " + " | ".join(pct(m["per_class_acc"][l]["acc"]) for l in LABELS) + " |")
        L.append("")
    L.append("## Confusion matrices\n")
    L.append("PNG per model x condition under `results/sprint8/eval/plots/confusion_<model>_<condition>.png`; "
             "the raw 11x11 counts (rows = true, columns = predicted, order 0-9, unknown) are in `test_metrics.json`.\n")
    path.write_text("\n".join(L))


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------
def cmd_environment(args) -> int:
    info = env_info()
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120).stdout
    except Exception as e:  # noqa: BLE001
        freeze = f"(pip freeze failed: {e})"
    keep = ("torch", "torchaudio", "numpy", "scipy", "soundfile", "scikit-learn", "matplotlib", "pandas")
    excerpt = [l for l in freeze.splitlines() if any(l.lower().startswith(k) for k in keep)]
    timings = []
    for stage, p in (("reference training", TRAIN_DIR / "reference/summary.json"),
                     ("fine-tune (finetuned_cnn.pt, max 15 ep)", TRAIN_DIR / "finetune/summary.json"),
                     ("control continued_clean (no sim)", TRAIN_DIR / "control/summary.json"),
                     ("fine-tune long (max 30 ep)", TRAIN_DIR / "finetune_long/summary.json"),
                     ("calibration round 1 (val; last run = additions)", EVAL_DIR / "calibration_val.json"),
                     ("test round 1 (last run = additions only)", EVAL_DIR / "test_metrics.json"),
                     ("fine-tune v2 (manifest_v2, max 30 ep)", TRAIN_DIR / "finetune_v2/summary.json"),
                     ("calibration round 2 (v2 val)", EVAL_DIR / "calibration_val_v2.json"),
                     ("test round 2 (v2 test)", EVAL_DIR / "test_metrics_v2.json")):
        if p.exists():
            d = json.loads(p.read_text())
            secs = d.get("wall_seconds_total", d.get("wall_seconds"))
            extra = ""
            if "mean_epoch_seconds" in d:
                extra = f" ({d.get('epochs_run')} epochs, mean {d['mean_epoch_seconds']:.0f} s/epoch, preload {d.get('wall_seconds_preload', 0):.0f} s)"
            timings.append(f"| {stage} | {secs / 60:.1f} min{extra} |")
        else:
            timings.append(f"| {stage} | not run |")
    md = ["# Sprint 8 training/evaluation environment (Agent B)\n",
          "Compute: cloud dev container (2 vCPU, CPU-only PyTorch 2.15).\n",
          f"- Python {info['python']}", f"- torch {info['torch']} (device {info['device']}, threads {info['torch_threads']})",
          f"- platform {info['platform']}", f"- CPU: {info['cpu']} x {info['cpu_count']} logical cores",
          f"- sample rate {info['sample_rate']} Hz", f"- LMML_OUTPUT_DIR={info['output_dir_env']}", "",
          "## pip freeze (excerpt)\n", "```", *excerpt, "```", "",
          "## Measured wall time per stage\n", "| stage | wall time |", "|---|---|", *timings, ""]
    (TRAIN_DIR / "ENVIRONMENT.md").write_text("\n".join(md))
    print(f"wrote {TRAIN_DIR / 'ENVIRONMENT.md'}")
    return 0


def main() -> int:
    global EVAL_DIR, PLOTS_DIR
    ap = argparse.ArgumentParser(description="Sprint 8 calibration (val) and single test run.")
    ap.add_argument("command", choices=["calibrate", "test", "environment"])
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--reference", default=str(MODELS_DIR / "reference_cnn.pt"))
    ap.add_argument("--finetuned", default=str(MODELS_DIR / "finetuned_cnn.pt"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="debug only")
    ap.add_argument("--force", action="store_true", help="test: overwrite an existing test_metrics.json")
    ap.add_argument("--eval-dir", default=str(EVAL_DIR), help="debug: redirect eval outputs (default results/sprint8/eval)")
    ap.add_argument("--extra", action="append", help="calibrate: add a model after the first freeze, as name=path (repeatable)")
    ap.add_argument("--models", nargs="+", help="test: score only these frozen model names and merge into existing results")
    ap.add_argument("--note", default=None, help="calibrate --extra: note recorded in FROZEN_SETTINGS.json additions")
    ap.add_argument("--tag", default="", help="round tag, e.g. v2: suffixes calibration/test outputs and stores settings under FROZEN_SETTINGS.rounds")
    ap.add_argument("--carry", nargs="*", help="calibrate --tag: round-1 frozen models carried into this round with unchanged settings")
    ap.add_argument("--ft-name", default="finetuned", help="test: which model plays 'fine-tuned' in the acceptance criteria")
    args = ap.parse_args()
    global TAG
    TAG = f"_{args.tag}" if args.tag else ""
    EVAL_DIR = Path(args.eval_dir)
    PLOTS_DIR = EVAL_DIR / f"plots{TAG}"
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    return {"calibrate": cmd_calibrate, "test": cmd_test, "environment": cmd_environment}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
