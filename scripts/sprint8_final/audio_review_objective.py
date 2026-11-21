"""
audio_review_objective.py (Sprint 8 final, Agent A objective checks)
Verifies every entry of the 57-clip six-review playlist against (1) byte-identical source files,
(2) the saved validation predictions (manifest pipeline), and computes objective audio checks.
Writes results/sprint8_final/audio_review/ . No labels are changed. No listening is claimed.
"""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "sprint8_followup"))
import common as C
from train_reference import LABELS

ROOT = C.ROOT
PLAY = ROOT / "results/sprint8_followup/listening_six_review"
OUT = ROOT / "results/sprint8_final/audio_review"
OUT.mkdir(parents=True, exist_ok=True)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    rows = list(csv.DictReader(open(PLAY / "label_review_sheet.csv")))
    # saved validation predictions (manifest pipeline, frozen T, threshold)
    l, y, mrows = C.raw_logits("finetuned_v2", "val", "clean")
    probs = C.probs_from(l, C.FROZEN_T["finetuned_v2"])
    pred, conf = C.decide(probs, C.FROZEN_THR["finetuned_v2"])
    idx_by_file = {r["file"]: i for i, r in enumerate(mrows)}
    out = []
    for r in rows:
        src = ROOT / r["source_file"]
        wav = PLAY / r["wav"]
        i = idx_by_file.get(r["source_file"])
        x, sr = sf.read(src, dtype="float32", always_2d=True)
        m = x.mean(axis=1)
        active = np.abs(m) > 10 ** (-40 / 20)           # samples above -40 dBFS
        lead = np.argmax(active) if active.any() else -1
        trail = (len(m) - 1 - np.argmax(active[::-1])) if active.any() else -1
        spec = np.abs(np.fft.rfft(m)) ** 2; fr = np.fft.rfftfreq(len(m), 1 / sr)
        full = np.abs(m) >= 0.999 * np.abs(m).max() if np.abs(m).max() > 0 else np.zeros(len(m), bool)
        # flat-top runs: >=3 consecutive samples at the file's own peak magnitude
        flat = 0; run = 0
        for v in full:
            run = run + 1 if v else 0
            if run == 3: flat += 1
        
        rec = {
            "playlist_no": r["playlist_no"], "group": r["group"], "source_file": r["source_file"],
            "wav_identical_to_source": sha(src) == sha(wav),
            "source_sha256": sha(src)[:16],
            "in_saved_validation_set": i is not None,
            "saved_pred_matches_sheet": i is not None and LABELS[pred[i]] == r["model_pred"],
            "saved_conf_matches_sheet": i is not None and abs(float(conf[i]) - float(r["confidence"])) < 1e-3,
            "manifest_label_matches_filename": Path(r["source_file"]).name.split("_")[0] == r["manifest_label"],
            "sample_rate": int(sr), "duration_s": round(len(m) / sr, 3),
            "peak_abs": round(float(np.abs(m).max()), 4),
            "at_full_scale_peak": bool(np.abs(m).max() >= 0.999),
            "flat_top_runs_ge3": int(flat),
            "leading_silence_ms": round(1000 * lead / sr, 1) if lead >= 0 else None,
            "trailing_silence_ms": round(1000 * (len(m) - 1 - trail) / sr, 1) if trail >= 0 else None,
            "frac_energy_above_3kHz": round(float(spec[fr > 3000].sum() / (spec.sum() + 1e-12)), 4),
            "frac_energy_below_200Hz": round(float(spec[fr < 200].sum() / (spec.sum() + 1e-12)), 4),
            "model_pred": r["model_pred"], "manifest_label": r["manifest_label"],
            "confidence": float(r["confidence"]),
        }
        out.append(rec)
    keys = list(out[0].keys())
    with open(OUT / "objective_checks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out)
    summary = {
        "n_items": len(out),
        "all_wav_identical_to_source": all(o["wav_identical_to_source"] for o in out),
        "all_in_saved_validation_set": all(o["in_saved_validation_set"] for o in out),
        "all_saved_pred_matches_sheet": all(o["saved_pred_matches_sheet"] for o in out),
        "all_saved_conf_matches_sheet": all(o["saved_conf_matches_sheet"] for o in out),
        "manifest_label_matches_filename": all(o["manifest_label_matches_filename"] for o in out),
        "sample_rates": {str(k): sum(1 for o in out if o["sample_rate"] == k) for k in {o["sample_rate"] for o in out}},
        "items_at_full_scale_peak": sum(1 for o in out if o["at_full_scale_peak"]),
        "items_with_flat_top_runs": [o["playlist_no"] for o in out if o["flat_top_runs_ge3"] > 0],
        "error_items": sum(1 for o in out if o["group"] == "error"),
        "control_items": sum(1 for o in out if o["group"] == "control_correct"),
        "error_pred_counts": {p: sum(1 for o in out if o["group"] == "error" and o["model_pred"] == p) for p in sorted({o["model_pred"] for o in out if o["group"] == "error"})},
        "median_duration_error_s": float(np.median([o["duration_s"] for o in out if o["group"] == "error"])),
        "median_duration_control_s": float(np.median([o["duration_s"] for o in out if o["group"] == "control_correct"])),
        "median_peak_error": float(np.median([o["peak_abs"] for o in out if o["group"] == "error"])),
        "median_peak_control": float(np.median([o["peak_abs"] for o in out if o["group"] == "control_correct"])),
        "median_hf_share_error": float(np.median([o["frac_energy_above_3kHz"] for o in out if o["group"] == "error"])),
        "median_hf_share_control": float(np.median([o["frac_energy_above_3kHz"] for o in out if o["group"] == "control_correct"])),
    }
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pick = [o for o in out if o["group"] == "error"][:6] + [o for o in out if o["group"] == "control_correct"][:3]
    fig, axes = plt.subplots(len(pick), 2, figsize=(9, 1.6 * len(pick)))
    for ax_row, o in zip(axes, pick):
        x, sr = sf.read(ROOT / o["source_file"], dtype="float32", always_2d=True); m = x.mean(axis=1)
        ax_row[0].plot(np.arange(len(m)) / sr, m, lw=0.6); ax_row[0].set_ylim(-1.05, 1.05)
        ax_row[0].set_title(f"#{o['playlist_no']} {o['group']} label={o['manifest_label']} pred={o['model_pred']} sr={sr}", fontsize=7)
        ax_row[1].specgram(m, Fs=sr, NFFT=256, noverlap=128, cmap="magma"); ax_row[1].set_ylim(0, 4000)
        ax_row[1].set_title("spectrogram (0-4 kHz)", fontsize=7)
    for ax in axes.flat: ax.tick_params(labelsize=6)
    fig.suptitle("Inspection plots only (not listening). Top rows: errors; bottom: correct controls.", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "waveform_spectrogram_inspection.png", dpi=110)
    (OUT / "objective_checks_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
