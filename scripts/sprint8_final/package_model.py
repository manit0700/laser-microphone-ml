"""
package_model.py (Sprint 8 final, Agent C)
Creates the reproducible package for the historical Sprint 8 frozen candidate `finetuned_v2`:
  - copies (never moves or overwrites) the frozen checkpoint into results/sprint8_final/package/model/
  - writes config/model_config.json with every setting the predictor needs (no reads of deployed models/)
  - writes SHA256SUMS
Run from anywhere:  python scripts/sprint8_final/package_model.py
"""
import hashlib, json, shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results/sprint8/models/finetuned_v2_cnn.pt"
PKG = REPO / "results/sprint8_final/package"
EXPECTED_SHA = "9d8a11081dcfc9e6d97d9afc84717f405623fd602117248390687e28458d1598"
FROZEN_T = 0.5721844434738159     # results/sprint8/eval/FROZEN_SETTINGS.json rounds.v2.models.finetuned_v2
FROZEN_THR = 0.26
LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "unknown"]

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    (PKG / "model").mkdir(parents=True, exist_ok=True)
    (PKG / "config").mkdir(parents=True, exist_ok=True)
    dst = PKG / "model" / "finetuned_v2_cnn.pt"
    if not dst.exists():
        shutil.copy2(SRC, dst)
    got = sha256(dst)
    if got != EXPECTED_SHA or sha256(SRC) != EXPECTED_SHA:
        raise SystemExit(f"checksum mismatch: package {got}, source {sha256(SRC)}, expected {EXPECTED_SHA}")
    cfg = {
        "package_id": "lmml-sprint8-finetuned_v2",
        "status": "historical Sprint 8 frozen candidate; EXPERIMENTAL, not validated on hardware",
        "checkpoint": {"path": "model/finetuned_v2_cnn.pt", "sha256": EXPECTED_SHA,
                        "source": "results/sprint8/models/finetuned_v2_cnn.pt (frozen Round 2, epoch 30)"},
        "labels": LABELS, "unknown_index": 10,
        "sample_rate": 8000,        # pinned: do NOT read models/model_meta.json (deployed-model dependency)
        "max_audio_seconds": 1.0,
        "feature": "mel",           # src/features.py extract_features(kind='mel') -> (40, T) standardized log-mel
        "decision": {"temperature": FROZEN_T, "threshold": FROZEN_THR,
                     "rule": "probs = softmax(logits / temperature); pred = argmax over 11 classes; "
                             "reject (unknown) if max(probs) < threshold"},
        "modes": {
            "clean": "load_audio -> preprocess_waveform (no band-pass, no enhancement). Matches training clean path and predict.predict_file.",
            "laser": "load_audio -> Butterworth order-4 zero-phase band-pass 200-3800 Hz -> enhance_waveform -> preprocess_waveform. "
                     "Matches the training 'sim' path and the dashboard laser channel default (LMML_LASER_BANDPASS=200,3800). "
                     "predict.predict_file does NOT do this for laser recordings."},
        "filter_setting": "ENABLE_FILTER=False (reduce_noise off) for Sprint 8 training and replay; the predictor pins this.",
        "dependencies": "Python 3.10+, torch, torchaudio, numpy, scipy, soundfile. Uses the repository's src/ and scripts/sprint8/ modules (repo-relative; no /home/claude paths).",
        "intended_use": "Development demo and offline scoring of development WAVs. Not for deployment decisions; no real-rig accuracy is claimed.",
    }
    (PKG / "config" / "model_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    lines = [f"{EXPECTED_SHA}  model/finetuned_v2_cnn.pt"]
    (PKG / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print("package ready:", PKG, "checkpoint sha256 ok", got[:16])

if __name__ == "__main__":
    main()
