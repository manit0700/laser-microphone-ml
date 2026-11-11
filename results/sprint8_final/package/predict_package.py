#!/usr/bin/env python3
"""
predict_package.py - standalone predictor for the Sprint 8 package (lmml-sprint8-finetuned_v2).

Loads ONLY the checkpoint named in config/model_config.json (sha256-checked), never models/best_model*.pt.
Pins the sample rate and decision settings from the config, so results do not depend on deployed-model files.

Usage (run from any directory):
  python /path/to/package/predict_package.py clean  path/to/file.wav [more.wav ...]
  python /path/to/package/predict_package.py laser  path/to/file.wav [more.wav ...] --json out.json
  python predict_package.py --selftest           # checks the package against the repository pipeline

Repository location: defaults to the repo that contains this package (results/sprint8_final/package -> repo root).
Override with --repo or the LMML_REPO environment variable if the package is copied elsewhere.
Exit codes: 0 ok, 2 usage/config error, 3 checkpoint problem, 4 input problem.
"""
import argparse, hashlib, json, os, sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
DEFAULT_REPO = PKG.parents[2]          # results/sprint8_final/package -> repo root


def fail(code, msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config():
    cfg_path = PKG / "config" / "model_config.json"
    if not cfg_path.exists():
        fail(2, f"config not found: {cfg_path}. Re-run scripts/sprint8_final/package_model.py.")
    try:
        return json.loads(cfg_path.read_text())
    except json.JSONDecodeError as e:
        fail(2, f"config is not valid JSON ({cfg_path}): {e}")


def setup(repo: Path, cfg: dict):
    if not (repo / "src" / "features.py").exists():
        fail(2, f"repository src/ not found under {repo}. Pass --repo /path/to/laser_microphone_ml or set LMML_REPO.")
    # pin the sample rate BEFORE the repository config is imported (it otherwise reads models/model_meta.json)
    os.environ["LMML_SAMPLE_RATE"] = str(cfg["sample_rate"])
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo / "scripts" / "sprint8"))


def load_checkpoint(cfg: dict):
    import torch
    ck = PKG / cfg["checkpoint"]["path"]
    if not ck.exists():
        fail(3, f"checkpoint not found: {ck}")
    got = sha256(ck)
    if got != cfg["checkpoint"]["sha256"]:
        fail(3, f"checkpoint checksum mismatch for {ck}: got {got[:16]}..., expected {cfg['checkpoint']['sha256'][:16]}...")
    try:
        state = torch.load(ck, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        fail(3, f"could not read checkpoint {ck}: {type(e).__name__}: {e}")
    from model import model_from_checkpoint
    model = model_from_checkpoint(state)
    model.eval()
    return model


def bandpass_sos(sample_rate: int):
    """Same filter as scripts/sprint8/train_reference._bandpass_sos (laser path)."""
    from scipy.signal import butter
    low, high = 200.0, 3800.0
    high = min(high, sample_rate / 2 * 0.95)
    return butter(4, [low, high], btype="band", fs=sample_rate, output="sos")


def features_for(path: Path, mode: str, cfg: dict):
    import numpy as np
    from scipy.signal import sosfiltfilt
    from preprocess import load_audio, preprocess_waveform
    from features import extract_features
    try:
        w = load_audio(path)
    except Exception as e:  # noqa: BLE001
        fail(4, f"cannot read audio {path}: {type(e).__name__}: {e}")
    if w.numel() == 0:
        fail(4, f"empty audio file: {path}")
    if mode == "laser":
        from enhance import enhance_waveform
        import torch
        if w.numel() > 64:
            w = torch.from_numpy(np.ascontiguousarray(sosfiltfilt(bandpass_sos(cfg["sample_rate"]), w.numpy()),
                                                      dtype=np.float32))
        w = enhance_waveform(w)
    clean = preprocess_waveform(w)
    return extract_features(clean, cfg["feature"]).unsqueeze(0)


def decide(logits, cfg):
    import torch
    T = cfg["decision"]["temperature"]; thr = cfg["decision"]["threshold"]
    probs = torch.softmax(logits / T, dim=1)[0]
    idx = int(torch.argmax(probs))
    conf = float(probs[idx])
    rejected = conf < thr
    label = cfg["labels"][cfg["unknown_index"]] if rejected else cfg["labels"][idx]
    return label, conf, rejected, probs


def predict(paths, mode: str, cfg: dict, model):
    import torch
    out = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            fail(4, f"input file not found: {p}")
        feat = features_for(p, mode, cfg)
        with torch.no_grad():
            logits = model(feat)
        label, conf, rejected, probs = decide(logits, cfg)
        out.append({"file": str(p), "mode": mode, "label": label, "confidence": round(conf, 6),
                    "rejected_as_unknown": bool(rejected),
                    "probabilities": {l: round(float(v), 6) for l, v in zip(cfg["labels"], probs)},
                    "checkpoint_sha256": cfg["checkpoint"]["sha256"], "temperature": cfg["decision"]["temperature"],
                    "threshold": cfg["decision"]["threshold"], "sample_rate": cfg["sample_rate"]})
    return out


def selftest(repo: Path, cfg: dict, model):
    """Check the package against the repository's own pipeline on fixed development examples (validation)."""
    import csv, torch
    from calibration import apply_threshold  # noqa: F401  (import check)
    from train_reference import load_live_path
    from features import extract_features
    rows = [r for r in csv.DictReader(open(repo / "results/sprint8/data/manifest_v2.csv")) if r["split"] == "val"]
    clean = [r for r in rows if r["kind"] == "clean"][:12]
    sim = [r for r in rows if r["kind"] == "sim"][:12]
    max_dlog = 0.0; mism = []
    for mode, group in [("clean", clean), ("laser", sim)]:
        for r in group:
            p = repo / r["file"]
            mine = predict([p], mode, cfg, model)[0]
            # repository pipeline: training-path waveform -> same model/temperature/threshold
            from train_reference import load_clip
            w = load_live_path(p) if mode == "laser" else load_clip(p, False)
            feat = extract_features(torch.as_tensor(w), "mel").unsqueeze(0)
            with torch.no_grad():
                ref_logits = model(feat)
            label, conf, rejected, _ = decide(ref_logits, cfg)
            if label != mine["label"]:
                mism.append((mode, r["file"], label, mine["label"]))
    ok = not mism
    print(json.dumps({"selftest": "pass" if ok else "FAIL", "clean_examples": len(clean), "laser_examples": len(sim),
                      "label_mismatches": mism}, indent=2))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", choices=["clean", "laser"], help="clean = file as recorded; laser = laser-channel recording")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--repo", default=os.environ.get("LMML_REPO", str(DEFAULT_REPO)))
    ap.add_argument("--json", default=None, help="also write the results to this JSON file")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    repo = Path(args.repo).expanduser().resolve()
    cfg = load_config()
    setup(repo, cfg)
    model = load_checkpoint(cfg)
    if args.selftest:
        sys.exit(0 if selftest(repo, cfg, model) else 1)
    if not args.mode or not args.files:
        fail(2, "usage: predict_package.py {clean|laser} FILE.wav [FILE.wav ...]   (or --selftest)")
    res = predict(args.files, args.mode, cfg, model)
    for r in res:
        print(f"{r['file']}\t{r['label']}\tconf={r['confidence']:.3f}\tunknown={r['rejected_as_unknown']}")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
