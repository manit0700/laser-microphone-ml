"""
export_model.py
===============
Package a trained model for deployment on the edge-AI computer, and measure how
fast it runs.

WHAT IT DOES
------------
1. Exports the trained network to TorchScript (models/<name>_scripted.pt).
   TorchScript is a self-contained, Python-optional format: the edge device can
   load and run it without our training code, which is what you want on the
   embedded board.
2. Benchmarks latency on CPU (a stand-in for the edge computer):
     - model-only forward pass (feature -> logits)
     - full pipeline per clip (preprocess -> feature -> model)
   so we know whether real-time recognition is realistic on the target and
   where the time goes.

RUN
---
    python scripts/export_model.py --model lstm
    python scripts/export_model.py --model cnn

LASER NOTE
----------
Latency here is measured on this machine's CPU. Re-run this same script ON the
edge board once it arrives to get the real numbers; the code doesn't change.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import (  # noqa: E402
    DEVICE, FEATURE_FOR_MODEL, MAX_AUDIO_SAMPLES, MODELS_DIR, MODEL_TYPE,
    SAMPLE_RATE, model_checkpoint,
)
from model import build_model  # noqa: E402
from features import extract_features  # noqa: E402
from preprocess import preprocess_waveform  # noqa: E402


def _example_feature(feature_kind):
    """Build one realistic feature tensor (batch of 1) to trace/benchmark with."""
    dummy_wave = torch.randn(MAX_AUDIO_SAMPLES) * 0.2
    clean = preprocess_waveform(dummy_wave)
    return extract_features(clean, feature_kind).unsqueeze(0)


def export(model_type):
    ckpt_path = model_checkpoint(model_type)
    if not Path(ckpt_path).exists():
        print(f"No trained model at {ckpt_path}. Train it first: "
              f"python src/train.py --model {model_type}")
        return None

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    loaded_type = checkpoint.get("model_type", model_type)
    feature = checkpoint.get("feature", FEATURE_FOR_MODEL[loaded_type])

    model = build_model(loaded_type).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    example = _example_feature(feature).to(DEVICE)
    with torch.no_grad():
        scripted = torch.jit.trace(model, example)

    out_path = MODELS_DIR / f"{Path(ckpt_path).stem}_scripted.pt"
    scripted.save(str(out_path))
    print(f"Exported TorchScript -> {out_path}")
    return model, feature, example


def benchmark(model, feature, example, runs=200):
    """Time model-only and full-pipeline inference; print mean/95th latency."""
    # Model-only forward pass.
    model.eval()
    with torch.no_grad():
        for _ in range(10):                      # warm-up
            model(example)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(example)
            times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times)
    print(f"\nModel-only forward ({runs} runs):")
    print(f"  mean {times.mean():.2f} ms | p95 {np.percentile(times, 95):.2f} ms")

    # Full pipeline: raw waveform -> preprocess -> feature -> model.
    from predict import infer
    wave = torch.randn(MAX_AUDIO_SAMPLES) * 0.2
    for _ in range(5):
        infer(wave)                              # warm-up (also caches model)
    pipe = []
    for _ in range(50):
        t0 = time.perf_counter()
        infer(wave)
        pipe.append((time.perf_counter() - t0) * 1000.0)
    pipe = np.array(pipe)
    print(f"Full pipeline per clip (50 runs):")
    print(f"  mean {pipe.mean():.2f} ms | p95 {np.percentile(pipe, 95):.2f} ms")
    print(f"\nContext: one clip covers {MAX_AUDIO_SAMPLES / SAMPLE_RATE:.2f} s of audio, "
          f"so real-time needs latency well under that. Device: {DEVICE}.")


def main(model_type):
    result = export(model_type)
    if result is None:
        return
    model, feature, example = result
    benchmark(model, feature, example)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export + benchmark a trained model.")
    parser.add_argument("--model", choices=["lstm", "cnn"], default=MODEL_TYPE)
    args = parser.parse_args()
    main(args.model)
