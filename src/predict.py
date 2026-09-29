"""
predict.py
==========
Runs the trained model on a SINGLE input clip and prints the JSON-style result
the frontend/integration team expects.

PIPELINE STAGE
--------------
one WAV file  ->  preprocess -> MFCC -> LSTM -> confidence check  ->  JSON output

OUTPUT CONTRACT  (what the frontend receives)
---------------------------------------------
Recognized:
    {"prediction": "3",       "confidence": 0.87, "status": "recognized"}
Low confidence (treated as not a digit / noise / silence):
    {"prediction": "unknown", "confidence": 0.42, "status": "low_confidence"}

The "unknown" decision happens here via CONFIDENCE_THRESHOLD (config.py): if the
model's top softmax probability is below the threshold, we report "unknown".
This is the "confidence check" box in the project pipeline diagram.

RUN
---
    python src/predict.py path/to/sample.wav

LASER NOTE
----------
This function accepts a file path. For a live laser/DAQ stream, import
`predict_waveform` and feed it a raw waveform tensor obtained via
preprocess.load_waveform_from_array(); the model + thresholding are identical.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from config import (
    CONFIDENCE_THRESHOLD,
    DEVICE,
    DIGIT_LABELS,
    FEATURE_FOR_MODEL,
    MODEL_TYPE,
    model_checkpoint,
)
from features import extract_features
from model import build_model, model_from_checkpoint
from preprocess import load_audio, preprocess_waveform
from utils import format_prediction

# Cache loaded models by type so repeated calls (e.g. a stream) don't reload weights.
# Each entry: {"model": nn.Module, "labels": [...], "feature": "mfcc"|"mel"}.
_cache: dict = {}


def _load(model_type: str = MODEL_TYPE) -> dict:
    """Load a trained model (once) and reuse it.

    Reads model_type/feature from the checkpoint so it always rebuilds the correct
    network (LSTM or CNN) and extracts the matching feature.
    """
    if model_type not in _cache:
        ckpt_path = model_checkpoint(model_type)
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(
                f"No trained model at {ckpt_path}. Run `python src/train.py "
                f"--model {model_type}` first."
            )
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        loaded_type = checkpoint.get("model_type", model_type)
        labels = checkpoint.get("labels", DIGIT_LABELS)
        # Build with the checkpoint's class count so an 'unknown'-class model
        # (11 outputs) loads correctly alongside plain 10-class models.
        model = model_from_checkpoint(checkpoint).to(DEVICE)
        model.eval()
        _cache[model_type] = {
            "model": model,
            "labels": labels,
            "feature": checkpoint.get("feature", FEATURE_FOR_MODEL[loaded_type]),
            # Temperature fitted on validation data after training (calibration.py);
            # 1.0 for older checkpoints = raw softmax, exactly as before.
            "temperature": float(checkpoint.get("temperature", 1.0)),
            # 'unknown' threshold tuned on validation data after training (None = use config).
            "threshold": checkpoint.get("threshold"),
            # Threshold for the LSTM+CNN ensemble, tuned by scripts/tune_threshold.py.
            "ensemble_threshold": checkpoint.get("ensemble_threshold"),
        }
    return _cache[model_type]


def _probs(clean_waveform: torch.Tensor, entry: dict) -> torch.Tensor:
    """Softmax probabilities (1, num_classes) for one loaded model on a clean clip."""
    feat = extract_features(clean_waveform, entry["feature"]).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return F.softmax(entry["model"](feat) / entry.get("temperature", 1.0), dim=1)


def _result_from_probs(probs: torch.Tensor, labels, threshold: float):
    """Turn a probability row into (JSON result, per-label probability dict)."""
    confidence, idx = probs.max(dim=1)
    label = labels[int(idx.item())]
    result = format_prediction(label, confidence.item(), threshold)
    probabilities = {labels[i]: float(probs[0, i]) for i in range(len(labels))}
    return result, probabilities


def _resolve_threshold(threshold, entries, ensemble: bool) -> float:
    """Explicit argument > threshold tuned on validation (stored in the checkpoint) > config."""
    if threshold is not None:
        return float(threshold)
    if ensemble:
        tuned = [e.get("ensemble_threshold") for e in entries]
        if tuned and all(t is not None for t in tuned):
            return float(sum(tuned) / len(tuned))
        tuned = [e.get("threshold") for e in entries]
        if tuned and all(t is not None for t in tuned):
            return float(sum(tuned) / len(tuned))
    else:
        if entries[0].get("threshold") is not None:
            return float(entries[0]["threshold"])
    return CONFIDENCE_THRESHOLD


def infer(waveform: torch.Tensor, threshold: float | None = None,
          model_type: str = MODEL_TYPE):
    """Core single-model inference on a raw waveform (shared by all entry points).

    Returns (result, probabilities). `model_type` selects "lstm" or "cnn".
    `threshold=None` uses the model's validation-tuned threshold if it has one.
    """
    entry = _load(model_type)
    clean = preprocess_waveform(waveform)               # normalize/trim/pad
    probs = _probs(clean, entry)
    return _result_from_probs(probs, entry["labels"], _resolve_threshold(threshold, [entry], False))


def infer_ensemble(waveform: torch.Tensor, threshold: float | None = None,
                   model_types=("lstm", "cnn")):
    """Ensemble inference: average the probabilities of several models.

    The LSTM (MFCC) and CNN (mel spectrogram) look at the sound differently and
    make DIFFERENT mistakes, so averaging their probabilities usually beats either
    one alone. We preprocess the audio ONCE, then extract each model's own feature
    from that same clean clip.

    When every model has a trained 'unknown' class, its probability is averaged
    too, so the ensemble can answer "unknown" directly (previously only the 10
    digit outputs were averaged and the reject class was thrown away).

    Returns (result, probabilities) in the same format as infer().
    """
    entries = [_load(mt) for mt in model_types]
    clean = preprocess_waveform(waveform)
    out_labels = list(DIGIT_LABELS)
    if all("unknown" in e["labels"] for e in entries):
        out_labels.append("unknown")
    probs_sum = None
    for entry in entries:
        p = _probs(clean, entry)
        # Average by label NAME, so 10-class and 11-class checkpoints still combine safely.
        aligned = torch.zeros((1, len(out_labels)), dtype=p.dtype, device=p.device)
        for src_idx, label in enumerate(entry["labels"]):
            if label in out_labels:
                aligned[0, out_labels.index(label)] = p[0, src_idx]
        probs_sum = aligned if probs_sum is None else probs_sum + aligned
    probs = probs_sum / len(entries)                    # simple average
    return _result_from_probs(probs, out_labels, _resolve_threshold(threshold, entries, True))


def predict_waveform(waveform: torch.Tensor, threshold: float | None = None) -> dict:
    """Predict a digit from a raw (un-preprocessed) waveform tensor.

    Use this for live laser/stream input. Returns the JSON-style dict.
    """
    result, _ = infer(waveform, threshold)
    return result


def predict_file(path: str | Path, threshold: float | None = None,
                 model: str = MODEL_TYPE) -> dict:
    """Predict a digit from a WAV file path. Returns the JSON-style dict.

    `model` may be "lstm", "cnn", or "ensemble" (LSTM+CNN averaged).
    """
    waveform = load_audio(path)
    if model == "ensemble":
        result, _ = infer_ensemble(waveform, threshold)
    else:
        result, _ = infer(waveform, threshold, model_type=model)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Recognize a spoken digit in a WAV file.")
    parser.add_argument("wav", help="path to the .wav file")
    parser.add_argument(
        "--model", choices=["lstm", "cnn", "ensemble"], default=MODEL_TYPE,
        help="which model to use ('ensemble' averages LSTM + CNN)",
    )
    args = parser.parse_args()

    if not Path(args.wav).exists():
        print(json.dumps({"error": f"file not found: {args.wav}"}))
        sys.exit(1)

    result = predict_file(args.wav, model=args.model)
    # Print as JSON so the frontend/integration code can parse it directly.
    print(json.dumps(result, indent=2))

    # Save a record of this recognition (row in predictions_log.csv + the clip).
    from utils import log_prediction
    paths = log_prediction(result, samples=load_audio(args.wav).numpy(), source=args.model)
    print(f"Logged to {paths['log_file']}")


if __name__ == "__main__":
    main()
