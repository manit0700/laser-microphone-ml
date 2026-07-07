"""
utils.py
========
Small helper functions used across the pipeline. Keeping them here avoids copy-
pasting the same boilerplate into every script.

INPUT/OUTPUT
------------
These are plain utilities (make folders, save JSON, set seeds, format the
prediction dictionary). They do not touch audio directly.
"""

import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from config import (
    CAPTURES_DIR,
    CONFIDENCE_THRESHOLD,
    DATA_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    PREDICTIONS_LOG,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    SAMPLE_RATE,
    UNKNOWN_LABEL,
)


def set_seed(seed: int) -> None:
    """Make results reproducible by fixing all random number generators.

    Call this once at the start of training/evaluation so the same data split
    and initialization happen every run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    """Create the output folders we write to, if they do not already exist.

    Safe to call repeatedly. Run this before saving models/plots/reports.

    We only *need* to create the OUTPUT folders (models/results). The input
    folders (data/raw) may live on a read-only filesystem in hosted environments
    like Kaggle/Colab, so we try them but ignore read-only errors instead of
    crashing. (On Kaggle the WAV input is a read-only dataset mount.)
    """
    # Output dirs must be writable; let a real failure here surface.
    for d in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR, REPORTS_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Input/optional dirs: nice to have locally, but may be read-only when hosted.
    for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # read-only mount (e.g. Kaggle input) -- inputs already exist


def save_json(data: Dict[str, Any], path: os.PathLike) -> None:
    """Write a dictionary to a JSON file (pretty-printed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: os.PathLike) -> Dict[str, Any]:
    """Read a JSON file into a dictionary."""
    with open(path, "r") as f:
        return json.load(f)


def format_prediction(
    label: str,
    confidence: float,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """Build the JSON-style result that the frontend/integration team expects.

    This is the single source of truth for the output contract. Everyone
    (predict.py, a future API, the live laser stream) should call this so the
    output format never drifts.

    Returns one of:
        {"prediction": "3",       "confidence": 0.87, "status": "recognized"}
        {"prediction": "unknown", "confidence": 0.42, "status": "low_confidence"}
    """
    confidence = float(confidence)
    if confidence < threshold:
        return {
            "prediction": UNKNOWN_LABEL,
            "confidence": round(confidence, 4),
            "status": "low_confidence",
        }
    return {
        "prediction": str(label),
        "confidence": round(confidence, 4),
        "status": "recognized",
    }


def log_prediction(
    result: Dict[str, Any],
    samples: Optional[np.ndarray] = None,
    sample_rate: int = SAMPLE_RATE,
    source: str = "cli",
    save_audio: bool = True,
) -> Dict[str, str]:
    """Save a record of one spoken number: append to a CSV log and store the clip.

    Builds two things over time:
      1. results/predictions_log.csv  -- one row per prediction:
             timestamp, source, prediction, confidence, status, audio_file
      2. results/captures/<digit>_<source>_<timestamp>.wav  -- the actual audio,
         so you accumulate REAL recordings (useful later for training on your own
         voices / laser captures).

    Args:
        result      : the dict from format_prediction (prediction/confidence/status).
        samples     : optional 1-D audio to save (numpy or list). If None, no clip.
        sample_rate : sample rate of `samples`.
        source      : where it came from ("cli", "ui", "live", "laser", ...).
        save_audio  : set False to log the row only, without storing the clip.

    Returns a small dict with the log/audio paths (handy for showing in the UI).
    """
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms precision

    audio_path = ""
    if save_audio and samples is not None:
        # Imported here to avoid a circular import (preprocess imports config only).
        from preprocess import save_wav

        Path(CAPTURES_DIR).mkdir(parents=True, exist_ok=True)
        fname = f"{result['prediction']}_{source}_{timestamp}.wav"
        audio_path = str(Path(CAPTURES_DIR) / fname)
        save_wav(samples, audio_path, sample_rate)

    # Append a row to the CSV, writing the header the first time.
    log_path = Path(PREDICTIONS_LOG)
    new_file = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(
                ["timestamp", "source", "prediction", "confidence", "status", "audio_file"]
            )
        writer.writerow([
            timestamp,
            source,
            result["prediction"],
            result["confidence"],
            result["status"],
            audio_path,
        ])

    return {"log_file": str(log_path), "audio_file": audio_path}
