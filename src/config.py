"""
config.py
=========
Central configuration for the Laser Microphone digit-recognition ML pipeline.

WHY THIS FILE EXISTS
--------------------
Every other file (dataset, preprocessing, features, model, training, etc.) imports
its settings from here. If you want to change the sample rate, number of MFCCs,
model size, or where files are saved, change it HERE in ONE place. Do not scatter
"magic numbers" across the codebase.

HOW IT CONNECTS TO THE LASER MICROPHONE SYSTEM
----------------------------------------------
Right now we use public spoken-digit audio (WAV files) to develop the pipeline.
Later, laser-vibrometry data from the DAQ (data acquisition hardware) will arrive
as WAV, CSV (voltage/time), or a live stream. When that happens you mostly change
SAMPLE_RATE and the data-loading path here, and the rest of the pipeline keeps
working. Search for "LASER" comments throughout the project to find the spots that
will need attention.
"""

from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------------------------
# We compute paths relative to this file so the project works no matter where
# it is cloned (laptop, lab PC, server). PROJECT_ROOT = the laser_microphone_ml/ folder.
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths can be overridden with environment variables. This is what lets the SAME
# code run on a laptop AND on Kaggle/Colab, where inputs are read-only and outputs
# must be written to a separate working folder:
#   LMML_RAW_DIR     -> where the input WAV files live (e.g. a Kaggle dataset path)
#   LMML_OUTPUT_DIR  -> where models/ and results/ are written (e.g. /kaggle/working)
# If a variable is unset, we fall back to the normal in-project folders.
_OUTPUT_ROOT = Path(os.environ.get("LMML_OUTPUT_DIR", PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = Path(os.environ.get("LMML_RAW_DIR", DATA_DIR / "raw"))  # input WAVs
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # optional cached/processed data
METADATA_DIR = DATA_DIR / "metadata"       # optional CSV label files

MODELS_DIR = _OUTPUT_ROOT / "models"       # trained model checkpoints
RESULTS_DIR = _OUTPUT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"          # confusion matrix, training curves
REPORTS_DIR = RESULTS_DIR / "reports"      # JSON / text evaluation reports

# Default filename for the best trained model.
BEST_MODEL_PATH = MODELS_DIR / "best_model.pt"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.json"

# Where recognized numbers get logged. Every prediction appends a row to the CSV,
# and (optionally) the audio clip is saved under CAPTURES_DIR. This builds a record
# of what was spoken and a growing set of REAL captures for future training.
PREDICTIONS_LOG = RESULTS_DIR / "predictions_log.csv"
CAPTURES_DIR = RESULTS_DIR / "captures"

# ---------------------------------------------------------------------------
# 2. AUDIO / SIGNAL SETTINGS
# ---------------------------------------------------------------------------
# Sample rate in Hz. The Free Spoken Digit Dataset (FSDD) is 8000 Hz.
# Google Speech Commands is 16000 Hz. We resample everything to this value so
# the model always sees a consistent input.
# LASER NOTE: laser/DAQ data may be sampled at a very different rate. Set this to
# match (or resample to) whatever rate makes the digit content clear.
#
# 8000 keeps only < 4 kHz: fine for vowels, but it drops most of the "s" in "six"
# (measured: ~11% of that word's energy is above 4 kHz) and the quiet "th"/"f"/"v"
# consonants. 16000 keeps up to 8 kHz. Choose with LMML_SAMPLE_RATE=16000 when
# TRAINING; train.py records the rate in each checkpoint and in
# models/model_meta.json, and this file picks it up from there automatically, so
# the dashboard always runs at the rate its models were trained at.
def _default_sample_rate() -> int:
    env = os.environ.get("LMML_SAMPLE_RATE")
    if env:
        return int(env)
    try:
        import json as _json
        meta = MODELS_DIR / "model_meta.json"
        if meta.exists():
            return int(_json.loads(meta.read_text())["sample_rate"])
    except Exception:  # noqa: BLE001 - unreadable meta: fall back to the default
        pass
    return 8000


SAMPLE_RATE = _default_sample_rate()
if SAMPLE_RATE % 8000:
    raise ValueError(f"SAMPLE_RATE must be a multiple of 8000 (8000, 16000, ...), got {SAMPLE_RATE}")
# Window/hop sizes below are defined for 8 kHz and scaled with the rate, so every
# rate gives the same ~32 ms windows, 16 ms hops and 63 frames per second -- the
# model's input shape does not change.
_SR_SCALE = SAMPLE_RATE // 8000

# New recordings are always saved at this rate (or higher), whatever rate the
# model uses, so no capture ever throws away its high frequencies.
RECORD_SAMPLE_RATE = max(16000, SAMPLE_RATE)

# All clips are padded or truncated to this fixed length (in seconds) so every
# sample produces a tensor of the same size. Spoken digits are short (< 1 s).
MAX_AUDIO_SECONDS = 1.0
MAX_AUDIO_SAMPLES = int(SAMPLE_RATE * MAX_AUDIO_SECONDS)

# ---------------------------------------------------------------------------
# 2b. VOICE ENHANCEMENT (band-pass filter + pre-emphasis)
# ---------------------------------------------------------------------------
# Applied in preprocess.reduce_noise() before feature extraction. Cleans the
# signal so the model sees speech, not hum/hiss. Also helps future laser data.
ENABLE_FILTER = False

# Live audio enhancement (DC removal + spectral denoise + auto-gain), applied to
# microphone/laser captures before feature extraction (see enhance.py). Unlike
# the band-pass filter, this makes noisy/quiet live audio look MORE like the
# clean training clips, so it helps live recognition without a train/inference
# mismatch. On already-clean clips it is near a no-op. Default on for live use.
ENABLE_ENHANCE = True

# Band-pass keeps only the speech band and removes low-frequency rumble/DC and
# high-frequency hiss. Butterworth = flat in-band. High cutoff must stay below
# the Nyquist frequency (SAMPLE_RATE / 2 = 4000 Hz at 8 kHz).
BANDPASS_LOW_HZ = 200
BANDPASS_HIGH_HZ = int(SAMPLE_RATE / 2 * 0.95)   # just under Nyquist (3800 Hz at 8 kHz)
BANDPASS_ORDER = 6               # 6-pole, matches the team's DSP filter plan

# Pre-emphasis boosts high frequencies to sharpen consonants (y[n] = x[n] - a*x[n-1]).
PREEMPHASIS_COEF = 0.97

# ---------------------------------------------------------------------------
# 3. ENDPOINT DETECTION / SILENCE TRIMMING
# ---------------------------------------------------------------------------
# Energy-based endpoint detection keeps only the part of the clip that is
# "loud enough" to be speech. See preprocess.py for the full explanation.
# - FRAME_LENGTH/FRAME_HOP: window used to measure energy over time.
# - ENERGY_THRESHOLD_RATIO: a frame is "speech" if its energy is above this
#   fraction of the clip's peak frame energy. Higher = more aggressive trimming.
FRAME_LENGTH = 256 * _SR_SCALE
FRAME_HOP = 128 * _SR_SCALE
ENERGY_THRESHOLD_RATIO = 0.05

# ---------------------------------------------------------------------------
# 4. MFCC FEATURE SETTINGS  (main feature extraction method)
# ---------------------------------------------------------------------------
# MFCC = Mel-Frequency Cepstral Coefficients. These compress an audio frame into
# a small set of numbers that capture the *shape* of the sound (timbre), which is
# what distinguishes spoken digits. See features.py for details.
N_MFCC = 13          # number of MFCC coefficients per time frame (13 is standard)
N_FFT = 256 * _SR_SCALE       # FFT window (32 ms at any rate)
HOP_LENGTH = 128 * _SR_SCALE  # step between frames (16 ms); controls the time resolution
N_MELS = 40          # mel filterbank size used before the cepstral step

# Mel spectrogram settings (backup / comparison feature, and for visualization).
MEL_N_MELS = 40

# ---------------------------------------------------------------------------
# 4b. MODEL SELECTION
# ---------------------------------------------------------------------------
# Which network to train/use by default. Two options:
#   "lstm" -> DigitLSTM on MFCC sequences   (main proof-of-concept model)
#   "cnn"  -> DigitCNN on mel spectrograms  (backup/comparison model)
# Override at the command line: `python src/train.py --model cnn`.
MODEL_TYPE = "lstm"

# Each model needs a different feature. This mapping is the single source of truth
# so train/evaluate/predict always pair the right feature with the right model.
FEATURE_FOR_MODEL = {"lstm": "mfcc", "cnn": "mel"}


def model_checkpoint(model_type: str = MODEL_TYPE):
    """Return the checkpoint path for a given model type.

    LSTM keeps the original 'best_model.pt' name (backward compatible); other
    models get their own file so the two never overwrite each other and can be
    compared side by side.
    """
    if model_type == "lstm":
        return BEST_MODEL_PATH
    return MODELS_DIR / f"best_model_{model_type}.pt"


# ---------------------------------------------------------------------------
# 5. MODEL (LSTM) HYPERPARAMETERS
# ---------------------------------------------------------------------------
# Kept intentionally small so it trains on a normal laptop CPU in minutes.
# input_size is set automatically to N_MFCC in the model, so it is not repeated here.
LSTM_HIDDEN_SIZE = 128   # neurons in each LSTM layer
LSTM_NUM_LAYERS = 2      # stacked LSTM layers
LSTM_BIDIRECTIONAL = True
# Dropout probability used by BOTH models (LSTM between/after layers, CNN before the
# classifier). Override per run with `train.py --dropout 0.25` or LMML_DROPOUT=0.25.
# Sensible range 0.2-0.5: lower = less regularization (fits faster, may overfit),
# higher = more regularization (more robust, may underfit).
DROPOUT = float(os.environ.get("LMML_DROPOUT", 0.3))

# ---------------------------------------------------------------------------
# 6. TRAINING HYPERPARAMETERS  (initial values for Sprint 2)
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 40
WEIGHT_DECAY = 1e-5      # small L2 regularization
SEED = 42                # random seed for reproducible splits/results

# Early stopping: stop training when validation LOSS has not improved by at least
# EARLY_STOP_MIN_DELTA for EARLY_STOP_PATIENCE epochs in a row, and keep the checkpoint
# with the lowest validation loss. Set patience to 0 to disable. Override with
# `train.py --patience N` or LMML_PATIENCE=N.
EARLY_STOP_PATIENCE = int(os.environ.get("LMML_PATIENCE", 6))
EARLY_STOP_MIN_DELTA = 1e-3

# How train/val/test are formed. "grouped" (default) removes duplicate recordings
# and puts every speaker in exactly ONE split, so the test set measures accuracy on
# unseen speakers (see src/splits.py). "random" is the old clip-by-clip split, kept
# only for reproducing earlier numbers. Override with LMML_SPLIT or train.py --split.
SPLIT_METHOD = os.environ.get("LMML_SPLIT", "grouped")

# Learning-rate schedule: "plateau" halves the LR when val loss stalls for 2 epochs
# (ReduceLROnPlateau), "cosine" decays it smoothly to ~0 over the run, "none" keeps
# LEARNING_RATE fixed (the old behaviour). Override with train.py --scheduler.
LR_SCHEDULER = os.environ.get("LMML_SCHEDULER", "plateau")

# SpecAugment: randomly mask frequency bands and time steps of the training features
# (never at eval/inference). Helps with uneven frequency response, e.g. the laser.
# Off by default so old runs reproduce; enable with train.py --specaugment.
SPECAUGMENT = os.environ.get("LMML_SPECAUGMENT", "0") == "1"

# LSTM features: plain MFCC (13 per frame) or MFCC + delta + delta-delta (39).
# Enable with train.py --deltas or LMML_DELTAS=1. The feature a model was trained
# with is stored in its checkpoint, so prediction always matches automatically.
MFCC_DELTAS = os.environ.get("LMML_DELTAS", "0") == "1"

# Label smoothing: train toward 0.9/0.01... instead of 1/0 targets so the model is
# less over-confident. 0 = off (default); 0.1 is typical. train.py --label-smoothing.
LABEL_SMOOTHING = float(os.environ.get("LMML_LABEL_SMOOTHING", 0.0))

# Dataset split ratios (must sum to 1.0).
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# ---------------------------------------------------------------------------
# 7. CLASS LABELS
# ---------------------------------------------------------------------------
# The model is trained to output one of these 10 digit classes.
DIGIT_LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
NUM_CLASSES = len(DIGIT_LABELS)

# "unknown" is NOT a trained class by default. Instead we decide "unknown" at
# prediction time using the confidence threshold below (see predict.py). This
# matches the pipeline diagram: ... -> confidence check -> digit or unknown.
UNKNOWN_LABEL = "unknown"

# If the model's top softmax probability is below this value, we report "unknown".
# Tune this after looking at validation confidences. 0.60 is a reasonable start.
CONFIDENCE_THRESHOLD = 0.60

# ---------------------------------------------------------------------------
# 8. RUNTIME
# ---------------------------------------------------------------------------
def _select_device() -> "torch.device":
    """Pick GPU only if it can ACTUALLY run a tensor op, else fall back to CPU.

    `torch.cuda.is_available()` returning True is not enough: on some hosted
    images (e.g. Kaggle) the installed PyTorch build has no compiled kernels for
    the specific GPU that got assigned, which crashes with
    "CUDA error: no kernel image is available for execution on the device".
    We do a tiny real GPU op to confirm it works before committing to CUDA.
    """
    if torch.cuda.is_available():
        try:
            _ = (torch.zeros(8, device="cuda") + 1).sum().item()  # real GPU work
            torch.cuda.synchronize()
            return torch.device("cuda")
        except Exception as e:  # noqa: BLE001 - any CUDA failure -> use CPU
            print(f"[config] GPU present but unusable ({type(e).__name__}: {e}). "
                  "Falling back to CPU so the run still completes.")
    return torch.device("cpu")


DEVICE = _select_device()

# How many usable GPUs (only meaningful when DEVICE is cuda). train.py uses this
# to spread the model across both cards ("2x GPU") via DataParallel when > 1.
GPU_COUNT = torch.cuda.device_count() if DEVICE.type == "cuda" else 0

# Number of background workers for the DataLoader. 0 is safest/most portable.
NUM_WORKERS = int(os.environ.get("LMML_NUM_WORKERS", 0))
