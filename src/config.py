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
SAMPLE_RATE = 8000

# All clips are padded or truncated to this fixed length (in seconds) so every
# sample produces a tensor of the same size. Spoken digits are short (< 1 s).
MAX_AUDIO_SECONDS = 1.0
MAX_AUDIO_SAMPLES = int(SAMPLE_RATE * MAX_AUDIO_SECONDS)

# ---------------------------------------------------------------------------
# 3. ENDPOINT DETECTION / SILENCE TRIMMING
# ---------------------------------------------------------------------------
# Energy-based endpoint detection keeps only the part of the clip that is
# "loud enough" to be speech. See preprocess.py for the full explanation.
# - FRAME_LENGTH/FRAME_HOP: window used to measure energy over time.
# - ENERGY_THRESHOLD_RATIO: a frame is "speech" if its energy is above this
#   fraction of the clip's peak frame energy. Higher = more aggressive trimming.
FRAME_LENGTH = 256
FRAME_HOP = 128
ENERGY_THRESHOLD_RATIO = 0.05

# ---------------------------------------------------------------------------
# 4. MFCC FEATURE SETTINGS  (main feature extraction method)
# ---------------------------------------------------------------------------
# MFCC = Mel-Frequency Cepstral Coefficients. These compress an audio frame into
# a small set of numbers that capture the *shape* of the sound (timbre), which is
# what distinguishes spoken digits. See features.py for details.
N_MFCC = 13          # number of MFCC coefficients per time frame (13 is standard)
N_FFT = 256          # FFT window size used inside MFCC
HOP_LENGTH = 128     # step between frames; controls the time resolution
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
DROPOUT = 0.3            # regularization to reduce overfitting

# ---------------------------------------------------------------------------
# 6. TRAINING HYPERPARAMETERS  (initial values for Sprint 2)
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 40
WEIGHT_DECAY = 1e-5      # small L2 regularization
SEED = 42                # random seed for reproducible splits/results

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
NUM_WORKERS = 0
