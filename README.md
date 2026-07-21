# Laser Microphone - Spoken Digit Recognition (ML Pipeline)

**Team Eclipse - Senior Design**

This repository contains the **machine-learning / data pipeline** for the Laser
Microphone project. The goal of the proof-of-concept is to recognize spoken
digits **0-9** and output **"unknown"** when the input is unclear, noisy, or not
a digit.

---

## Quick Start — Run the Live Demo

Recognizes spoken digits **0–9** live from the microphone in a desktop dashboard.
Works on **Windows** and **macOS**. The trained model is included in the repo, so
there is nothing to train first.

**1. Get the code and install dependencies**
```bash
git clone https://github.com/manit0700/laser-microphone-ml.git
cd laser-microphone-ml
pip install -r requirements.txt
pip install PyQt5 pyqtgraph sounddevice
```

**2. Run the dashboard**
```bash
python signal_dashboard.py
```

**3. Allow microphone access** (do this once, or the app hears only silence and
guesses wrong):
- **Windows:** Settings → Privacy & security → Microphone → turn ON
  “Microphone access” and “Let desktop apps access your microphone.”
- **macOS:** System Settings → Privacy & Security → Microphone → enable your
  terminal (Terminal / iTerm / VS Code).

**4. Demo it**
- Click the red **Record** button, allow the mic prompt.
- Say a digit clearly, ~15 cm from the mic (moderate volume — not a whisper, not
  a shout). The **Prediction** + **Confidence** boxes update and hold steady.
- The **Enhance Audio** toggle (denoise + auto-gain) should stay ON.
- **Stop** ends capture. **F11** = fullscreen.

The dashboard uses the **ensemble** (LSTM + CNN) model by default and works for
any speaker. If recognition seems off, it is almost always the mic: check step 3
and the input volume (watch the oscilloscope — you want a clear burst when you
speak, not a flat line or a clipped/squared-off wave).

---

## 1. What is the Laser Microphone project?

Instead of a normal microphone, the final hardware uses a **laser / laser
vibrometry** setup: a laser measures tiny vibrations on a surface caused by
speech, and a detector + data-acquisition (DAQ) hardware turns those vibrations
into a signal. That signal is then processed by **this** software to figure out
which digit was spoken.

We **do not have real laser data yet**, so we are building and testing the entire
ML pipeline using **public spoken-digit audio** (regular microphone recordings).
The code is written so that, later, laser data can replace the audio data with
minimal changes.

## 2. Why start with public audio data?

- It lets us build, debug, and validate the full pipeline **today**, before the
  hardware/data is ready.
- Spoken digits are a small, well-understood problem - good for a proof of concept.
- Recommended datasets:
  - **Free Spoken Digit Dataset (FSDD)** - small, 8 kHz, easy (we include a downloader).
  - **Google Speech Commands** (digit subset) - larger, 16 kHz.

When the laser data arrives, we **swap the input**, not the whole system.

## 3. How the ML pipeline works

```
raw signal / audio
      |
      v
endpoint detection / silence trimming   (preprocess.py)
      |
      v
preprocessing / normalization           (preprocess.py)
      |
      v
MFCC feature extraction                 (features.py)   <- main feature
      |
      v
LSTM neural network                     (model.py)
      |
      v
confidence check (threshold)            (predict.py)
      |
      v
output: digit 0-9   OR   "unknown"
```

**Design decisions (agreed by the team):**
- **MFCCs** are the main features; **mel spectrograms** are a backup/comparison
  feature; plain spectrograms are mostly for visualization.
- **LSTM** is the main model because MFCC features are **time sequences**. A CNN
  on mel spectrograms is a possible backup/comparison model later.
- We **avoid** DeepSpeech (discontinued) and Kaldi (too heavy). This is **digit
  classification, not full speech-to-text**.
- **"unknown"** is decided by a **confidence threshold** at prediction time, not
  a separate trained class (you can add an unknown class later - see dataset.py).

## 4. Project structure

```
laser_microphone_ml/
  README.md              <- you are here
  requirements.txt       <- Python dependencies
  data/
    raw/                 <- put WAV files here (downloader fills this)
    processed/           <- optional cached/processed data
    metadata/            <- optional label CSVs
  src/
    config.py            <- ALL settings (sample rate, MFCC, model, paths...)
    utils.py             <- helpers (seeds, folders, JSON, output formatting)
    preprocess.py        <- load, mono, resample, normalize, trim, pad
    features.py          <- MFCC (main) + mel spectrogram (backup)
    dataset.py           <- PyTorch Dataset -> (MFCC, label) pairs
    model.py             <- the LSTM classifier
    train.py             <- train + save best model + history
    evaluate.py          <- accuracy, per-class, confusion matrix, report
    predict.py           <- single-clip prediction -> JSON output
    app.py               <- browser UI to test with your microphone (Gradio)
  notebooks/
    exploration.ipynb    <- visualize waveform / MFCC / mel spectrogram
  scripts/
    download_fsdd.py        <- fetch the Free Spoken Digit Dataset into data/raw/
    record_wav.py           <- record digits from your microphone -> WAV
    wav_from_csv.py         <- convert laser/DAQ CSV (voltage/time) -> WAV
    array_to_wav.py         <- convert any signal array (.npy/text) -> WAV
    make_synthetic_wavs.py  <- generate fake labeled WAVs to test the pipeline
  models/                <- trained model checkpoints
  results/
    plots/               <- confusion matrix, training curves
    reports/             <- evaluation reports (JSON + text)
```

## 5. How to run it

```bash
# 0. (recommended) create a virtual environment
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 1. install dependencies
pip install -r requirements.txt

# 2. get some data (Free Spoken Digit Dataset -> data/raw/)
python scripts/download_fsdd.py

# 3. train the model  (saves models/best_model.pt)
python src/train.py

# 4. evaluate         (saves confusion matrix + report to results/)
python src/evaluate.py

# 5. predict one clip (prints JSON)
python src/predict.py data/raw/7_jackson_0.wav
```

> Tip: run the scripts from the project root (`laser_microphone_ml/`) so the
> relative paths in `config.py` resolve correctly.

### Outputs you should see
- `models/best_model.pt` - trained model.
- `results/plots/confusion_matrix.png` - confusion matrix.
- `results/reports/evaluation_report.{json,txt}` - accuracy + per-class report.
- A JSON object printed by `predict.py`, e.g.:

```json
{ "prediction": "3", "confidence": 0.87, "status": "recognized" }
```
or
```json
{ "prediction": "unknown", "confidence": 0.42, "status": "low_confidence" }
```

### Train on Kaggle (free GPU, ~1-2 min instead of ~20 min)

Local training runs on your Mac's CPU (~20 min for 40 epochs). To train much
faster on a free GPU, use `notebooks/kaggle_train.ipynb`:

1. On kaggle.com: **Datasets -> New Dataset** and upload this project (at least
   the `src/` folder). Name it e.g. `laser-microphone-ml`.
2. **New Notebook** -> *Add Input* (your dataset) -> *Settings*: Accelerator =
   **GPU**, Internet = **On**.
3. Upload/open `notebooks/kaggle_train.ipynb` and **Run All**.
4. Download the resulting `models/best_model.pt` from the notebook's **Output**
   tab and drop it into your local `models/` folder.

The notebook imports the **same** `src/` code (no logic duplicated). It works
because `config.py` honors two environment variables so inputs/outputs can point
at Kaggle's folders:

```bash
LMML_RAW_DIR=/path/to/wavs      # where input WAV files live
LMML_OUTPUT_DIR=/writable/dir   # where models/ and results/ are written
```

These also help on Colab, lab servers, or any read-only-input environment.

### Test it live with your microphone (web UI)

A small browser UI lets you speak a digit into your Mac mic and see the model's
guess, confidence, per-digit probabilities, and the MFCC it used.

```bash
pip install gradio          # one-time
python src/train.py         # need a trained model first
python src/app.py           # opens http://127.0.0.1:7860
```

Open the printed URL, allow microphone access, click **Record**, say a digit, then
**Recognize digit**. Drag the threshold slider to see when it flips to "unknown".
You can also upload a WAV instead of recording.

### Making your own WAV files

You don't have to rely on the downloaded dataset — there are four ways to create
WAV files in `data/raw/` (all save FSDD-style `<digit>_<speaker>_<index>.wav`
names so labels are detected automatically):

```bash
# A) Record from your microphone (needs: pip install sounddevice)
python scripts/record_wav.py --all --speaker manit --takes 3
python scripts/record_wav.py --digit 7 --speaker alex --takes 5

# B) Convert laser/DAQ CSV (voltage vs. time) -> WAV
python scripts/wav_from_csv.py capture.csv --out data/raw/7_laser_0.wav
#   (auto-detects sample rate from a 'time' column; override with --sample-rate)

# C) Convert any signal array (.npy / text) -> WAV  (e.g. a live DAQ buffer)
python scripts/array_to_wav.py buffer.npy --sample-rate 50000 --out data/raw/3_laser_0.wav

# D) Generate synthetic labeled WAVs to test the pipeline with no data/internet
python scripts/make_synthetic_wavs.py --per-digit 18
```

All four share one core helper, `save_wav()` in `src/preprocess.py`, so the WAV
format stays consistent. For a **live laser stream** you usually skip WAV entirely
and call `predict.predict_waveform(buffer)` directly.

## 6. Explaining the pipeline to teammates (non-ML)

- **Preprocessing** = "clean up the recording": make it mono, set a fixed sample
  rate, make loudness consistent, and cut the silence before/after the digit so
  the model only sees the spoken part. Every clip ends up the same length.
- **MFCC features** = "describe the sound in a few numbers per moment". Instead
  of thousands of raw samples, each short slice of time becomes ~13 numbers that
  capture the *character* of the sound. The result is a short **sequence** of
  feature vectors.
- **LSTM** = "a network that reads sequences and remembers context". It reads the
  MFCC sequence step by step and outputs a score for each digit.
- **Confidence check** = "how sure are we?". We convert the scores to
  probabilities; if the top one is below a threshold, we say **"unknown"** instead
  of guessing.
- **Output** = a small JSON object the frontend/integration team can use directly.

Everything tunable lives in `src/config.py`. Each file has a header comment
describing its input, output, and role.

## 7. Adapting to real laser-microphone data (from the DAQ)

The pipeline is modular so the **laser/hardware team can plug in their data**.
Search the code for the comment marker `LASER NOTE` to find the exact spots.

Depending on how the DAQ delivers data:

1. **WAV files** (easiest): export laser captures as WAV named like FSDD
   (`<digit>_<speaker>_<index>.wav`) into `data/raw/`. No code changes - just set
   `SAMPLE_RATE` in `config.py` to match the laser data, then run `train.py`.

2. **CSV (voltage vs. time)**: read the voltage column and call
   `preprocess.load_waveform_from_array(values, sample_rate=<DAQ_rate>)`. Extend
   the single hook `SpokenDigitDataset._load_one_waveform` in `dataset.py` to
   branch on the `.csv` suffix. Everything downstream is unchanged.

3. **Live stream**: collect a buffer of samples, build a waveform tensor, and call
   `predict.predict_waveform(waveform)`. It runs the same preprocess -> MFCC ->
   LSTM -> confidence path and returns the JSON result.

Other likely adjustments for laser signals (all isolated and documented):
- `SAMPLE_RATE`, `N_FFT`, `HOP_LENGTH`, `N_MELS` in `config.py` (window sizes).
- `preprocess.reduce_noise()` - currently a pass-through placeholder; add
  band-pass filtering / denoising tuned to the laser signal here.
- The energy-based `endpoint_detection()` thresholds in `config.py`.

The **most important point for the team**: the model, training, evaluation, and
output format **do not change** when we move to laser data - only the front of
the pipeline (how we obtain the raw waveform) does.
```
