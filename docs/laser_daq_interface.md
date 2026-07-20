# Laser / DAQ Data Interface Spec

**Team Eclipse — Laser Microphone — CSE 4316/4317**
Owner: ML/data (Manit). Audience: hardware/DAQ team + frontend.
Status: Sprint 3 draft. Hardware connects in **Sprint 4** — this locks the seam
so integration is a plug-in, not a rewrite.

---

## 1. Purpose

Define **exactly** how a captured signal reaches the ML pipeline, so that when
the laser + ADC + DAQ are connected in Sprint 4, the ML side needs **no model
changes** — only pointing at the new source. Everything below is already
supported in code today (tested with mic + file replay); the hardware team just
has to produce data in one of these forms.

---

## 2. Accepted input forms

The pipeline accepts a signal in any of three forms. All converge to the same
internal representation: a **1-D mono float waveform resampled to the project
sample rate**.

| Form | When | Reader |
|------|------|--------|
| **WAV file** | batch captures, saved clips | `preprocess.load_audio()` |
| **CSV (time, voltage)** | DAQ export, laser voltage traces | `signal_backend._read_signal_file()` |
| **Live stream (array chunks)** | real-time mic / DAQ | `preprocess.load_waveform_from_array()` |

### 2a. WAV
- Mono or stereo (stereo is averaged to mono).
- Any sample rate (auto-resampled). 16-bit PCM preferred.
- Filename for **labeled training** clips: FSDD style `‹digit›_‹speaker›_‹index›.wav`
  e.g. `7_manit_3.wav`. Unlabeled capture files can be named anything.

### 2b. CSV (laser/DAQ)
- Header row required. Columns (case-insensitive):
  - `time` — seconds (optional but recommended; used to derive sample rate as `1/median(dt)`)
  - `voltage` — the signal. If no `voltage` column, the **last numeric column** is used.
- One sample per row. Example:
  ```
  time,voltage
  0.00000,0.0123
  0.00002,0.0157
  0.00004,0.0201
  ```
- If `time` is absent, the pipeline falls back to the project sample rate
  (`config.SAMPLE_RATE`), so **include `time`** whenever the DAQ rate differs.

### 2c. Live stream
- Push chunks of `float32` samples (mono) as they arrive; the backend keeps a
  rolling buffer and classifies the most recent window continuously.
- This is exactly how the desktop dashboard's mic path works today
  (`signal_backend._MicSource`); a DAQ stream fills the same buffer.

---

## 3. Sample rate

- Project internal rate: **`config.SAMPLE_RATE = 8000 Hz`** (FSDD proof-of-concept).
- The DAQ may sample far higher (e.g. 44.1 kHz, 50 kHz). **That's fine** — every
  reader resamples to 8000 Hz automatically (`preprocess.resample_if_needed`).
- **Action for Sprint 4:** once we know the laser's real signal band, we may
  raise `SAMPLE_RATE` (one line in `config.py`) and retrain, if 8 kHz clips the
  content. Until then, capture at the DAQ's native rate and let the code resample.

---

## 4. What the ML side returns (output contract)

Every prediction is the same JSON-style dict (`predict.py`):

```json
{ "prediction": "3", "confidence": 0.87, "status": "recognized" }
{ "prediction": "unknown", "confidence": 0.42, "status": "low_confidence" }
```

- `prediction`: `"0"`–`"9"` or `"unknown"`.
- `confidence`: top softmax probability, 0.0–1.0.
- `status`: `"recognized"` or `"low_confidence"` (below `CONFIDENCE_THRESHOLD`).

Frontend already consumes this (dashboard Prediction/Confidence panels).

---

## 5. How to plug in (Sprint 4)

Nothing in the model, features, or training changes. The only new code is a
**source** that fills the rolling buffer. Two integration points, both already
staged:

1. **Replay a capture file today** (no hardware needed) to test the full path:
   ```python
   from signal_backend import SignalBackend
   be = SignalBackend(source="path/to/laser_capture.csv")   # or .wav
   ```
   The dashboard treats a replayed laser file exactly like a live mic.

2. **Live DAQ in Sprint 4**: add a `_DAQSource` mirroring `_MicSource`
   (same `start/stop/latest` interface) that reads the DAQ SDK stream into the
   buffer. Swap it in `SignalBackend.__init__`. That's the whole change.

---

## 6. Open items to confirm with hardware team (Sprint 3)

- [ ] DAQ export format: WAV or CSV? (both supported — pick one primary)
- [ ] Native DAQ sample rate (Hz) → decides whether we raise `SAMPLE_RATE`
- [ ] Voltage range / units → decides whether extra scaling is needed before
      normalization
- [ ] Streaming API: does the ADC/DAQ SDK expose a callback or a poll? (decides
      `_DAQSource` shape)
- [ ] Expected background noise profile → tune augmentation + confidence threshold

---

## 7. Collecting team voices (Sprint 3 task, no hardware needed)

To move off the public dataset and onto real voices before laser data exists:

1. Run the tester UI: `python src/app.py` (or the desktop dashboard).
2. In **Record / Upload → Save this clip for training**, each member records
   digits 0–9 several times, saving with their name as speaker.
   Files land in `data/raw/` as `‹digit›_‹name›_‹index›.wav`.
3. Retrain including the new clips, with augmentation for robustness:
   ```
   python src/train.py --model lstm --augment
   python src/train.py --model cnn  --augment
   ```
4. Re-check the operating point:
   ```
   python scripts/threshold_sweep.py --model lstm
   ```

Target: at least ~20 clips per digit per speaker for a meaningful retrain.
