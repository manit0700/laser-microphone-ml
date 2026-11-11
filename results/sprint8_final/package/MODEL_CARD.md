# Model card: lmml-sprint8-finetuned_v2

**Status: experimental, historical Sprint 8 frozen candidate.** Not validated on hardware. Do not use for deployment decisions.

## Model
- Architecture: CNN on log-mel features (40 mel bands, 8 kHz, 1.0 s clips). Source: src/model.py (`model_from_checkpoint`).
- Output: 11 classes, digits 0-9 plus `unknown` (index 10).
- Checkpoint: results/sprint8/models/finetuned_v2_cnn.pt, SHA-256 `9d8a1108...8458d1598` (full value in SHA256SUMS). Trained with simulated laser conditions on top of clean data (Round 2, epoch 30; see results/sprint8/eval/FROZEN_SETTINGS.json under rounds.v2).
- Decision: `probs = softmax(logits / T)`, T = 0.5721844434738159; prediction = argmax; reject to `unknown` if max probability < 0.26. Both values are frozen, read from FROZEN_SETTINGS.json.

## Preprocessing (must match)
- Sample rate: 8000 Hz, pinned by the package (`LMML_SAMPLE_RATE`). The repository reads models/model_meta.json otherwise.
- Clean mode: load, preprocess_waveform (no band-pass, no enhancement).
- Laser mode: load, zero-phase Butterworth order-4 band-pass 200-3800 Hz, enhance_waveform, preprocess_waveform. Matches the training "sim" path and the dashboard default (LMML_LASER_BANDPASS=200,3800). The repository's `predict.predict_file` does not apply the laser filter.
- reduce_noise (ENABLE_FILTER) is off, as in Sprint 8 training and replay.

## Evidence (provenance)
- Development and held-out results are in results/sprint8/ and results/sprint8_followup/. The follow-up validation selected settings on validation data, so these figures are not independent confirmatory test results.
- Package self-test: 24 validation examples (12 clean, 12 laser), 0 label mismatches. Probabilities match the repository pipeline exactly (max |dp| = 0 on 20 clips).
- Replay (validation clips, controlled stream): see results/sprint8_final/replay/REPLAY_ANALYSIS.md.

## Intended use
- Demonstration and offline scoring of development WAV files. Teaching and review of the Sprint 8 pipeline.

## Out of scope / limitations
- No validated real-rig accuracy. The simulator's SNR-parameter bins are not measured rig SNR.
- Speaker-level consistency and non-digit rejection are weak (see the follow-up report). The rejection threshold trades non-digit rejection against valid-digit rejection.
- Historical data: 40 voice-speaker IDs plus five synthetic-noise groups. Not a broad population.
- Jetson / TensorRT / FP16 behaviour is untested.

## Status of newer models
Newer checkpoints from the follow-up are not part of this package. They are not validated on an independent set and are not used here.
