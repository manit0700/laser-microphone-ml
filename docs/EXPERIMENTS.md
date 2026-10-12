# Experiment Log — Laser Microphone Spoken-Digit Recognition

Team Eclipse · CSE 4316/4317 · ML/data pipeline

Every training/evaluation experiment, what it tested, the numbers, and what we
concluded. Newest first. Numbers marked **(leaky)** come from the old random
clip-by-clip split, where the same speakers (and some duplicate clips) were in
both training and test — they overstate accuracy on new voices. From Exp 4 on,
all test numbers are on **speakers never seen in training**.

---

## Status at a glance

| Item | State |
|---|---|
| Best honest result (CNN, unseen speakers) | **93.8 % overall · 95.2 % digits · 80.8 % unknown rejected** (Exp 4, "improved") |
| Report-grade result (5-fold CV, CNN, 8 kHz) | **92.4 % ± 0.5 overall · 93.6 % ± 0.4 digits · 76.2 % ± 3.2 unknown rejected** (Exp 5) |
| Laser hardware | Laser **does** pick up loud speech, but only ~2× above its noise after a 200–4000 Hz band-pass (raw signal is ~96 % 60 Hz hum). Live laser recognition works on some loud digits, not reliably (Exp 6). |
| Training data | Public microphone datasets (Speech Commands, FSDD) + team PCM1808 recordings + 3,500 **simulated** laser clips. **No real laser recordings yet.** |

---

## Exp 6 — First laser session on the Jetson + simulated-laser fine-tune (Oct 12)

**1. Simulated laser audio (cloud, newest Kaggle CNN, 8 kHz fold 0, 1,100 clips from unseen speakers).**
Each test clip was passed through `physics_laser_sim.py` (surface resonance, knife-edge
optics, shot/electronic noise, 60/120 Hz hum, AC coupling, ADC clipping), then the model
was fine-tuned on simulated clips from **training speakers only** (+ clean replay).

| Model | Mic audio: digits / unknown rejected | Simulated laser: digits / unknown rejected |
|---|---|---|
| Base CNN (never saw laser audio) | 94.0 % / 74 % | **63.8 %** / 52 % |
| Fine-tuned on simulated laser | 92.5 % / 84 % | **75.9 %** / 70 % |

By simulated laser quality (digits, base → fine-tuned): good SNR > 20 dB 67 → 79 %,
medium 10–20 dB 65 → 78 %, poor < 10 dB 62 → 72 %. ±3 % at this sample size.

**2. Same fine-tune on the Jetson** (`physics_laser_sim.py --per-digit 300 --unknown 500`,
then `finetune.py --speakers physlaser --model cnn`): held-out simulated laser clips
68.6 % → 77.0 % (10 epochs, still improving), base-data accuracy 98.0 → 97.7 %
(that check includes training voices, so it only shows nothing was forgotten).
Saved with 15 epochs on the Nano; the previous model is in `models/backups/`.
**Not pushed to `main`** — it is an experiment until it is checked on real laser audio.

**3. Real laser channel measurements (PCM1808, 44.1 kHz, `test_daq_signal.py` + a band-limited version).**

| Channel | Raw RMS quiet | 200–4000 Hz RMS quiet | 200–4000 Hz RMS speaking |
|---|---|---|---|
| Standard mic | 0.0037 | 0.003 | 0.004–0.05 |
| Laser | 0.25–0.43 | 0.012–0.015 | 0.021–0.031 (same moments as the mic) |

- ~96 % of the laser signal is low-frequency hum. In the speech band the laser **does**
  respond to speech (~2× its floor), and the jump is ~8× larger than the mic's, so it is
  not electrical crosstalk.
- The dashboard was gating and classifying the laser channel on the **raw** signal
  (`ENABLE_FILTER = False`), so the 0.03 gate was always open and the model mostly saw hum.
  **Fix (commit `b794ce9`):** the laser classifier band-passes 200–3800 Hz before the gate
  and the model; gate 0.02 on the filtered signal (`LMML_LASER_BANDPASS`,
  `LMML_LASER_SILENCE_RMS` override). Commit `13a9d1d`: the laser side uses the
  sim-fine-tuned CNN instead of the ensemble (`LMML_LASER_MODEL`).
- Result: better, but laser recognition is still unreliable — some digits wrong, some not
  detected. At a speech-to-noise ratio of ~2 this is expected; the limit is signal quality,
  not the model.

**4. Speaker playback.** The first automated collection (`replay_record_laser.py`, 2,300
clips) produced nothing usable: the Nano's default output is HDMI (no speaker), so both
channels heard silence (mic SNR 1.0). The 231 clips it kept by chance were moved out of
`data/raw`. The rig speaker is driven by the function generator, so automatic collection
needs a USB/Bluetooth speaker on the Nano. Also fixed: Ubuntu's PyAudio playback crash on
Python 3.10 (`replay_record_laser.py` falls back to `aplay`, commit `19ab92c`).

**Conclusion.** The ML side is ready for laser audio; the hardware needs a higher
speech-to-noise ratio (target ≥ 5: battery power, shared ground, lower receiver gain,
beam on the photodiode edge). Next: record real team voices through the laser
(`record_daq_wav.py --channel laser`) and fine-tune on those.

---

## Exp 5 — Sample rate: 8 kHz vs 16 kHz (5-fold CV)

**Question.** Everything was resampled to 8 kHz, which discards all content above
4 kHz. Does keeping 16 kHz improve recognition — especially "six", which the
model over-predicts?

**Evidence that motivated it** (740 Speech Commands clips, share of each digit's
speech energy above 4 kHz = what 8 kHz throws away):

| 0 | 1 | 2 | 3 | 4 | 5 | **6** | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| 0.3 % | 0.0 % | 0.3 % | 0.1 % | 0.0 % | 0.1 % | **11.5 %** | 1.4 % | 0.4 % | 0.1 % |

"Six" loses most of its "s". "Three" loses almost nothing by energy (the "th" is
very quiet), so 8 kHz is unlikely to be the cause of 3→2 confusions. Energy ≠
information, hence the controlled experiment.

**Setup.** CNN, dropout 0.4, max 40 epochs, early stopping (patience 6).
Same "improved" recipe for both (SpecAugment, ReduceLROnPlateau, noise down to
0 dB SNR, label smoothing 0.1). Only difference: `LMML_SAMPLE_RATE` 8000 vs
16000 (FFT/hop scaled so features keep the same shape). Speaker-grouped
5-fold CV = 10 trainings. Winner chosen on mean **validation** balanced accuracy
and saved to `selection.json` before testing; test folds scored once.

**Run.**
```bash
KAGGLE_ACCELERATOR=NvidiaTeslaT4 KAGGLE_MODE=cvcompare KAGGLE_MODEL=cnn KAGGLE_COMPARE=sr \
  KAGGLE_FOLDS=5 KAGGLE_DROPOUT=0.4 KAGGLE_EPOCHS=40 KAGGLE_PATIENCE=6 bash scripts/kaggle_push_kernel.sh
```

**Results** (5 folds, 44.8 K clips, 793 duplicates removed; each test fold ≈ 8,800
clips from ~500 unseen speakers; "unknown" threshold tuned on validation):

| Test metric (mean ± std over 5 folds) | 8 kHz | 16 kHz |
|---|---|---|
| Balanced accuracy (11 classes) | 92.00 % ± 0.63 | 92.35 % ± 0.89 |
| Overall accuracy | 92.42 % ± 0.54 | 92.77 % ± 0.65 |
| Digits correct | 93.58 % ± 0.42 | 93.96 % ± 0.70 |
| Digits wrongly rejected | 1.31 % ± 0.15 | 1.38 % ± 0.20 |
| Unknown rejected | 76.22 % ± 3.24 | 76.29 % ± 2.99 |
| Validation balanced acc (used for selection) | **92.06 %** | 91.92 % |
| Epochs run per fold | 40, 37, 29, 32, 40 | 40, 40, 40, 40, 40 |
| Tuned threshold / temperature | 0.30–0.31 / ≈0.49 | 0.31–0.33 / ≈0.50 |

Paired difference (16 kHz − 8 kHz, same folds): **+0.35 % ± 0.75 %**, 16 kHz better
on 3 of 5 folds. Selected on validation (before testing): **8 kHz**.

**Conclusions.**
- **No meaningful difference.** The gap (+0.35 pts) is smaller than the fold-to-fold
  spread; validation slightly preferred 8 kHz, test slightly preferred 16 kHz.
  Keep **8 kHz** (the pre-registered choice; cheaper; the laser's usable bandwidth
  is likely below 4 kHz anyway).
- Caveat: every 16 kHz fold hit the 40-epoch cap (8 kHz mostly early-stopped), so
  16 kHz may still have been improving. Worth one longer run only if needed.
- The tuned "unknown" threshold landed at ~0.30, confirming a fixed 0.60 would
  reject too many real digits under label smoothing.
- Unknown rejection (~76 %) is the weakest number → more/realistic "unknown"
  data (hardware recordings) is the next lever, not sample rate.
- Per-class ("six") breakdown was not saved by this run's test phase.

---

## Exp 4 — Baseline vs improved recipe, honest split (Kaggle, CNN)

**Question.** With duplicates removed and a speaker-grouped split, what is the
real accuracy on new voices, and do SpecAugment + LR scheduler + stronger noise
+ label smoothing help?

**Setup.** ~44.8 K clips (Speech Commands digits + FSDD + 3,000 non-digit words
as "unknown"). Split by speaker: train 35,193 / val 4,413 / test 4,405 clips;
0 speakers shared. CNN, dropout 0.4, max 60 epochs, patience 8. Temperature
calibration on validation. (This run predates validation threshold tuning —
"unknown" = argmax.)

| | Baseline | Improved |
|---|---|---|
| Recipe | no SpecAugment · fixed LR · noise ≥ 8 dB | SpecAugment · ReduceLROnPlateau · noise ≥ 0 dB · label smoothing 0.1 |
| Epochs (best) | 48 (40) | 60 (53) |
| **Overall (test)** | 93.6 % | **93.8 %** |
| **Digits correct** | **95.3 %** | 95.2 % |
| **Unknown rejected** | 77.0 % | **80.8 %** |
| "Three" correct | 94.9 % | 95.1 % |
| Predicted "6" vs actual "6" | 428 vs 383 | 412 vs 383 |
| Temperature | 0.91 | 0.53 |
| Fit verdict | good fit | good fit |

**Conclusions.**
- Honest accuracy (~93.7 %) is close to the old leaky 94 %: Speech Commands has
  thousands of speakers, so the leak mattered less than feared. Report these.
- "Improved" mainly helps rejection (+3.8 pts); digits unchanged. One split and
  ~300 unknown test clips → needs CV to confirm.
- "Three" is fine (≈95 %); the earlier 3→2 confusion was specific to the old split.
- "Six" is still over-predicted → Exp 5.
- Validation losses are **not comparable** (0.78 vs 0.21): label smoothing raises
  the loss by design. Model selection therefore uses balanced accuracy, not loss.

Outputs: `kaggle_failed/compare/cnn_{baseline,improved}/` (run finished; only the
final summary-CSV write crashed — fixed in `de2d551`).

---

## Exp 3 — Dropout sweep 0.2–0.5 (Kaggle, CNN) **(leaky split)**

**Setup.** 44,804 clips, random split, max 40 epochs, early stopping on
validation loss (patience 6), fixed LR 1e-3.

| Dropout | Epochs | Best val loss | Best val acc | Verdict |
|---|---|---|---|---|
| 0.2 | 16 | 0.243 | 93.0 % | good fit |
| 0.3 | 24 | 0.213 | 93.8 % | good fit |
| **0.4** | 40 | **0.185** | **94.4 %** | still improving |
| 0.5 | 40 | 0.207 | 93.9 % | still improving |

Test (dropout 0.4, 4,481 clips, leaky): 94.0 % overall · 94.6 % digits · 85.1 % unknown rejected.

**Conclusions.** Dropout 0.4 is best; lowering to 0.2–0.25 made it worse. No run
overfit (train accuracy is below val because training clips are augmented), so
L2 regularization is not needed. 0.4/0.5 were still improving at 40 epochs → an
LR scheduler was added (used from Exp 4).

Figures: `kaggle_sweep_cnn/cnn_dropout_sweep.png`,
`kaggle_sweep_cnn/confusion_matrix_cnn_report.png`.

---

## Exp 2 — Physics-simulated laser audio (local, CNN + LSTM)

**Question.** Can the model be prepared for laser audio before the hardware
works?

**Setup.** `scripts/physics_laser_sim.py` turns clean clips into simulated laser
captures: surface as damped mass-spring modes (250–3000 Hz), Gaussian-spot
knife-edge optics (erf), shot + electronic noise, 60 Hz hum + harmonics, 120 Hz
light flicker, AC coupling, amplifier bandwidth, ADC clipping — all randomised
per clip. Fine-tuned the shipped models on them (`finetune.py --speakers physlaser`).

| Model | Before | After |
|---|---|---|
| LSTM (4 epochs) | 63.7 % | 72.6 % |
| CNN (15 epochs) | 58.4 % | 76.3 % |

Accuracy on held-out simulated clips; normal-audio accuracy unchanged. Still
improving when stopped. Some simulated clips are near-impossible by design
(0 dB SNR, bad alignment). Preview figure: `results/plots/physics_laser_sim.png`.

---

## Exp 1 — Shipped Kaggle models (Sep 27) **(leaky split)**

11-class (digits + unknown), trained on Kaggle T4, 44,804 clips, random split.

| | Digits | Unknown rejected |
|---|---|---|
| LSTM | 93.7 % | 85.7 % |
| CNN | 94.4 % | 89.0 % |
| Ensemble | 95.7 % | 96.3 % |

These are the models currently on the Nano.

---

## Data findings

- **Duplicates:** 473 byte-identical files in `data/raw` (e.g. the 552 "manit"
  clips are 92 recordings × 6). Removed automatically before every split
  (`src/splits.py`). An earlier fine-tune result on Manit's voice (86 % → 97 %) was
  inflated by this; honest: 88.9 % → 100 % on only 18 held-out clips.
- **Class balance:** fine. Test clips per digit 381–451 (ratio 1.18×); "unknown"
  ~301. Over-predicted "2"/"6" is acoustic confusion, not imbalance.
- **Sources:** all 44.8 K training clips are public microphone recordings —
  Speech Commands v0.02 (16 kHz, thousands of speakers) and FSDD (8 kHz,
  6 speakers). AudioMNIST (30 K clips, 60 speakers, 48 kHz) is on the Nano but
  not yet used on Kaggle.

---

## Deployment measurements (Jetson prep, CPU in the dev container)

| | Time per clip |
|---|---|
| Features (preprocess + mel) | 0.5 ms |
| CNN, PyTorch FP32 | 0.65 ms |
| **CNN, ONNX Runtime** | **0.15 ms** |
| LSTM, PyTorch FP32 | 13 ms |

The CNN's `AdaptiveAvgPool2d` (10×15 → 4×4) could not be exported to ONNX; the
export uses an exact matrix-multiply equivalent (max difference ~1e-7). Real
Jetson FP16/TensorRT numbers: `python3 scripts/export_jetson.py` on the Nano.

---

## Method changes over time (why numbers before/after differ)

| Commit | Change |
|---|---|
| `9981c19` | Dropout configurable; early stopping on val loss; overfitting report |
| `f366cd2` | **Speaker-grouped split + duplicate removal** (honest test); SpecAugment; LR scheduler; noise to 0 dB |
| `cc29a70` | Delta MFCCs (LSTM); temperature calibration; K-fold CV; Jetson export; "unknown" recording mode |
| `37870f6` | "Unknown" threshold tuned on **validation**; ensemble keeps the unknown class; CV comparison that tests once |
| `8b0b266` | Sample rate configurable (8/16 kHz); recordings saved at 16 kHz |
| `de2d551` | Fix summary-CSV crash (Exp 4) |

---

## Open items

1. Laser: raise the speech-band SNR from ~2 to ≥ 5 (battery power, shared ground,
   lower receiver gain, beam on the photodiode edge). Measure with the band-limited
   RMS check (Exp 6) or `replay_record_laser.py --test` (needs a speaker on the Nano).
1b. Real team voices through the laser: `record_daq_wav.py --all --channel laser
   --speaker laser<name> --takes 10`, then fine-tune with the same band-pass as the dashboard.
2. Real laser data via `replay_record_laser.py`, then calibrate the physics
   simulator (`--calibrate-from laserplay`).
3. Device test set: team PCM1808 recordings kept out of training (`data/device_test/`).
4. Hardware "unknown" clips: `record_daq_wav.py --unknown 50`.
5. After Exp 5: train the final LSTM + CNN at the winning rate, run
   `tune_threshold.py`, deploy to the Nano.
