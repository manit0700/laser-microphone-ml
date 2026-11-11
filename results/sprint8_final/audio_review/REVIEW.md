# Six-failure audio review (Sprint 8 final, Agent A)

Status: **objective checks complete; human listening NOT done** (the agent cannot listen to audio).
Scope: the 57-clip development playlist (validation clean clips, `results/sprint8_followup/listening_six_review/`).
No test-split clip is used. No label has been changed. Nothing here is a human-confirmed label.

## 1. Objective checks (verified; `objective_checks.csv`, `objective_checks_summary.json`)

| check | result |
|---|---|
| playlist WAV byte-identical to source file (sha256) | 57/57 |
| saved model prediction and confidence reproduce the sheet | 57/57 (uses the frozen T and threshold) |
| manifest label equals the filename's digit | 57/57 |
| sample rates | 33 at 16 kHz, 24 at 8 kHz (both groups) |
| files at full-scale peak (|x| >= 0.999) | 33 |
| flat-topped clipping (>=3 consecutive full-scale samples) | **0** |
| leading/trailing silence and spectral shares | recorded per clip |

Full-scale peaks are most likely peak normalisation, not clipping: no clip has a flat top. The earlier claim that the
rig might clip cannot be checked from these files.

## 2. Structure of the errors (measured, `error_vs_control_structure.json`, `label_review_status.csv`)

Error population (45 clips predicted as another class):
- **six→eight: 29 clips.** Subset of 17 are short and quiet (duration < 0.8 s and peak < 0.5); 12 are not.
  Median duration 0.56 s and peak 0.25 for the six→eight group, against 0.74 s and 0.66 for the correct controls.
- other errors: 7 six→seven, 6 six→unknown, 2 six→three, 1 six→zero (16 clips; mostly full 1 s, full-scale).

Error versus correct controls on burst count, onset and sample rate do **not** separate the groups (median 2 bursts in both;
16 kHz share 60% vs 50%). With 12 controls, these comparisons are weak.

Short quiet clips are not generally harder: across all validation digits, short clips score higher on clean audio
(92.8% vs 87.8%, `diagnostics/duration_effect_val.json`). The six→eight short-quiet pattern is therefore specific to the
digit six, not a general short-clip failure.

## 3. Preprocessing checks (code read, verified against source)

- Training clean path (`train_reference.load_clip(live=False)` = `preprocess.preprocess_file`) and the deployed file
  predictor (`predict.predict_file` -> `infer` -> `preprocess_waveform`) use the same resample, `reduce_noise`,
  normalise, endpoint/centre, pad steps. They agree.
- Training laser path (`load_live_path`): 200–3800 Hz 4th-order zero-phase band-pass, `enhance_waveform`, then
  `preprocess_waveform`. The dashboard laser channel uses the same band (`LMML_LASER_BANDPASS` default `200,3800`) and
  the same enhancement, applied block-wise. The predictor's **file** path does **not** apply that band-pass or enhancement.
  Any laser-channel file scored with `predict_file` is therefore not preprocessed as the model was trained on laser data.
- `reduce_noise` also applies a 6-pole band-pass plus pre-emphasis, but only when `ENABLE_FILTER` (default False) or the
  runtime toggle is on. No Sprint 8 training or replay script calls `set_filter`, so the Sprint 8 models were trained and
  replayed with this step off. The package must record that setting.

## 4. Diagnosis (provisional, not confirmed by listening)

- The six→eight failures contain two sub-populations: short quiet recordings (17) and other six→eight clips (12).
- Other errors (16) are full-length, full-scale recordings. Several show two or more separated bursts in the 1 s window
  (visual inspection of `waveform_spectrogram_inspection.png`). That pattern is consistent with a second word or noise in
  the window, but it is **not confirmed**: visual inspection is not listening.
- No objective defect (clipping, sample-rate mismatch, byte corruption, label-filename mismatch) was found.

## 5. Human actions required (exact)

1. Open `results/sprint8_final/audio_review/label_review_status.csv` (or the playlist folder) and listen to each row in
   `playlist_no` order.
2. Fill `human_heard_digit` (the digit heard, or `unclear`), `human_ambiguous_or_noise` (yes/no and what is audible), and
   `human_confirmed_label` only when the heard digit matches the manifest label.
3. A second reviewer should check any row where the heard digit differs from the manifest label before any relabelling.
4. Do not change manifest labels from this sheet automatically. A label change needs a new manifest version and a note in
   `docs/EXPERIMENTS.md`.

## 6. Recommended next action

- Human listening of the 57 clips, then decide whether the 16 full-length errors are label or content problems.
- Keep the frozen model. Do not retrain or re-threshold on these clips: they are development data already inspected, and
  tuning on them would be the repeated-tuning pattern the audit warned against.
