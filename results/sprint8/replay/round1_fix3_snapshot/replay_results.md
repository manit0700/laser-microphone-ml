# Sprint 8 — software replay of the live classifier path (agent C)

Machine: **cloud dev container, 2 vCPU, CPU-only PyTorch 2.15.0** — all timings are local software timings, not Jetson/hardware latency.

Deployed model: `/home/claude/lmml/models/best_model_cnn.pt` (sha256 da27ca83ab93…, threshold 0.26, temperature 0.498); LSTM for the ensemble: `/home/claude/lmml/models/best_model.pt`.

What was replayed: synthetic streams built from REAL clips in `data/raw` (digits from test-split speakers of `results/sprint8/data/manifest.csv`; non-digit words from `data/raw/unknown`; background noise = `data/raw/unknown/noise_*.wav` clips or numpy white noise, both labelled synthetic), pushed to `signal_backend._LiveClassifier.push()` at the dashboard's 48 kHz stream rate in 256, 50 ms chunks, with `predict()` after every push (256 ms = one PCM1808 hardware read of 12288 frames, exactly what `signal_dashboard2.update_frame()` does; 50 ms = GUI-tick-like).

Levels: `speech_rms` is the 200-3800 Hz band RMS of the word (default 0.05); the laser gate is the dashboard default `LMML_LASER_SILENCE_RMS=0.02` (measured on the band-passed signal, like the dashboard; the rig measured quiet ~0.013 / speech ~0.03). Word/gate ratio here is 2.5 vs ~1.5 on the rig, so the `low_level_*` scenarios sweep the ratio down to 0.5. The std channel uses `DAQ_SILENCE_RMS=0.03` on the raw signal, as in the dashboard.

An **event** is a NEW decision by the classifier (`_LiveClassifier.on_decision` / `n_decisions`, added in this sprint): one per finished word in end-of-word mode, one per commit in sliding mode. Tuples returned by `predict()` are NOT compared (two identical words return identical tuples).

## 1. Event handling (deterministic) — laser channel as deployed (CNN, band-pass 200-3800 Hz, end-of-word mode)

### push size 256.0ms

| scenario | words | events | missed | dup | false act. | spurious Unknown | 1 event/word | latency mean/max (s) |
|---|---|---|---|---|---|---|---|---|
| single_digit | 1 | 1 | 0 | 0 | 0 | 0 | yes | 0.47/0.47 |
| repeat_3x_gap0.6 | 3 | 3 | 0 | 0 | 0 | 0 | yes | 0.40/0.54 |
| five_digits_gap0.8 | 5 | 5 | 0 | 0 | 0 | 0 | yes | 0.32/0.45 |
| long_pauses_3s | 4 | 4 | 0 | 0 | 0 | 0 | yes | 0.34/0.46 |
| silence_5s | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.02 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.04 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.08 | 0 | 4 | 0 | 0 | 4 | 0 | NO | -/- |
| noise_only_real_noise_clipA_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.02 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.04 | 0 | 5 | 0 | 0 | 5 | 0 | NO | -/- |
| noise_only_real_noise_clipA_rms0.08 | 0 | 4 | 0 | 0 | 4 | 0 | NO | -/- |
| noise_only_real_noise_clipB_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipB_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipB_rms0.02 | 0 | 5 | 0 | 0 | 5 | 0 | NO | -/- |
| noise_only_real_noise_clipB_rms0.04 | 0 | 4 | 0 | 0 | 4 | 0 | NO | -/- |
| noise_only_real_noise_clipB_rms0.08 | 0 | 4 | 0 | 0 | 4 | 0 | NO | -/- |
| unknown_words_gap0.8 | 5 | 5 | 0 | 0 | 0 | 0 | yes | 0.27/0.34 |
| low_level_rms0.03 | 3 | 3 | 0 | 0 | 0 | 0 | yes | 0.27/0.35 |
| low_level_rms0.02 | 3 | 3 | 0 | 0 | 0 | 0 | yes | 0.24/0.33 |
| low_level_rms0.015 | 3 | 3 | 0 | 0 | 0 | 0 | yes | 0.19/0.21 |
| low_level_rms0.01 | 3 | 1 | 2 | 0 | 0 | 0 | NO | 0.55/0.55 |
| back_to_back_gap0.2 | 3 | 3 | 0 | 0 | 0 | 0 | yes (merge allowed) | 0.31/0.38 |
| digits_in_noise_snr20dB | 4 | 4 | 0 | 0 | 0 | 0 | yes | 0.36/0.48 |
| digits_in_noise_snr10dB | 4 | 4 | 0 | 0 | 0 | 0 | yes | 0.37/0.40 |
| digits_in_noise_snr5dB | 4 | 4 | 0 | 0 | 0 | 0 | yes | 0.42/0.61 |
| mixed_digit_unknown_digit | 3 | 3 | 0 | 0 | 0 | 0 | yes | 0.33/0.38 |

### push size 50.0ms

| scenario | words | events | missed | dup | false act. | spurious Unknown | 1 event/word | latency mean/max (s) |
|---|---|---|---|---|---|---|---|---|
| single_digit | 1 | 1 | 0 | 0 | 0 | 0 | yes | 0.27/0.27 |
| repeat_3x_gap0.6 | 3 | 3 | 0 | 0 | 0 | 0 | yes | 0.29/0.30 |
| five_digits_gap0.8 | 5 | 5 | 0 | 0 | 0 | 0 | yes | 0.22/0.27 |
| long_pauses_3s | 4 | 4 | 0 | 0 | 0 | 0 | yes | 0.22/0.33 |
| silence_5s | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.02 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.04 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.08 | 0 | 4 | 0 | 0 | 4 | 0 | NO | -/- |
| noise_only_real_noise_clipA_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.02 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.04 | 0 | 5 | 0 | 0 | 5 | 0 | NO | -/- |
| noise_only_real_noise_clipA_rms0.08 | 0 | 4 | 0 | 0 | 4 | 0 | NO | -/- |
| noise_only_real_noise_clipB_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipB_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipB_rms0.02 | 0 | 5 | 0 | 0 | 5 | 0 | NO | -/- |
| noise_only_real_noise_clipB_rms0.04 | 0 | 4 | 0 | 0 | 4 | 0 | NO | -/- |
| noise_only_real_noise_clipB_rms0.08 | 0 | 4 | 0 | 0 | 4 | 0 | NO | -/- |
| unknown_words_gap0.8 | 5 | 6 | 0 | 1 | 0 | 0 | NO | 0.08/0.22 |
| low_level_rms0.03 | 3 | 3 | 0 | 0 | 0 | 0 | yes | 0.22/0.25 |
| low_level_rms0.02 | 3 | 3 | 0 | 0 | 0 | 0 | yes | 0.10/0.16 |
| low_level_rms0.015 | 3 | 3 | 0 | 0 | 0 | 0 | yes | 0.08/0.11 |
| low_level_rms0.01 | 3 | 1 | 2 | 0 | 0 | 0 | NO | 0.38/0.38 |
| back_to_back_gap0.2 | 3 | 3 | 0 | 0 | 0 | 0 | yes (merge allowed) | 0.24/0.29 |
| digits_in_noise_snr20dB | 4 | 4 | 0 | 0 | 0 | 0 | yes | 0.27/0.34 |
| digits_in_noise_snr10dB | 4 | 4 | 0 | 0 | 0 | 0 | yes | 0.19/0.25 |
| digits_in_noise_snr5dB | 4 | 4 | 0 | 0 | 0 | 0 | yes | 0.28/0.42 |
| mixed_digit_unknown_digit | 3 | 3 | 0 | 0 | 0 | 0 | yes | 0.23/0.26 |

Reading: `low_level_rms0.015/0.01` are words at or below the silence gate (0.02) — not detected by design; `noise_only_*` rows with events are continuous noise ABOVE the gate: the end-of-word timer never fires, so `WORD_MAX_SEC=1.1` forces one classification per 1.1 s; whether it is a digit (false activation) or Unknown is a model question (section 3).

### Before the fixes (same scenarios, original code)

- chunk_256.0ms: scenarios failing one-event-per-word: low_level_rms0.015, low_level_rms0.01, digits_in_noise_snr20dB
- chunk_50.0ms: scenarios failing one-event-per-word: low_level_rms0.015, low_level_rms0.01

### Short-word placement sweep (6 real digits of 0.22-0.26 s × 16 onset offsets in 16 ms steps)

| push size | before fix | after fix |
|---|---|---|
| 256 ms | 81/96 | 96/96 |
| 50 ms | 96/96 | 96/96 |

## 2. Event handling — other configurations

### laser channel with `LMML_LASER_WORD_MODE=0` (sliding window, CNN, band-pass)

**chunk_256.0ms** — 13/30 scenarios give exactly one event per word

| scenario | words | events | missed | dup | false act. | spurious Unknown | 1 event/word | latency mean/max (s) |
|---|---|---|---|---|---|---|---|---|
| single_digit | 1 | 2 | 0 | 1 | 0 | 0 | NO | -0.04/-0.04 |
| repeat_3x_gap0.6 | 3 | 1 | 2 | 0 | 0 | 0 | NO | 0.25/0.25 |
| five_digits_gap0.8 | 5 | 6 | 0 | 1 | 0 | 0 | NO | -0.04/0.19 |
| long_pauses_3s | 4 | 4 | 0 | 0 | 0 | 0 | yes | 0.15/0.72 |
| silence_5s | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.02 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.04 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.08 | 0 | 2 | 0 | 0 | 1 | 1 | NO | -/- |
| noise_only_real_noise_clipA_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.02 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.04 | 0 | 3 | 0 | 0 | 3 | 0 | NO | -/- |
| noise_only_real_noise_clipA_rms0.08 | 0 | 2 | 0 | 0 | 2 | 0 | NO | -/- |
| noise_only_real_noise_clipB_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipB_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipB_rms0.02 | 0 | 1 | 0 | 0 | 1 | 0 | NO | -/- |
| noise_only_real_noise_clipB_rms0.04 | 0 | 1 | 0 | 0 | 1 | 0 | NO | -/- |
| noise_only_real_noise_clipB_rms0.08 | 0 | 1 | 0 | 0 | 1 | 0 | NO | -/- |
| unknown_words_gap0.8 | 5 | 2 | 3 | 0 | 0 | 0 | NO | 0.06/0.16 |
| low_level_rms0.03 | 3 | 1 | 2 | 0 | 0 | 0 | NO | 0.29/0.29 |
| low_level_rms0.02 | 3 | 0 | 3 | 0 | 0 | 0 | NO | -/- |
| low_level_rms0.015 | 3 | 0 | 3 | 0 | 0 | 0 | NO | -/- |
| low_level_rms0.01 | 3 | 0 | 3 | 0 | 0 | 0 | NO | -/- |
| back_to_back_gap0.2 | 3 | 3 | 0 | 0 | 0 | 0 | yes (merge allowed) | 0.14/0.23 |
| digits_in_noise_snr20dB | 4 | 4 | 1 | 1 | 0 | 0 | NO | -0.14/0.06 |
| digits_in_noise_snr10dB | 4 | 6 | 0 | 2 | 0 | 0 | NO | -0.08/0.06 |
| digits_in_noise_snr5dB | 4 | 4 | 0 | 0 | 0 | 0 | yes | 0.04/0.12 |
| mixed_digit_unknown_digit | 3 | 2 | 1 | 0 | 0 | 0 | NO | 0.04/0.12 |

**chunk_50.0ms** — 12/30 scenarios give exactly one event per word

| scenario | words | events | missed | dup | false act. | spurious Unknown | 1 event/word | latency mean/max (s) |
|---|---|---|---|---|---|---|---|---|
| single_digit | 1 | 2 | 0 | 1 | 0 | 0 | NO | -0.33/-0.33 |
| repeat_3x_gap0.6 | 3 | 1 | 2 | 0 | 0 | 0 | NO | -0.03/-0.03 |
| five_digits_gap0.8 | 5 | 6 | 0 | 1 | 0 | 0 | NO | -0.17/-0.10 |
| long_pauses_3s | 4 | 5 | 0 | 1 | 0 | 0 | NO | -0.16/-0.07 |
| silence_5s | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.02 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.04 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.08 | 0 | 2 | 0 | 0 | 1 | 1 | NO | -/- |
| noise_only_real_noise_clipA_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.02 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.04 | 0 | 2 | 0 | 0 | 2 | 0 | NO | -/- |
| noise_only_real_noise_clipA_rms0.08 | 0 | 2 | 0 | 0 | 2 | 0 | NO | -/- |
| noise_only_real_noise_clipB_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipB_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipB_rms0.02 | 0 | 2 | 0 | 0 | 2 | 0 | NO | -/- |
| noise_only_real_noise_clipB_rms0.04 | 0 | 2 | 0 | 0 | 2 | 0 | NO | -/- |
| noise_only_real_noise_clipB_rms0.08 | 0 | 2 | 0 | 0 | 2 | 0 | NO | -/- |
| unknown_words_gap0.8 | 5 | 2 | 3 | 0 | 0 | 0 | NO | 0.41/0.95 |
| low_level_rms0.03 | 3 | 1 | 2 | 0 | 0 | 0 | NO | 0.07/0.07 |
| low_level_rms0.02 | 3 | 0 | 3 | 0 | 0 | 0 | NO | -/- |
| low_level_rms0.015 | 3 | 0 | 3 | 0 | 0 | 0 | NO | -/- |
| low_level_rms0.01 | 3 | 0 | 3 | 0 | 0 | 0 | NO | -/- |
| back_to_back_gap0.2 | 3 | 3 | 0 | 0 | 0 | 0 | yes (merge allowed) | 0.03/0.09 |
| digits_in_noise_snr20dB | 4 | 7 | 1 | 4 | 0 | 0 | NO | -0.26/-0.23 |
| digits_in_noise_snr10dB | 4 | 4 | 0 | 0 | 0 | 0 | yes | -0.02/0.10 |
| digits_in_noise_snr5dB | 4 | 8 | 0 | 4 | 0 | 0 | NO | -0.13/-0.03 |
| mixed_digit_unknown_digit | 3 | 4 | 1 | 2 | 0 | 0 | NO | -0.16/0.01 |

### standard-mic channel as deployed (LSTM+CNN ensemble, sliding window, no band-pass, gate 0.03)

**chunk_256.0ms** — 16/30 scenarios give exactly one event per word

| scenario | words | events | missed | dup | false act. | spurious Unknown | 1 event/word | latency mean/max (s) |
|---|---|---|---|---|---|---|---|---|
| single_digit | 1 | 2 | 0 | 1 | 0 | 0 | NO | -0.04/-0.04 |
| repeat_3x_gap0.6 | 3 | 1 | 2 | 0 | 0 | 0 | NO | -0.12/-0.12 |
| five_digits_gap0.8 | 5 | 5 | 1 | 1 | 0 | 0 | NO | -0.10/0.07 |
| long_pauses_3s | 4 | 2 | 2 | 0 | 0 | 0 | NO | 0.23/0.26 |
| silence_5s | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.02 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.04 | 0 | 1 | 0 | 0 | 0 | 1 | yes | -/- |
| noise_only_white_synthetic_rms0.08 | 0 | 1 | 0 | 0 | 0 | 1 | yes | -/- |
| noise_only_real_noise_clipA_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.02 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.04 | 0 | 1 | 0 | 0 | 0 | 1 | yes | -/- |
| noise_only_real_noise_clipA_rms0.08 | 0 | 1 | 0 | 0 | 0 | 1 | yes | -/- |
| noise_only_real_noise_clipB_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipB_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipB_rms0.02 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipB_rms0.04 | 0 | 1 | 0 | 0 | 1 | 0 | NO | -/- |
| noise_only_real_noise_clipB_rms0.08 | 0 | 1 | 0 | 0 | 1 | 0 | NO | -/- |
| unknown_words_gap0.8 | 5 | 1 | 4 | 0 | 0 | 0 | NO | -0.22/-0.22 |
| low_level_rms0.03 | 3 | 0 | 3 | 0 | 0 | 0 | NO | -/- |
| low_level_rms0.02 | 3 | 0 | 3 | 0 | 0 | 0 | NO | -/- |
| low_level_rms0.015 | 3 | 0 | 3 | 0 | 0 | 0 | NO | -/- |
| low_level_rms0.01 | 3 | 0 | 3 | 0 | 0 | 0 | NO | -/- |
| back_to_back_gap0.2 | 3 | 3 | 1 | 1 | 0 | 0 | NO | -0.17/0.12 |
| digits_in_noise_snr20dB | 4 | 3 | 1 | 0 | 0 | 0 | NO | 0.03/0.22 |
| digits_in_noise_snr10dB | 4 | 4 | 0 | 0 | 0 | 0 | yes | -0.08/0.06 |
| digits_in_noise_snr5dB | 4 | 5 | 0 | 0 | 0 | 1 | yes | 0.04/0.12 |
| mixed_digit_unknown_digit | 3 | 2 | 1 | 0 | 0 | 0 | NO | 0.04/0.12 |

**chunk_50.0ms** — 17/30 scenarios give exactly one event per word

| scenario | words | events | missed | dup | false act. | spurious Unknown | 1 event/word | latency mean/max (s) |
|---|---|---|---|---|---|---|---|---|
| single_digit | 1 | 3 | 0 | 2 | 0 | 0 | NO | -0.33/-0.33 |
| repeat_3x_gap0.6 | 3 | 1 | 2 | 0 | 0 | 0 | NO | 0.04/0.04 |
| five_digits_gap0.8 | 5 | 5 | 1 | 1 | 0 | 0 | NO | -0.19/-0.12 |
| long_pauses_3s | 4 | 2 | 2 | 0 | 0 | 0 | NO | -0.05/-0.01 |
| silence_5s | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.02 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.04 | 0 | 1 | 0 | 0 | 0 | 1 | yes | -/- |
| noise_only_white_synthetic_rms0.08 | 0 | 1 | 0 | 0 | 0 | 1 | yes | -/- |
| noise_only_real_noise_clipA_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.02 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.04 | 0 | 1 | 0 | 0 | 0 | 1 | yes | -/- |
| noise_only_real_noise_clipA_rms0.08 | 0 | 1 | 0 | 0 | 0 | 1 | yes | -/- |
| noise_only_real_noise_clipB_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipB_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipB_rms0.02 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipB_rms0.04 | 0 | 1 | 0 | 0 | 0 | 1 | yes | -/- |
| noise_only_real_noise_clipB_rms0.08 | 0 | 1 | 0 | 0 | 0 | 1 | yes | -/- |
| unknown_words_gap0.8 | 5 | 1 | 4 | 0 | 0 | 0 | NO | -0.23/-0.23 |
| low_level_rms0.03 | 3 | 0 | 3 | 0 | 0 | 0 | NO | -/- |
| low_level_rms0.02 | 3 | 0 | 3 | 0 | 0 | 0 | NO | -/- |
| low_level_rms0.015 | 3 | 0 | 3 | 0 | 0 | 0 | NO | -/- |
| low_level_rms0.01 | 3 | 0 | 3 | 0 | 0 | 0 | NO | -/- |
| back_to_back_gap0.2 | 3 | 2 | 1 | 0 | 0 | 0 | NO (merge allowed) | -0.11/0.09 |
| digits_in_noise_snr20dB | 4 | 4 | 1 | 1 | 0 | 0 | NO | -0.14/0.09 |
| digits_in_noise_snr10dB | 4 | 4 | 0 | 0 | 0 | 0 | yes | -0.02/0.10 |
| digits_in_noise_snr5dB | 4 | 4 | 1 | 0 | 0 | 1 | NO | -0.07/0.11 |
| mixed_digit_unknown_digit | 3 | 3 | 1 | 1 | 0 | 0 | NO | -0.16/0.01 |

### PROPOSED: standard-mic channel in end-of-word mode (ensemble, gate 0.03) — not what the dashboard does today

**chunk_256.0ms** — 26/30 scenarios give exactly one event per word

| scenario | words | events | missed | dup | false act. | spurious Unknown | 1 event/word | latency mean/max (s) |
|---|---|---|---|---|---|---|---|---|
| single_digit | 1 | 1 | 0 | 0 | 0 | 0 | yes | 0.47/0.47 |
| repeat_3x_gap0.6 | 3 | 3 | 0 | 0 | 0 | 0 | yes | 0.31/0.40 |
| five_digits_gap0.8 | 5 | 5 | 0 | 0 | 0 | 0 | yes | 0.27/0.41 |
| long_pauses_3s | 4 | 4 | 0 | 0 | 0 | 0 | yes | 0.34/0.46 |
| silence_5s | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.02 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.04 | 0 | 4 | 0 | 0 | 0 | 4 | yes | -/- |
| noise_only_white_synthetic_rms0.08 | 0 | 4 | 0 | 0 | 0 | 4 | yes | -/- |
| noise_only_real_noise_clipA_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.02 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.04 | 0 | 4 | 0 | 0 | 0 | 4 | yes | -/- |
| noise_only_real_noise_clipA_rms0.08 | 0 | 4 | 0 | 0 | 0 | 4 | yes | -/- |
| noise_only_real_noise_clipB_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipB_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipB_rms0.02 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipB_rms0.04 | 0 | 4 | 0 | 0 | 0 | 4 | yes | -/- |
| noise_only_real_noise_clipB_rms0.08 | 0 | 4 | 0 | 0 | 0 | 4 | yes | -/- |
| unknown_words_gap0.8 | 5 | 5 | 0 | 0 | 0 | 0 | yes | 0.27/0.34 |
| low_level_rms0.03 | 3 | 3 | 0 | 0 | 0 | 0 | yes | 0.27/0.35 |
| low_level_rms0.02 | 3 | 2 | 1 | 0 | 0 | 0 | NO | 0.23/0.33 |
| low_level_rms0.015 | 3 | 2 | 1 | 0 | 0 | 0 | NO | 0.18/0.19 |
| low_level_rms0.01 | 3 | 0 | 3 | 0 | 0 | 0 | NO | -/- |
| back_to_back_gap0.2 | 3 | 3 | 0 | 0 | 0 | 0 | yes (merge allowed) | 0.31/0.38 |
| digits_in_noise_snr20dB | 4 | 5 | 0 | 1 | 0 | 0 | NO | 0.17/0.34 |
| digits_in_noise_snr10dB | 4 | 4 | 0 | 0 | 0 | 0 | yes | 0.37/0.40 |
| digits_in_noise_snr5dB | 4 | 7 | 0 | 0 | 0 | 3 | yes | 0.48/0.61 |
| mixed_digit_unknown_digit | 3 | 3 | 0 | 0 | 0 | 0 | yes | 0.33/0.38 |

**chunk_50.0ms** — 25/30 scenarios give exactly one event per word

| scenario | words | events | missed | dup | false act. | spurious Unknown | 1 event/word | latency mean/max (s) |
|---|---|---|---|---|---|---|---|---|
| single_digit | 1 | 1 | 0 | 0 | 0 | 0 | yes | 0.27/0.27 |
| repeat_3x_gap0.6 | 3 | 3 | 0 | 0 | 0 | 0 | yes | 0.27/0.29 |
| five_digits_gap0.8 | 5 | 5 | 0 | 0 | 0 | 0 | yes | 0.20/0.27 |
| long_pauses_3s | 4 | 4 | 0 | 0 | 0 | 0 | yes | 0.22/0.33 |
| silence_5s | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.02 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.04 | 0 | 4 | 0 | 0 | 0 | 4 | yes | -/- |
| noise_only_white_synthetic_rms0.08 | 0 | 4 | 0 | 0 | 0 | 4 | yes | -/- |
| noise_only_real_noise_clipA_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.02 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.04 | 0 | 4 | 0 | 0 | 0 | 4 | yes | -/- |
| noise_only_real_noise_clipA_rms0.08 | 0 | 4 | 0 | 0 | 0 | 4 | yes | -/- |
| noise_only_real_noise_clipB_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipB_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipB_rms0.02 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipB_rms0.04 | 0 | 4 | 0 | 0 | 0 | 4 | yes | -/- |
| noise_only_real_noise_clipB_rms0.08 | 0 | 4 | 0 | 0 | 0 | 4 | yes | -/- |
| unknown_words_gap0.8 | 5 | 6 | 0 | 1 | 0 | 0 | NO | 0.16/0.21 |
| low_level_rms0.03 | 3 | 3 | 0 | 0 | 0 | 0 | yes | 0.19/0.22 |
| low_level_rms0.02 | 3 | 2 | 1 | 0 | 0 | 0 | NO | 0.04/0.11 |
| low_level_rms0.015 | 3 | 2 | 1 | 0 | 0 | 0 | NO | 0.06/0.06 |
| low_level_rms0.01 | 3 | 0 | 3 | 0 | 0 | 0 | NO | -/- |
| back_to_back_gap0.2 | 3 | 3 | 0 | 0 | 0 | 0 | yes (merge allowed) | 0.28/0.30 |
| digits_in_noise_snr20dB | 4 | 5 | 0 | 1 | 0 | 0 | NO | 0.09/0.30 |
| digits_in_noise_snr10dB | 4 | 4 | 0 | 0 | 0 | 0 | yes | 0.23/0.30 |
| digits_in_noise_snr5dB | 4 | 7 | 0 | 0 | 0 | 3 | yes | 0.32/0.42 |
| mixed_digit_unknown_digit | 3 | 3 | 0 | 0 | 0 | 0 | yes | 0.22/0.26 |

## 3. Recognition (label correctness of the events) — deployed CNN

### laser channel, push size 256.0ms

| scenario | digit words | correct | wrong label | rejected as Unknown | digit acc (of words) | non-digit words | rejected | false accept |
|---|---|---|---|---|---|---|---|---|
| single_digit | 1 | 1 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| repeat_3x_gap0.6 | 3 | 3 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| five_digits_gap0.8 | 5 | 5 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| long_pauses_3s | 4 | 4 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| unknown_words_gap0.8 | 0 | 0 | 0 | 0 | - | 5 | 5 | 0 |
| low_level_rms0.03 | 3 | 3 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| low_level_rms0.02 | 3 | 3 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| low_level_rms0.015 | 3 | 3 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| low_level_rms0.01 | 3 | 0 | 1 | 0 | 0.00 | 0 | 0 | 0 |
| back_to_back_gap0.2 | 3 | 3 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| digits_in_noise_snr20dB | 4 | 4 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| digits_in_noise_snr10dB | 4 | 4 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| digits_in_noise_snr5dB | 4 | 3 | 0 | 1 | 0.75 | 0 | 0 | 0 |
| mixed_digit_unknown_digit | 2 | 2 | 0 | 0 | 1.00 | 1 | 1 | 0 |

### laser channel, push size 50.0ms

| scenario | digit words | correct | wrong label | rejected as Unknown | digit acc (of words) | non-digit words | rejected | false accept |
|---|---|---|---|---|---|---|---|---|
| single_digit | 1 | 1 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| repeat_3x_gap0.6 | 3 | 3 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| five_digits_gap0.8 | 5 | 5 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| long_pauses_3s | 4 | 4 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| unknown_words_gap0.8 | 0 | 0 | 0 | 0 | - | 5 | 4 | 1 |
| low_level_rms0.03 | 3 | 3 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| low_level_rms0.02 | 3 | 3 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| low_level_rms0.015 | 3 | 3 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| low_level_rms0.01 | 3 | 0 | 1 | 0 | 0.00 | 0 | 0 | 0 |
| back_to_back_gap0.2 | 3 | 3 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| digits_in_noise_snr20dB | 4 | 4 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| digits_in_noise_snr10dB | 4 | 4 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| digits_in_noise_snr5dB | 4 | 3 | 0 | 1 | 0.75 | 0 | 0 | 0 |
| mixed_digit_unknown_digit | 2 | 2 | 0 | 0 | 1.00 | 1 | 1 | 0 |

### ensemble in end-of-word mode (proposed), chunk_256.0ms

| scenario | digit words | correct | wrong label | rejected as Unknown | digit acc (of words) | non-digit words | rejected | false accept |
|---|---|---|---|---|---|---|---|---|
| single_digit | 1 | 1 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| repeat_3x_gap0.6 | 3 | 3 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| five_digits_gap0.8 | 5 | 5 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| long_pauses_3s | 4 | 4 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| unknown_words_gap0.8 | 0 | 0 | 0 | 0 | - | 5 | 5 | 0 |
| low_level_rms0.03 | 3 | 3 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| low_level_rms0.02 | 3 | 2 | 0 | 0 | 0.67 | 0 | 0 | 0 |
| low_level_rms0.015 | 3 | 2 | 0 | 0 | 0.67 | 0 | 0 | 0 |
| low_level_rms0.01 | 3 | 0 | 0 | 0 | 0.00 | 0 | 0 | 0 |
| back_to_back_gap0.2 | 3 | 3 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| digits_in_noise_snr20dB | 4 | 3 | 0 | 1 | 0.75 | 0 | 0 | 0 |
| digits_in_noise_snr10dB | 4 | 4 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| digits_in_noise_snr5dB | 4 | 3 | 0 | 1 | 0.75 | 0 | 0 | 0 |
| mixed_digit_unknown_digit | 2 | 2 | 0 | 0 | 1.00 | 1 | 1 | 0 |

Noise-only streams: digit events are listed under *false activations* in section 1; Unknown events there are correct rejections.

### Word-window anchoring (validation split, 200 clean clips, single-word streams)

| variant | push | replay acc | agreement with direct file prediction | Unknown | no event |
|---|---|---|---|---|---|
| A_end_now | 256 ms | 0.895 | 0.875 | 5 | 2 |
| A_end_now | 50 ms | 0.900 | 0.900 | 7 | 2 |
| B_anchored | 256 ms | 0.930 | 0.935 | 3 | 2 |
| B_anchored | 50 ms | 0.930 | 0.935 | 4 | 2 |

Direct `predict.infer()` on the same 200 files: 0.940. Chosen: `B_anchored` (now in `signal_backend._word_window`).

### Same scenarios with `results/sprint8/models/finetuned_cnn.pt` (`finetuned_cnn`, threshold 0.26, T 0.581), laser channel, 256 ms pushes

Event handling:

| scenario | words | events | missed | dup | false act. | spurious Unknown | 1 event/word | latency mean/max (s) |
|---|---|---|---|---|---|---|---|---|
| single_digit | 1 | 1 | 0 | 0 | 0 | 0 | yes | 0.47/0.47 |
| repeat_3x_gap0.6 | 3 | 3 | 0 | 0 | 0 | 0 | yes | 0.40/0.54 |
| five_digits_gap0.8 | 5 | 5 | 0 | 0 | 0 | 0 | yes | 0.32/0.45 |
| long_pauses_3s | 4 | 4 | 0 | 0 | 0 | 0 | yes | 0.34/0.46 |
| silence_5s | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.02 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.04 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.08 | 0 | 4 | 0 | 0 | 0 | 4 | yes | -/- |
| noise_only_real_noise_clipA_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.02 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.04 | 0 | 5 | 0 | 0 | 0 | 5 | yes | -/- |
| noise_only_real_noise_clipA_rms0.08 | 0 | 4 | 0 | 0 | 0 | 4 | yes | -/- |
| noise_only_real_noise_clipB_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipB_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipB_rms0.02 | 0 | 5 | 0 | 0 | 1 | 4 | NO | -/- |
| noise_only_real_noise_clipB_rms0.04 | 0 | 4 | 0 | 0 | 0 | 4 | yes | -/- |
| noise_only_real_noise_clipB_rms0.08 | 0 | 4 | 0 | 0 | 0 | 4 | yes | -/- |
| unknown_words_gap0.8 | 5 | 5 | 0 | 0 | 0 | 0 | yes | 0.27/0.34 |
| low_level_rms0.03 | 3 | 3 | 0 | 0 | 0 | 0 | yes | 0.27/0.35 |
| low_level_rms0.02 | 3 | 3 | 0 | 0 | 0 | 0 | yes | 0.24/0.33 |
| low_level_rms0.015 | 3 | 3 | 0 | 0 | 0 | 0 | yes | 0.19/0.21 |
| low_level_rms0.01 | 3 | 1 | 2 | 0 | 0 | 0 | NO | 0.55/0.55 |
| back_to_back_gap0.2 | 3 | 3 | 0 | 0 | 0 | 0 | yes (merge allowed) | 0.31/0.38 |
| digits_in_noise_snr20dB | 4 | 4 | 0 | 0 | 0 | 0 | yes | 0.36/0.48 |
| digits_in_noise_snr10dB | 4 | 4 | 0 | 0 | 0 | 0 | yes | 0.37/0.40 |
| digits_in_noise_snr5dB | 4 | 4 | 0 | 0 | 0 | 0 | yes | 0.42/0.61 |
| mixed_digit_unknown_digit | 3 | 3 | 0 | 0 | 0 | 0 | yes | 0.33/0.38 |

Recognition:

| scenario | digit words | correct | wrong label | rejected as Unknown | digit acc (of words) | non-digit words | rejected | false accept |
|---|---|---|---|---|---|---|---|---|
| single_digit | 1 | 1 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| repeat_3x_gap0.6 | 3 | 3 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| five_digits_gap0.8 | 5 | 5 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| long_pauses_3s | 4 | 4 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| unknown_words_gap0.8 | 0 | 0 | 0 | 0 | - | 5 | 4 | 1 |
| low_level_rms0.03 | 3 | 3 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| low_level_rms0.02 | 3 | 3 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| low_level_rms0.015 | 3 | 3 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| low_level_rms0.01 | 3 | 0 | 1 | 0 | 0.00 | 0 | 0 | 0 |
| back_to_back_gap0.2 | 3 | 3 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| digits_in_noise_snr20dB | 4 | 4 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| digits_in_noise_snr10dB | 4 | 4 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| digits_in_noise_snr5dB | 4 | 2 | 2 | 0 | 0.50 | 0 | 0 | 0 |
| mixed_digit_unknown_digit | 2 | 2 | 0 | 0 | 1.00 | 1 | 0 | 1 |

### Same scenarios with `results/sprint8/models/reference_cnn.pt` (`reference_cnn`, threshold 0.26, T 0.581), laser channel, 256 ms pushes

Event handling:

| scenario | words | events | missed | dup | false act. | spurious Unknown | 1 event/word | latency mean/max (s) |
|---|---|---|---|---|---|---|---|---|
| single_digit | 1 | 1 | 0 | 0 | 0 | 0 | yes | 0.47/0.47 |
| repeat_3x_gap0.6 | 3 | 3 | 0 | 0 | 0 | 0 | yes | 0.40/0.54 |
| five_digits_gap0.8 | 5 | 5 | 0 | 0 | 0 | 0 | yes | 0.32/0.45 |
| long_pauses_3s | 4 | 4 | 0 | 0 | 0 | 0 | yes | 0.34/0.46 |
| silence_5s | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.02 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.04 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_white_synthetic_rms0.08 | 0 | 4 | 0 | 0 | 0 | 4 | yes | -/- |
| noise_only_real_noise_clipA_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.02 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipA_rms0.04 | 0 | 5 | 0 | 0 | 0 | 5 | yes | -/- |
| noise_only_real_noise_clipA_rms0.08 | 0 | 4 | 0 | 0 | 0 | 4 | yes | -/- |
| noise_only_real_noise_clipB_rms0.005 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipB_rms0.01 | 0 | 0 | 0 | 0 | 0 | 0 | yes | -/- |
| noise_only_real_noise_clipB_rms0.02 | 0 | 5 | 0 | 0 | 1 | 4 | NO | -/- |
| noise_only_real_noise_clipB_rms0.04 | 0 | 4 | 0 | 0 | 0 | 4 | yes | -/- |
| noise_only_real_noise_clipB_rms0.08 | 0 | 4 | 0 | 0 | 0 | 4 | yes | -/- |
| unknown_words_gap0.8 | 5 | 5 | 0 | 0 | 0 | 0 | yes | 0.27/0.34 |
| low_level_rms0.03 | 3 | 3 | 0 | 0 | 0 | 0 | yes | 0.27/0.35 |
| low_level_rms0.02 | 3 | 3 | 0 | 0 | 0 | 0 | yes | 0.24/0.33 |
| low_level_rms0.015 | 3 | 3 | 0 | 0 | 0 | 0 | yes | 0.19/0.21 |
| low_level_rms0.01 | 3 | 1 | 2 | 0 | 0 | 0 | NO | 0.55/0.55 |
| back_to_back_gap0.2 | 3 | 3 | 0 | 0 | 0 | 0 | yes (merge allowed) | 0.31/0.38 |
| digits_in_noise_snr20dB | 4 | 4 | 0 | 0 | 0 | 0 | yes | 0.36/0.48 |
| digits_in_noise_snr10dB | 4 | 4 | 0 | 0 | 0 | 0 | yes | 0.37/0.40 |
| digits_in_noise_snr5dB | 4 | 4 | 0 | 0 | 0 | 0 | yes | 0.42/0.61 |
| mixed_digit_unknown_digit | 3 | 3 | 0 | 0 | 0 | 0 | yes | 0.33/0.38 |

Recognition:

| scenario | digit words | correct | wrong label | rejected as Unknown | digit acc (of words) | non-digit words | rejected | false accept |
|---|---|---|---|---|---|---|---|---|
| single_digit | 1 | 1 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| repeat_3x_gap0.6 | 3 | 3 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| five_digits_gap0.8 | 5 | 5 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| long_pauses_3s | 4 | 4 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| unknown_words_gap0.8 | 0 | 0 | 0 | 0 | - | 5 | 4 | 1 |
| low_level_rms0.03 | 3 | 3 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| low_level_rms0.02 | 3 | 3 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| low_level_rms0.015 | 3 | 3 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| low_level_rms0.01 | 3 | 0 | 1 | 0 | 0.00 | 0 | 0 | 0 |
| back_to_back_gap0.2 | 3 | 1 | 0 | 2 | 0.33 | 0 | 0 | 0 |
| digits_in_noise_snr20dB | 4 | 3 | 1 | 0 | 0.75 | 0 | 0 | 0 |
| digits_in_noise_snr10dB | 4 | 4 | 0 | 0 | 1.00 | 0 | 0 | 0 |
| digits_in_noise_snr5dB | 4 | 1 | 3 | 0 | 0.25 | 0 | 0 | 0 |
| mixed_digit_unknown_digit | 2 | 2 | 0 | 0 | 1.00 | 1 | 0 | 1 |

## 4. Recognition accuracy on held-out test clips (single-word replay vs. direct file inference)

Replay = laser-channel path (band-pass → gate → end-of-word window → enhance → model) on a 1 s lead-in + word + 1.2 s tail stream at 256 ms pushes; direct = `predict.infer(load_audio(file))`. Clean = test-split clean clips; sim = `data/sprint8_sim/test` (agent A's simulated laser audio).

| model | set | n digit | replay acc | direct acc | one event/clip | n non-digit | replay Unknown-rejection | direct Unknown-rejection |
|---|---|---|---|---|---|---|---|---|
| deployed | clean_test | 380 | 0.947 | 0.945 | 1.000 | 20 | 0.600 | 0.650 |
| deployed | sim_test | 551 | 0.766 | 0.673 | 0.853 | 49 | 0.245 | 0.347 |
| finetuned_cnn | clean_test | 380 | 0.908 | 0.932 | 1.000 | 20 | 0.500 | 0.450 |
| finetuned_cnn | sim_test | 551 | 0.739 | 0.486 | 0.853 | 49 | 0.755 | 0.878 |
| reference_cnn | clean_test | 380 | 0.829 | 0.874 | 1.000 | 20 | 0.700 | 0.600 |
| reference_cnn | sim_test | 551 | 0.472 | 0.381 | 0.853 | 49 | 0.714 | 0.653 |
| v2_deployed | clean_test | 380 | 0.947 | 0.945 | 1.000 | 20 | 0.600 | 0.650 |
| v2_deployed | sim_test | 551 | 0.804 | 0.809 | 0.902 | 49 | 0.245 | 0.327 |
| v2_reference_cnn | clean_test | 380 | 0.829 | 0.874 | 1.000 | 20 | 0.700 | 0.600 |
| v2_reference_cnn | sim_test | 551 | 0.566 | 0.539 | 0.902 | 49 | 0.653 | 0.653 |

**deployed — sim test by condition**

| condition | n | replay acc | direct acc | replay rejected as Unknown | replay no event |
|---|---|---|---|---|---|
| sim_good | 171 | 0.807 | 0.673 | 5 | 2 |
| sim_medium | 197 | 0.766 | 0.706 | 6 | 2 |
| sim_poor | 183 | 0.727 | 0.639 | 11 | 2 |
Selection: manifest

**finetuned_cnn — sim test by condition**

| condition | n | replay acc | direct acc | replay rejected as Unknown | replay no event |
|---|---|---|---|---|---|
| sim_good | 171 | 0.778 | 0.497 | 3 | 2 |
| sim_medium | 197 | 0.797 | 0.518 | 6 | 2 |
| sim_poor | 183 | 0.639 | 0.443 | 20 | 2 |
Selection: manifest

**reference_cnn — sim test by condition**

| condition | n | replay acc | direct acc | replay rejected as Unknown | replay no event |
|---|---|---|---|---|---|
| sim_good | 171 | 0.485 | 0.368 | 17 | 2 |
| sim_medium | 197 | 0.563 | 0.406 | 23 | 2 |
| sim_poor | 183 | 0.361 | 0.366 | 45 | 2 |
Selection: manifest

**v2_deployed — sim test by condition**

| condition | n | replay acc | direct acc | replay rejected as Unknown | replay no event |
|---|---|---|---|---|---|
| sim_good | 171 | 0.825 | 0.819 | 0 | 0 |
| sim_medium | 197 | 0.838 | 0.848 | 1 | 0 |
| sim_poor | 183 | 0.749 | 0.760 | 5 | 0 |
Selection: manifest

**v2_reference_cnn — sim test by condition**

| condition | n | replay acc | direct acc | replay rejected as Unknown | replay no event |
|---|---|---|---|---|---|
| sim_good | 171 | 0.626 | 0.532 | 8 | 0 |
| sim_medium | 197 | 0.650 | 0.589 | 15 | 0 |
| sim_poor | 183 | 0.421 | 0.492 | 44 | 0 |
Selection: manifest

## 4b. Round 2 (corrected sims) — `data/sprint8_sim_v2`, `manifest_v2.csv`

Agent A regenerated the simulated clips with steady-state filter initialisation (no start-up transient), identical seeds. Same 600-clip test selection (seed 8), same laser-channel replay (fixed code), same direct inference. Round 1 columns are from `accuracy_<model>.json` (v1 sims), Round 2 from `accuracy_v2_<model>.json`. Clean-test rows are the same clips in both rounds.

| model | set | n digit | replay acc R1 → R2 | direct acc R1 → R2 | one event/clip R1 → R2 | multi-event R1 → R2 | no-event R1 → R2 | Unknown-rej. replay R1 → R2 | Unknown-rej. direct R1 → R2 |
|---|---|---|---|---|---|---|---|---|---|
| deployed | clean_test | 380 | 0.947 → 0.947 | 0.945 → 0.945 | 1.000 → 1.000 | 0 → 0 | 0 → 0 | 0.600 → 0.600 | 0.650 → 0.650 |
| deployed | sim_test | 551 | 0.766 → 0.804 | 0.673 → 0.809 | 0.853 → 0.902 | 62 → 52 | 6 → 0 | 0.245 → 0.245 | 0.347 → 0.327 |
| reference_cnn | clean_test | 380 | 0.829 → 0.829 | 0.874 → 0.874 | 1.000 → 1.000 | 0 → 0 | 0 → 0 | 0.700 → 0.700 | 0.600 → 0.600 |
| reference_cnn | sim_test | 551 | 0.472 → 0.566 | 0.381 → 0.539 | 0.853 → 0.902 | 62 → 52 | 6 → 0 | 0.714 → 0.653 | 0.653 → 0.653 |

**deployed — v2 sim test by condition (R1 → R2)**

| condition | n | replay acc | direct acc | rejected as Unknown | multi-event | no event |
|---|---|---|---|---|---|---|
| sim_good | 171 | 0.807 → 0.825 | 0.673 → 0.819 | 5 → 0 | 11 → 10 | 2 → 0 |
| sim_medium | 197 | 0.766 → 0.838 | 0.706 → 0.848 | 6 → 1 | 15 → 9 | 2 → 0 |
| sim_poor | 183 | 0.727 → 0.749 | 0.639 → 0.760 | 11 → 5 | 36 → 33 | 2 → 0 |

**reference_cnn — v2 sim test by condition (R1 → R2)**

| condition | n | replay acc | direct acc | rejected as Unknown | multi-event | no event |
|---|---|---|---|---|---|---|
| sim_good | 171 | 0.485 → 0.626 | 0.368 → 0.532 | 17 → 8 | 11 → 10 | 2 → 0 |
| sim_medium | 197 | 0.563 → 0.650 | 0.406 → 0.589 | 23 → 15 | 15 → 9 | 2 → 0 |
| sim_poor | 183 | 0.361 → 0.421 | 0.366 → 0.492 | 45 → 44 | 36 → 33 | 2 → 0 |

### Fix 3 evidence on v1 vs v2 sims (600 test clips each, deployed CNN, 256 ms pushes)

Events per clip with the fixed `_LiveClassifier` and with the pre-fix segmentation emulated (single-frame gate, tick-anchored window, no effective click rejection). `head click` = clips whose first of several events ended within 0.4 s of the clip start, i.e. the transient.

| sims | code | 0 events | 1 event | 2+ events | of which head-click |
|---|---|---|---|---|---|
| v1 | prefix code emulated | 22 | 576 | 2 | 2 |
| v1 | fixed code | 7 | 512 | 81 | 9 |
| v2 | prefix code emulated | 3 | 594 | 3 | 2 |
| v2 | fixed code | 0 | 541 | 59 | 0 |

## 5. Pipeline stage timings (software replay on the dev container — NOT Jetson latency)

cloud dev container, 2 vCPU, CPU-only PyTorch 2.15.0; torch intra-op threads = 1; one word per stream, 40 words; median / p95 over word decisions.

| run | model | push | load avg (1 min) | band-pass gate per idle tick | band-pass of word window | enhance | preprocess | features | model forward | autosave | whole predict() on the decision tick |
|---|---|---|---|---|---|---|---|---|---|---|---|
| timing_cnn_256ms | cnn | 256 ms | 2.4 | 3.60 ms | 6.9 / 12.3 | 2.3 / 3.1 | 0.2 / 0.3 | 0.5 / 0.8 | 0.9 / 1.3 | - | 19.3 / 29.1 |
| timing_cnn_256ms_autosave | cnn | 256 ms | 2.4 | 4.07 ms | 8.2 / 14.5 | 2.7 / 3.2 | 0.2 / 0.4 | 0.6 / 0.8 | 1.3 / 1.5 | 2.5 / 9.2 | 27.0 / 38.0 |
| timing_cnn_50ms | cnn | 50 ms | 2.3 | 2.23 ms | 6.5 / 13.9 | 2.2 / 3.5 | 0.1 / 0.3 | 0.5 / 0.6 | 0.9 / 1.2 | - | 16.5 / 27.9 |
| timing_ensemble_256ms | ensemble | 256 ms | 2.5 | 0.00 ms | 0.0 / 0.0 | 2.0 / 2.5 | 0.1 / 0.4 | 1.0 / 1.1 | 21.3 / 22.7 | - | 26.2 / 28.7 |

Values in ms. Other agents' training ran on the same 2 vCPUs during these runs (load average shown); with 2 torch threads under that contention the CNN forward inflated from ~1 ms to >100 ms, which is why the harness pins 1 thread. The ensemble (LSTM+CNN) forward is ~20-25 ms; the CNN alone ~1 ms. The dominant per-word cost on this machine is the zero-phase band-pass of the 1 s word window at 48 kHz (~6-11 ms).

## Regression checks

`python -m pytest scripts/sprint8/test_replay_events.py -q`:

```
16 passed in 16.37s
```

## 6. Failures found, fixes, and what was left alone

Full diffs, line numbers and before/after evidence are in `CHANGES.md`; regression checks in `scripts/sprint8/test_replay_events.py`.

**Confirmed software bugs, fixed (all in `signal_backend._LiveClassifier`, end-of-word mode):**

1. *Gate dead zone.* Only the last 0.15 s of each pushed chunk was gated; the dashboard pushes 256 ms, so ~0.1 s of every chunk was never looked at and short digits placed there were lost (81/96 → 96/96 in the placement sweep). Fix: `_gate_frames()` steps 0.15 s frames every 0.05 s over all new audio; word times come from the audio, not the GUI tick.

2. *Model window tied to the tick.* The classified window ran up to `now`, so its trailing silence (and the 1 s re-cut by `center_on_peak`) changed with push size and timer phase. Fix: `_word_window()` anchors the window to the word (±0.25 s, ≥1 s). Validation split: 0.895/0.900 → 0.930/0.930 replay accuracy at 256/50 ms pushes; labels now identical across push sizes.

3. *Click rejection never fired.* `length` was ≥ one 0.15 s frame, so `WORD_MIN_SEC=0.12` could not reject anything; clicks (and the start-up transient on every simulated clip) became digit events with ~0.8 confidence (202/600 sim clips had a second, bogus event). Fix: measure activity on 20 ms frames of the band-passed window, `WORD_MIN_SEC=0.05` (digits ≥ 0.08 s on the val split; 5–20 ms clicks 0.02–0.04 s). Sim clips with two events: 202 → 62 of 600; the remaining ones have an in-band noise floor above the gate (see below).

**Not bugs / not changed:**

- Words below the silence gate (`low_level_rms0.01`, band-RMS = 0.5× gate) are not detected — by design of the gate.

- Continuous noise ABOVE the gate never ends a word; `WORD_MAX_SEC=1.1` then forces one classification per 1.1 s. With the deployed CNN those are digit events (6/1/0 at 0.5-0.9 confidence) = false activations; the Sprint 8 checkpoints mostly answer Unknown. The gate must sit above the measured floor on the rig (as the dashboard docs already say); 62/551 sim test clips (mostly `sim_poor`) are in this regime.

- Sliding-window mode (`word_mode=False`, the std-mic channel) cannot give one event per word by design (label held until a different digit; gate on the whole 1.3 s window): 1 event for 3 repeated digits, 4 for 5 digits, duplicates in noise. The proposed `std_ensemble_word` configuration fixes that without code changes (section 2). `signal_dashboard2.py` is not C-owned, so this is a recommendation.

- Two identical consecutive words return identical `predict()` tuples; the dashboard shows no visible change for the second one. The new `on_decision` callback / `n_decisions` counter lets the UI flash or log each event.

- Timing: on this container one word costs ~16–19 ms of CPU in the CNN path (band-pass of the 1 s window ~7 ms, enhance ~2 ms, mel + forward ~1.5 ms, autosave ~2.5 ms when on), plus ~3.6 ms of gate band-pass per 256 ms push; the ensemble forward is ~21 ms. Not Jetson numbers.

**Notes for other agents:** every `data/sprint8_sim` clip starts with a full-scale ~20 ms transient (filter start-up) and the speech sits at 0.005–0.01 RMS afterwards; direct file inference centres its 1 s window on that click, which is why `direct` sim accuracy is well below the replay path for every model (0.67 vs 0.77 deployed; 0.49 vs 0.74 finetuned). Worth fixing in the simulator before trusting sim-based evaluations.

The `low_level_*` rows of the pre-fix baseline used raw-RMS levelling (the final harness levels in the 200–3800 Hz band), so only the short-word sweep and the 'missed word' counts are directly comparable between before and after.
