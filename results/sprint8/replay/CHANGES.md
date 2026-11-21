# Sprint 8 — agent C — code changes outside `scripts/sprint8/`

Every change below is in `signal_backend.py` or `src/predict.py` (both C-owned
for confirmed bugs per PLAN.md). `models/`, `src/preprocess.py`, `src/enhance.py`,
`signal_dashboard2.py` were NOT changed. All evidence comes from the software
replay harness (`scripts/sprint8/replay_harness.py`) on the cloud dev container
(2 vCPU, CPU-only PyTorch 2.15); no microphone/ADC/GUI was involved. "Before"
numbers are from the original code (`baseline/`), "after" from the final code.
Regression checks: `python -m pytest scripts/sprint8/test_replay_events.py -q`
(17 checks; `test_short_word_detected_at_every_push_phase` and
`test_model_window_is_independent_of_push_size` fail when the original
`_gate_frames`/`_word_window` behaviour is emulated — verified by monkey-patching
the old logic back in a scratch copy of the test file).

---

## Fix 1 — end-of-word gate had a dead zone of (push size − 0.15 s) per push
**File:** `signal_backend.py`, class `_LiveClassifier`: new `_gate_frames()`
(≈ lines 897-941), `_predict_word()` head (≈ 973-982), new constants
`WORD_HOP_SEC = 0.05`, `WORD_EDGE_SEC = 0.03` (≈ 863-864), new state
`self._n_gated` (init ≈ 804, `reset()` ≈ 859).

**Bug.** `_predict_word()` measured loudness only on the last 0.15 s of the
buffer at each `predict()` call. `signal_dashboard2.update_frame()` pushes one
PCM1808 read = 12288 frames @ 48 kHz = **256 ms** per tick, so the first ~106 ms
of every chunk were never inspected by the gate; a short digit falling there was
never segmented, and `_word_start`/`_last_active` were GUI-tick times, not audio
times (word start error up to one push).

**Evidence (before → after).** `replay_phase_sweep.py`: 6 real test-split digits
of 0.22–0.26 s placed at 16 onset offsets (16 ms steps) relative to the push
boundary, laser-channel settings (CNN, 200–3800 Hz, gate 0.02):

| push size | before (`baseline/phase_sweep_before.json`) | after (`phase_sweep_after.json`) |
|---|---|---|
| 256 ms (dashboard) | **81/96** placements gave exactly one event | **96/96** |
| 50 ms | 96/96 | 96/96 |

Scenario suite before the fix (256 ms, original code): `digits_in_noise_snr20dB`
missed the 0.196 s "6" (`6_yweweler_46.wav`); first probe run (−30 dB trim,
0.13 s word) missed it too. After: no missed words in any clean/noise-below-gate
scenario at either push size (section 1 of `replay_results.md`).

**Change.** The gate now steps 0.15 s frames every 0.05 s over ALL audio pushed
since the previous call (`self._n_gated` bookmark), filtering once with ≥0.3 s of
context so the zero-phase filter's edge never lands in a gated frame; word
start/end and the 0.25 s end-of-word timer are measured in frame-end time
(`t_gate`), independent of the push size. Behaviour for 50 ms pushes is unchanged
(96/96 both before and after); CPU cost per idle 256 ms push ≈ 3.4 ms (band-pass
of ≤0.6 s @ 48 kHz), ≈ 2.2 ms per 50 ms push (`timing_cnn_*.json`).

## Fix 2 — the window handed to the model depended on the GUI tick, not the word
**File:** `signal_backend.py`, new `_LiveClassifier._word_window()` (≈ 952-971),
called from `_predict_word()`.

**Bug.** `raw = buf[-(now − word_start + 0.25 s):]` ran up to `now`, i.e. it
carried 0.25–0.5 s of trailing silence whose length depended on when the push
boundary and the 0.25 s timer coincided. Windows > 1 s are then re-cut by
`preprocess.center_on_peak`, so the clip the CNN saw changed from tick to tick
(and between 256 ms and 50 ms pushes), e.g. `long_pauses_3s` word "2":
`('2', 0.987)` at 50 ms pushes vs `('unknown', 0.777)` at 256 ms after Fix 1.

**Evidence (validation split only — no test-set tuning).** 200 clean val-split
digit clips, single-word streams, deployed CNN (`window_variants_val.json`):

| window | push | replay acc | agreement with direct `predict.infer(file)` |
|---|---|---|---|
| A: ends at `now` (old) | 256 ms | 0.895 | 0.875 |
| A: ends at `now` (old) | 50 ms | 0.900 | 0.900 |
| **B: anchored to the word (new)** | 256 ms | **0.930** | **0.935** |
| **B: anchored to the word (new)** | 50 ms | **0.930** | **0.935** |

Direct file inference on the same 200 files: 0.940. After the change the
scenario suite gives identical labels at 256 ms and 50 ms pushes
(`test_model_window_is_independent_of_push_size`).

**Change.** Window = `[word_start − 0.25 s, min(now, last_active + 0.25 s)]`,
at least 1.0 s, at most `BUFFER_SECONDS`.

## Fix 3 — `WORD_MIN_SEC` click rejection could never trigger
**File:** `signal_backend.py`: new `_LiveClassifier._active_seconds()` (≈ 943-950),
`_predict_word()` (≈ 984-995), constant `WORD_MIN_SEC` 0.12 → **0.05** with the
new 20 ms-resolution measure (≈ 866).

**Bug.** `length = last_active − word_start` is at least one gate frame
(0.15 s) by construction, so `length < WORD_MIN_SEC (0.12)` was always false: a
single click/tap, or the full-scale start-up transient at the head of every
`data/sprint8_sim` clip (first ~20 ms, samples 0.55/1.0/0.77… then speech at
0.005–0.009 RMS — see "Notes for A/B"), was classified and the CNN answered a
digit ("6", confidence 0.72–0.83).

**Evidence.** Single-word replay of 600 sim test clips (deployed CNN, fixed gate
0.02): with Fixes 1+2 applied but not Fix 3, **202/600 (34 %) clips produced two
events** (first one = the click, labelled "6"), replay digit accuracy of the first
event 0.612. (With the ORIGINAL code the head click fell into the gate's dead zone
for this stream geometry and was mostly missed by luck: `replay_click_evidence.py`
emulating the pre-fix segmentation gives 2/600 multi-event clips on v1 sims but
22 clips with NO event; Fix 1 exposed the click, Fix 3 removes it.) After Fix 3,
**62/600 multi-event clips (10 %, all of them clips whose in-band noise floor
is above the 0.02 gate so the 1.1 s `WORD_MAX_SEC` cut-off splits them — addressed
by Fix 4: 11 after it) and 6 with no event; one-event rate 0.853 (0.968 after Fix 4)**, first-event
digit accuracy **0.766** (`accuracy_deployed.json`; the segmentation numbers are
identical for all three checkpoints, as they should be). Clean test clips (n=400): one-event
rate 0.9975 → **1.000**, replay accuracy 0.945 → 0.947. Duration calibration (not on test data):
val-split digits (n=500) measure ≥ 0.08 s above the gate at 20 ms resolution
(min 0.08, 1st pct 0.10); synthetic 5/20/40 ms bursts measure 0.02/0.04/0.06 s;
test-split digits min 0.06 s. Threshold 0.05 s therefore rejects clicks ≤ ~30 ms
and keeps every digit seen. Regression: `replay_phase_sweep.py` still 96/96 and
all 16 pytest checks pass. (A first attempt that counted active 0.15 s gate frames
was rejected: it dropped short-word detection to 45/96 and was reverted.)

**Change.** After the window is cut and band-passed, count 20 ms frames above
`silence_rms`; fewer than `WORD_MIN_SEC` (0.05 s) → return without a decision.
The band-passed window is now reused for the model (one fewer 1 s filter per
word, ≈ −6 ms).

## Fix 4 — after a `WORD_MAX_SEC` cut-off, the remainder of the same activity became a new "word"
**File:** `signal_backend.py`: `_gate_frames()` (start a word only if not
`_hold_until_quiet`; any quiet frame clears the flag), `_predict_word()` (sets
the flag when the cut-off fires while still active), new state
`self._hold_until_quiet` (init + `reset()`).

**Bug.** When speech/noise stayed above the gate for `WORD_MAX_SEC` (1.1 s) the
classifier decided, set `_in_word=False`, and the very next active frame opened
a new word, so continuous noise above the gate produced one decision every
1.1 s (4-5 false activations in every 5 s `noise_only_*` replay above the gate)
and a 1.3 s noisy simulated clip got a second decision for its last 0.2 s. The
old chunk-based gate hid part of this because it under-measured the active span.

**Evidence (before → after, deployed CNN, 256 ms pushes).** `noise_only_white_synthetic_rms0.08`
4 → 1 events; `noise_only_real_noise_clipA_rms0.08` 4 → 1; `clipB_rms0.04` 4 → 1;
`clipB_rms0.08` 4 → 1 (noise that crosses the gate up and down, `clipA_rms0.04`,
`clipB_rms0.02`, still gives one decision per crossing: 4-5). Sim test clips with
2+ events (600, incl. non-digit): v1 81 → 12, v2 59 → 1
(`click_evidence_v1_v2.json`). Short-word sweep still 96/96; all other scenario
results unchanged; regression check `test_continuous_activity_above_gate_gives_one_event`
(1 event for 5 s of noise above the gate; 2 events for two bursts separated by silence).
Round-1 result files produced before this fix are preserved unchanged in
`round1_fix3_snapshot/`; every file at the top level of `replay/` was re-generated
with the final code.

## Instrumentation (backward compatible, no behaviour change when unused)
**File:** `signal_backend.py` `_LiveClassifier.__init__` (new keyword args
`on_decision=None, clock=None`, ≈ 760-781), `_emit_decision()` (≈ 877-895), calls
in `_predict_word()` (≈ 1019) and sliding `predict()` (≈ 1082, 1085), counters
`n_decisions`, `n_inferences`, `last_decision`; sliding-mode throttle now calls
`self._clock()` (default `time.monotonic`) instead of `time.monotonic()` directly
(≈ 1031). Reason: `predict()` returns the same `(label, conf)` tuple for two
identical consecutive words, so a caller cannot count words from return values;
the harness needs a clock that follows the audio. Checked by
`test_decision_counter_and_callback_agree`, `test_callback_exceptions_never_break_predict`,
`test_sliding_mode_uses_injected_clock`.

**File:** `src/predict.py` (≈ lines 57-85): `set_checkpoint_path(model_type, path)`,
`checkpoint_path(model_type)` (explicit override > `LMML_CKPT_<TYPE>` env >
`config.model_checkpoint()`), used by `_load()`. Needed because
`LMML_OUTPUT_DIR` moves `config.MODELS_DIR` away from `models/`, and to run
`results/sprint8/models/*.pt` without copying anything into `models/`. The
dashboard can use it as `LMML_CKPT_CNN=results/sprint8/models/finetuned_cnn.pt`.
Checked by `test_checkpoint_override_and_env`.

## Looked at and deliberately NOT changed
- **Sliding-window mode (`word_mode=False`, the std-mic channel today).** It
  cannot give one event per word by design: a label is held until a DIFFERENT
  digit is confirmed, and the gate is the RMS of the whole 1.3 s window, which
  rarely drops below the gate between words. Replay: `repeat_3x` → 1 event,
  `five_digits` → 4, words at band-RMS 0.03 (1.5× gate) → 0 events, several
  duplicates in noise. Not a bug to patch; the proposed configuration
  `std_ensemble_word` (ensemble + end-of-word mode) gives one event per word in
  the same scenarios (`replay_results.md` §2). Recommendation for the lead:
  expose word mode for the std channel too (`signal_dashboard2.py` is not C-owned).
- **Continuous noise above the gate** (`noise_only_* ≥ 0.02–0.04` band-RMS):
  the word never ends, `WORD_MAX_SEC=1.1` forces a classification every 1.1 s,
  and whether that is "Unknown" or a digit is up to the model (deployed CNN:
  digit "6" at 0.55–0.93 on `noise_63.wav`-type clips; the ensemble says
  Unknown). The dashboard's gate must stay above the measured floor (as
  documented in `signal_dashboard2.py`); this is an operating-range statement,
  not a software bug.
- **`reset()` keeps `_last_predict_t`** (sliding throttle stamp). Harmless with
  a monotonic clock; the harness resets it itself.
- **`SignalBackend.predict()` has `_last_logged` that is never read.** Dead code,
  mic-only path, left alone.
- **`signal_dashboard2.HardwareAudioSource.read()` returns `None` (chunk
  dropped, never pushed to the classifiers) when `trigger_window` finds no
  zero crossing** — practically impossible on real noise; not C-owned; noted.

## Notes for agents A / B (data), from the replay
- Every `data/sprint8_sim/**` clip starts with a full-scale transient
  (first samples ≈ 0.55, 1.00, 0.77 …, ~20 ms) and the speech sits at
  0.005–0.01 RMS: it looks like a filter start-up transient (hp/lp with no
  initial state on a signal with DC/hum) that then becomes the peak used for
  normalisation. It (a) triggers click events in the live path (Fix 3), and
  (b) makes `preprocess.center_on_peak` centre the 1 s window on the click for
  direct file inference (direct sim accuracy 0.65 vs 0.81 through the replay
  path, which band-passes and anchors on the word). Worth fixing in the
  simulator (e.g. `sosfiltfilt`, or `lfilter` with `zi`, or trimming the
  first 50 ms) before trusting sim accuracies.
