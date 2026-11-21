# Sprint 8 — simulation and software-prototype summary

Team Eclipse · Laser Microphone · ML/data pipeline · work executed 2026-11-08 (UTC) in a cloud dev
container (2 vCPU, CPU-only PyTorch 2.15); the full experiment record is `docs/EXPERIMENTS.md` → Exp 7.
Every number below is read from a saved file under `results/sprint8/` (paths in the completion table).

## What was asked (five tasks, planning estimates only)

1. Fine-tune the laser CNN on simulated laser audio.
2. Evaluate on held-out speakers and update the confusion matrix.
3. Calibrate confidence and tune the unknown threshold.
4. Verify one prediction per spoken word and unknown rejection through software replay.
5. Document the experiment and Sprint 8 results.

## Supported results

**Data integrity.** 9,682 local recordings → 9,209 unique (473 duplicates removed). Originals were split
by speaker before any simulation (train 6,391 clips / 276 speakers; val 1,433 / 63; test 1,385 / 45;
team speaker `manit` entirely in test). 9,818 simulated variants were generated with
`scripts/physics_laser_sim.py` from those originals, each inheriting its source's split, with per-clip
seeds and parameters in the manifest. Integrity check: 0 speakers, 0 source hashes and 0 sources shared
between splits (`data/split_integrity_v2.json`).

**Baseline provenance.** The deployed `models/best_model_cnn.pt` was trained on a Kaggle fold of a
44.8 K-clip superset of the local sources and then fine-tuned; speaker separation from the local test
set cannot be established. A clean-audio reference CNN was therefore trained from scratch on the train
split and a copy fine-tuned. The deployed model is reported for information only.

**Fine-tuning on simulated laser audio works, and the gain is from the simulation, not from extra
training** (test split, 45 unseen speakers, settings frozen on validation, scored once):

| digit accuracy, % | clean | sim ≥20 dB | sim 10–20 dB | sim <10 dB | all sim |
|---|---|---|---|---|---|
| reference (clean only, from scratch) | 89.1 | 55.9 | 56.5 | 48.5 | 53.7 |
| control: reference + more clean-only training | 90.9 | 58.3 | 56.0 | 48.0 | 54.3 |
| **reference fine-tuned on simulated laser** | **91.5** | **80.7** | **84.4** | **76.5** | **80.5** |
| deployed checkpoint (info only, not speaker-disjoint) | 94.1 | 83.8 | 87.7 | 81.6 | 84.4 |

Digits wrongly rejected as unknown: 2.1 % (reference) and 1.3 % (fine-tuned) on clean, 19.1 % → 4.2 %
on simulated audio. Unknown rejection on simulated audio 76.4 % → 78.2 % (synthetic noise ≈100 %, real
non-digit speech 60 %). Balanced 11-class accuracy on simulated audio 55.6 % → 80.3 %. All four
pre-registered acceptance criteria passed. Full tables and 24 confusion matrices:
`eval/test_metrics_v2.md`, `eval/plots_v2/`.

**Calibration.** Temperature 0.572 (ECE on validation 0.177 → 0.022) and unknown threshold 0.26 for the
fine-tuned model, fitted on validation clean + simulated clips only (`eval/calibration_val_v2.json`,
`eval/plots_v2/reliability_finetuned_v2.png`). The fixed 0.60 threshold in `config.py` would reject
far more real digits under label smoothing (validation digit accuracy 85.6 % at 0.26 vs 77.4 % at 0.60).

**Simulator defect found and fixed.** Every simulated clip started with a full-scale ~20 ms filter
start-up transient (AC-coupling high-pass applied from zero state to the ≈0.5 DC photodiode level), and
the ADC stage normalised to that click, leaving speech at 0.005–0.1 of full scale. Fixed in
`physics_laser_sim.electronics()` (steady-state filter initialisation); regenerated with identical
seeds, the click is gone (0 % of clips peak in the first 30 ms, speech RMS 0.056 → 0.162). The artifact
alone cost the reference model 5.3 points on simulated digits. Earlier simulated-laser results (Exp 2,
Exp 6) were produced with the clicky simulator.

**Prototype software replay** (real test-split clips streamed into the dashboard's `_LiveClassifier`,
no microphone/ADC/GUI): one decision per spoken word in every deterministic scenario — single digit,
same digit ×3 → 3 events, 5 digits, 3 s pauses, 5 s silence, noise below the gate, 5 non-digit words →
5 Unknown events, back-to-back 0.2 s gaps, digits in noise at 20/10/5 dB — at both 256 ms (one PCM1808
read) and 50 ms pushes. Four confirmed segmentation bugs were fixed in `signal_backend.py` with
before/after evidence and 17 passing regression checks (`replay/CHANGES.md`). On click-free simulated
test clips: one-event rate 0.998, 0 missed words, 1 duplicate in 600; replay accuracy equals direct
file inference within a few points.

**Physics review** (`research/assumptions_review.md`, 32 sources). The simulator has the right physical
forms at each stage but is miscalibrated against the only rig measurements that exist: SNR prior 0–30 dB
vs rig ≈3–7.5 dB in-band; hum harmonics decay too slowly (≈9 % of hum power leaks into the speech band
vs ≤0.15 % bounded on the rig); surface resonances placed in-band fit membranes, not window glass;
27 % of clips clip where the rig has not been seen to clip. Only the `sim <10 dB` column should be read
as indicative of the rig, and even that is optimistic.

## Negative or cautionary results (kept)

- Continued clean-only training did not help on simulated audio (+0.6 pts) — the control.
- Fine-tuned models do not beat the reference on the 92 team-speaker clips (85.9 vs 87.0 %, within noise).
- Digit "6" remains weak (reference 59.7 % clean; 0/9 on `manit`), consistent with earlier sprints.
- Real non-digit speech is rejected only 55–68 %; the unknown class needs more real speech.
- Fine-tunes ended at their epoch cap while still improving; the reference training set (6.4 K clips) is
  5× smaller than the Kaggle set, so these are controlled comparisons, not best achievable accuracy.
- Sliding-window mode (std channel) cannot give one event per word by design; continuous noise above
  the gate produces one forced classification per 1.1 s.

## Limitations

- All laser audio is **simulated**; no real laser recordings were made or scored. "Simulated conditions"
  are SNR bins drawn by the simulator, not recording sessions.
- Timings are from the dev container (2 vCPU, CPU PyTorch), not the Jetson; no FP16/TensorRT measurement.
- The dashboard GUI (`signal_dashboard2.py`) was not run; the replay drives `_LiveClassifier` directly.
- The deployed checkpoint's numbers are not comparable (possible speaker overlap with the test set).

## Remaining hardware tasks

1. Measure the rig: in-band SNR histogram over ≥50 paired std/laser utterances; a 10 s quiet spectrum
   (beam on / beam blocked / laser off) for hum harmonics; swept-sine response of the actual target
   surface; receiver bandwidth and clipping behaviour (`research/simulator_vs_rig.md` §C).
2. Replace the simulator's priors with those measurements (`--calibrate-from` path) and re-fine-tune.
3. Record real team-voice digits and non-digit speech through the laser; score the Sprint 8 fine-tuned
   model and the deployed model on them (per-digit detection rate and accuracy separately).
4. Run the fixed `_LiveClassifier` on the Jetson with the real stream and confirm the one-event-per-word
   behaviour and the gate level; measure latency there.
5. Expose end-of-word mode for the std channel in `signal_dashboard2.py`.

## Compute time actually spent (wall clock, not human hours)

Dataset build 1.3 + 1.8 min; reference training 23.6 min; fine-tunes 12.3 + 25.9 + 12.3 min;
control 5.4 min; calibration 3.4 min; tests 3.0 min; replay suites ≈ 60 min including re-runs;
physics research ≈ 15 min of agent time. Total ≈ 2.5 h of container time across four parallel agents.
