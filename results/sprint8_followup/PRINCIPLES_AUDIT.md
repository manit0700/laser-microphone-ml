# Sprint 8 follow-up: principles audit (findings, evidence, resolutions)

Scope: re-check of `results/sprint8/MODEL_PRINCIPLES_AUDIT.md` and `MODEL_PRINCIPLES_DIAGNOSTICS.json`
against saved artifacts, plus new train/validation-only diagnostics. The Sprint 8 test split was not read
by any command in this follow-up. Frozen checkpoints were not modified (sha256 of reference, continued_clean,
finetuned_v2 unchanged; deployed `models/best_model_cnn.pt` untouched).

Legend: **V** = verified in this pass by a script; **A** = accepted from the earlier audit, not re-run here;
**D** = disagreement or correction; **X** = not done.

## A. Data integrity and counts

| claim | status | evidence |
|---|---|---|
| 19,027 manifest rows; split/kind counts | V | manifest_v2.csv: train 6,391 clean + 7,000 sim; val 1,433 + 1,433; test 1,385 + 1,385 |
| 45 test "speakers" = 40 voice IDs + 5 synthetic-noise groups | V (correction of the "45 speakers" wording) | `synthetic_noise==0`: 40 test voice speakers; 45 total groups |
| train 271 / val 58 voice speakers (276 / 63 total groups) | V | same |
| one test speaker supplies 500 of 1,275 test digit clips (39%) | V | `yweweler`: 500 clean rows in test |
| every file's sha256 matches manifest (19,027) | A | not re-hashed in this pass |
| zero cross-split speaker / source hash / source filename overlaps | A | prior `split_integrity_v2.json`; not re-run |
| test: 36/60 real non-digit speech rejected (sim) = 60%; 24/60 false-accepted; 50/50 synthetic noise | V | `test_metrics_v2.json`, finetuned_v2 sim_all: unknown_rejection_rate_real_speech 0.60 (n=60), synthetic 1.00 (n=50) |
| 17 replay regression tests pass | V | re-run in this pass: `17 passed` (`logs/pytest_replay_regression.log`) |

**D1 (speaker concentration, new).** Pooled digit accuracy is dominated by one speaker. On validation,
speaker `george` (FSDD) supplies 39.4% of digit clips. Equal-weight over speaker groups gives lower numbers:

| finetuned_v2 (frozen T, thr 0.26) | pooled digit acc | speaker-equal digit acc | speaker groups |
|---|---:|---:|---:|
| train clean | 92.6 | 86.0 | 271 |
| val clean | 89.8 | 79.1 | 58 |
| train sim | 83.7 | 80.4 | 271 |
| val sim | 81.4 | 72.0 | 58 |

Source: `diagnostics/overfit_finetuned_v2.json`. The equal-weight test figure is **not** computed here:
doing so would mean reading the test split again, which this follow-up has not done. Report pooled and
speaker-equal values together in any future write-up.

## B. Overfitting and convergence (train vs validation, eval mode, no augmentation, frozen T/threshold)

| model | subset | train digit acc | val digit acc | gap (pts) | train NLL | val NLL |
|---|---|---:|---:|---:|---:|---:|
| finetuned_v2 | clean | 92.61 | 89.84 | 2.8 | 0.286 | 0.376 |
| finetuned_v2 | sim | 83.71 | 81.42 | 2.3 | 0.497 | 0.642 |
| reference | clean | 90.86 | 84.88 | 6.0 | 0.329 | 0.468 |
| reference | sim | 51.59 | 51.34 | 0.3 | 1.435 | 1.470 |

Verified: the finetuned digit accuracies reproduce the audit's 92.61/89.84 and 83.71/81.42 exactly.

**A (accepted) and D2:** the audit's uncalibrated NLL values (0.4294 clean train, 0.5096 clean val) were
re-computed with T=1 and reproduced exactly. They are uncalibrated. With the frozen T=0.572 the NLL is
0.286 / 0.376. So the audit's labelling is correct.

**D3 (overfitting verdict, refined).** The audit reports a "modest" gap. That holds for pooled clip accuracy
(2.8 points for the fine-tuned model). It does not hold cleanly for speaker-equal accuracy, where the gap is
about 7 points on clean and 8 on simulated audio. That gap is partly confounded: validation has 58 speakers
and training 271, and speakers differ in difficulty. It cannot be attributed to overfitting without
grouped cross-validation (not run). The reference model shows a 6-point pooled clean gap, which is
larger than the fine-tuned model's gap, so the fine-tune also reduces this gap relative to the reference.

**Verdict on overfitting:** the evidence supports a moderate generalization gap, consistent with some
speaker-level overfitting. It does not show severe memorization. Underfitting is not supported (train
accuracy is high). Domain mismatch to real laser audio is **not** tested here and remains the main
unresolved risk. The fine-tune stopped at its 30-epoch cap while validation loss was still falling
(audit values 1.0044 → 0.9979), so convergence is unresolved and longer training was not tested.

**X (not done):** multiple training seeds and grouped validation folds. Only one training seed exists per
arm, so seed variance and speaker-split variance are both unmeasured.

## C. Calibration (validation only, 2-fold speaker-group cross-fit)

Protocol: each held-out fold is scored with T fitted by NLL on the other fold; frozen threshold; 15 equal-width ECE bins.

| model | protocol | out-of-fold NLL | out-of-fold Brier | out-of-fold ECE | digit rejected (fold 0 / fold 1) |
|---|---|---:|---:|---:|---|
| reference | P_clean | 0.475 | 0.228 | 0.023 | 8.5% / 1.8% |
| continued_clean | P_clean | 0.438 | 0.202 | 0.035 | 9.4% / 2.8% |
| finetuned_v2 | P_clean | 0.383 | 0.170 | 0.032 | 4.8% / 2.2% |
| reference | P_mix | 0.940 | 0.424 | 0.031 | 23.4% / 10.0% |
| continued_clean | P_mix | 0.899 | 0.406 | 0.040 | 23.1% / 10.2% |
| finetuned_v2 | P_mix | 0.512 | 0.226 | 0.032 | 7.0% / 3.2% |

Source: `calibration/calibration_crossfit_val.json`.

Findings:
- Calibration protocol is a large confound for the reference model. Its NLL nearly doubles when the
  temperature is fitted on clean plus simulated validation (0.475 → 0.940), because the reference
  is poorly matched to simulated audio. Under either protocol, applied to every model, the fine-tuned
  model has the lowest NLL. The ranking is therefore not an artifact of the protocol difference.
- The fine-tuned model's out-of-fold ECE (0.032 under both protocols) is larger than the in-fit ECE the
  Sprint 8 summary quoted (0.022). The earlier figure was in-fit on the same validation data and was
  optimistic, as the audit warned. Out-of-fold ECE is the number to quote.
- Brier and NLL are reported as clearly defined; ECE depends on bin count (15 here) and sample count.
  This is not evidence of calibrated confidence on real hardware.

## D. Unknown rejection and the "six" failure

**Unknown rejection, validation only** (`unknown/unknown_rejection_dev_finetuned_v2.json`). Two
composition facts were verified: validation has 226 real non-digit speech clips (113 clean + 113 simulated)
drawn from **8 speakers**, and 100 synthetic-noise clips. The test set's 60 real non-digit clips come from
a different speaker set.

Pre-declared requirement (fixed before inspecting results): digit rejection ≤ 5% on validation; then
maximise real non-digit speech rejection; synthetic noise reported but not used to choose.

| rule | digit rejected | real non-digit rejected | synthetic noise rejected | threshold |
|---|---:|---:|---:|---:|
| frozen max-prob (baseline) | 4.2% | 73.9% | 98.0% | 0.26 |
| max-prob, selected | 4.3% | 74.3% | 98.0% | 0.27 |
| margin, selected | 3.9% | 73.9% | 98.0% | 0.00 |
| entropy, selected | 4.3% | 74.8% | 98.0% | 0.80 |

Findings:
- No alternative rule materially beats the frozen baseline on validation (differences < 1 point, on 226
  clips from 8 speakers). The frozen rule is retained. This is a null result, not a finding that the rule is
  good.
- The validation real-speech rejection (73.9%) is higher than the test figure (60%). With 8 speakers, that
  gap is well within speaker-sampling variation, and the 226-clip sample is far too small for a confident
  rate. The false-accept rate on real speech is 26% on validation and 40% on test. Either way it is high.
- The earlier audit's point stands: pooled rejection is dominated by easy synthetic noise (98–100%).
  Do not cite five replay successes as evidence of general unknown rejection.

**Six failure (dev data only).** The frozen fine-tuned model's clean digit six recall is 65.6% on validation
(86/131) and 83.3% on training. Errors go mostly to 8 (29 clips), then 7 (7) and unknown (6). The errors are
concentrated in a few speakers, not spread across the set. Speaker `george` contributes 50 of the 131
validation six clips (32 correct). Several speakers have 0 of 4–5 correct (e.g. `106a6183` 0/5, `1b18600d` 0/4).

Audio checks (inspection, not listening):
- Sample rate is mixed (8 kHz and 16 kHz) among both correct and wrong six clips. Sample rate does not explain it.
- Six clips predicted as 8 are short and quiet (median 0.56 s, median peak 0.25) versus correct six clips
  (median 1.0 s, peak 1.0). Across all validation digits, however, short clips score higher on clean audio
  (92.8% vs 87.8%, `diagnostics/duration_effect_val.json`), so duration is not a general cause. The six→8
  pattern is specific, with only 29 clips.
- Label correctness has **not** been verified by listening. This is an open check. A mislabelled or
  clipped/trimmed recording is still possible for some of these clips.

## E. Physics, Level 1 / Level 3, streaming (not re-researched in this pass)

- **Level 1 / Level 3:** no definition exists anywhere in the project or the Sprint reports
  (searched the Mac project and report folders for "Level 1" / "Level 3"; the only hit is the earlier audit
  itself). They could be laser-safety classes, device power settings or team-defined configurations. The
  specification is **missing**. No Level 1 vs Level 3 scenario has been modelled or claimed. Definition
  requested from the team (see HANDOFF §5).
- **Physics assumptions:** the earlier `research/assumptions_review.md` was not re-audited in this pass.
  Its main conclusion stands as a limitation: the simulator's 0–30 dB SNR prior is an optimistic detector-stage
  parameter, not the measured in-band SNR (rig ~3–7.5 dB per earlier notes). The `sim <10 dB` column is a
  simulator-parameter bin, not a measured rig condition.
- **Normalization:** the simulator and the ADC stage normalise amplitude, so absolute laser power is not
  recoverable from the normalised audio. No mapping from normalised amplitude to watts is claimed.
- **Streaming:** the 17 replay regression checks were re-run in this pass and all pass (`logs/pytest_replay_regression.log`).
  No segmentation or filtering code was changed in this follow-up, so the replay results remain as recorded in
  `results/sprint8/replay/`. The one-duplicate-in-600 result is not a universal one-event guarantee, and
  the regression checks are unit-level; they do not replace a live stream test.

## F. Test independence (disclosure)

Round 1 and Round 2 both scored the same original test speakers. Round 2 changed the simulated inputs, not
the speaker identities. The Sprint 8 test split is therefore **not** pristine and cannot be used again for a
confirmatory claim. A confirmatory result requires new speakers or a recording session, none of which exist
in this environment.

## G. Resolutions and what changed

- No model, threshold, temperature or preprocessing was changed. The frozen Round 2 settings stand.
- New evidence files were added (diagnostics, calibration cross-fit, rejection comparison, duration check).
- Reporting corrections recommended: use "40 voice speakers + 5 synthetic-noise groups"; quote out-of-fold
  ECE (0.032), not the in-fit 0.022; report pooled and speaker-equal digit accuracy together; state that
  real-speech rejection is 60% on test with 8-speaker validation evidence elsewhere.
- Not resolved: overfitting verdict under speaker-equal weighting (needs grouped CV); six errors (needs
  listening); unknown-rejection generalization (needs new speakers and real rig data); Level 1/3 (needs definitions).

## H. Verdict

Supported: a moderate train-to-validation gap; no sign of severe memorization in pooled metrics; a stable
calibration advantage of the fine-tuned model under two protocols; no validation-selectable unknown rule
that beats the frozen baseline.
Not supported: severe overfitting; underfitting; any real-rig claim; generalization of unknown rejection;
any Level 1 / Level 3 performance difference.
Unresolved: speaker-equal generalization gap; the six failure's cause; convergence at the epoch cap.

## I. Addendum (after seed replicates and the update-matched control)

Full details: `SEED_AND_CONTROL_RESULTS.md`, `diagnostics/seed_comparison_val.json`, `figures/reliability_oof_val.png`,
`PHYSICS_RECHECK.md`, `listening_six_review/`.

- **Update-matched control (resolves audit item "control not update-matched").** Clean-only control with ~12,600
  updates (63 epochs, early stop disabled) vs fine-tune with 12,570 updates. Validation, 3 seeds each: fine-tune
  speaker-equal digit accuracy 74.6% (mean; SD 1.3 across seeds) vs control 61.8% (SD 1.0). Paired speaker-clustered
  95% CIs for the difference exclude zero for every seed (+10.7 to +14.6 points). So the simulation gain is not
  explained by extra training updates. Clean exposure per epoch is equal; total clean passes differ (see the file).
- **Control's non-digit behaviour.** The control rejects more real non-digit speech (76–82% vs 67–74%) but also rejects
  8–10% of digits (fine-tune ~4%). This is a conservative threshold trade-off, not better unknown rejection.
- **Seed sensitivity of unknown rejection.** Fine-tune real non-digit rejection falls across seeds (73.9, 70.4, 67.3%).
  Three seeds cannot give a precise rate.
- **Speaker-equal vs pooled (confirmed).** Both arms show the pooled-vs-equal-weight gap. Speaker-clustered bootstrap
  CIs are wide (about ±9 points for a 58-speaker validation set), so speaker concentration is real but its exact size is
  imprecise.
- **Calibration (out-of-fold, reliability diagrams).** Fine-tune OOF NLL 0.51–0.55 vs control 0.79–0.81; OOF ECE
  0.028–0.032 vs 0.037–0.045. Reliability bars are in the figure; the diagram shows modest over-/under-confidence per bin,
  not a clear systematic miscalibration.
- **Physics (primary-source check).** PCM1808: 3 Vp-p full scale, 99 dB A-weighted DR, 1 µF AC coupling (2.7 Hz).
  Walker & Saxena 2022 measured recovery SNR 1.2–10 dB, supporting the view that the simulator's 0–30 dB prior is
  optimistic. Power-line interference: odd harmonics dominate in an electrical measurement, which may conflict with the
  simulator's all-harmonic hum. Level 1 / Level 3: still no specification.
- **Six review.** The listening playlist and blank label sheet are exported. No listening has been done. Labels are
  unchanged until a human verifies them.
- **Overfitting verdict (updated).** Pooled gap small; speaker-equal gap moderate and partly confounded. Seeds and
  matched controls show the fine-tune's gain is robust across three seeds. Grouped retraining was not done, so
  overfitting versus speaker difficulty remains unresolved.
- **Test split.** Still not re-read in this pass. No speaker-level test metrics computed (no saved per-clip predictions).
