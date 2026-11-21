# Sprint 8 independent model-principles audit

Date: 2026-11-08. Scope: local saved Sprint 8 artifacts, data construction,
training/calibration/evaluation code, replay code and existing physics review.
No models were trained, tuned, deployed or changed during this audit. Final
test scores were read from saved results, not re-scored. New diagnostics use
training/validation data only, in evaluation mode without augmentation.

## Verdict

No strong evidence of severe conventional overfitting in the final CNN on the
current simulated-data distribution. This is a credible simulation experiment,
not evidence that the model is ready for the real laser rig. Moderate generalization
gaps, weak unknown-speech rejection, a substantial per-speaker/per-digit failure,
and experimental-design limitations remain.

## Independent verification performed

- All 19,027 manifest rows: files present and SHA256 matches recorded value.
- Zero cross-split overlaps in speaker/group IDs, source SHA256, or source filename.
- Zero exact decoded-PCM duplicates across splits (sample rate and shape included).
  This does not exclude near duplicates, rescaled/re-encoded versions, or incorrect
  speaker metadata.
- All final checkpoint hashes match the frozen evaluation records. Training
  summaries contain earlier hashes because calibration subsequently saves metadata
  into checkpoints; the calibration records bridge these before/after hashes.
- Re-ran `python -m pytest scripts/sprint8/test_replay_events.py -q`: 17 passed.
- Evaluated the frozen fine-tuned CNN on train and validation with identical
  preprocessing, dropout disabled, no augmentation, stored T=0.572184 and
  threshold=0.26. Saved in `MODEL_PRINCIPLES_DIAGNOSTICS.json`.

| Data | Train digit accuracy | Validation digit accuracy | Gap |
|---|---:|---:|---:|
| Clean | 92.61% | 89.84% | 2.77 percentage points |
| Simulated | 83.71% | 81.42% | 2.29 percentage points |

Overall 11-class accuracy: train/val clean 91.86/89.18%; train/val sim
84.07/81.09%. Uncalibrated NLL: clean 0.429/0.510; sim 0.636/0.737.
There is some generalization gap, but not near-perfect memorization with a
collapse on unseen speakers. Augmented training accuracy in the original logs
is not directly comparable to unaugmented validation accuracy.

Final five fine-tune validation losses: 1.0044, 1.0095, 1.0035, 1.0016, 0.9979.
There is no sustained late validation-loss rise. Epoch 30 was selected at the cap;
this shows training was not demonstrably converged, not that longer training
is guaranteed to improve generalization. Reference selected epoch 38/40.

## Principle-by-principle findings

### Data separation and provenance — passes checked exact-overlap tests

Originals are speaker-grouped before simulation; derivatives inherit their
source split. Digit and non-digit Speech Commands speaker IDs are grouped together.
The new reference model starts from scratch. The deployed checkpoint is correctly
excluded from clean comparative claims because its training provenance overlaps
potentially with local test sources. No cross-split learned normalization was
found: feature normalization is per clip.

Reporting correction: the claimed 45 test speakers are 45 groups, including five
synthetic-noise groups; there are **40 voice-speaker IDs**. Test digits also have
unequal speaker representation: one FSDD speaker contributes 500/1,275 digit clips.
Pooled clip accuracy therefore does not measure equal-weight performance per speaker.

### Capacity and regularization — reasonable, not a proof against overfitting

Small CNN with dropout 0.4, weight decay, augmentation, label smoothing, validation
checkpoint selection and clean replay are defensible. BatchNorm/dropout are put
in evaluation mode for scoring. Final matched train/val checks support the modest-gap
verdict above. Only one training seed and one split were run: seed robustness and
speaker-group cross-validation remain unmeasured. Report speaker-level macro
accuracy and speaker-clustered uncertainty; independent clip-binomial intervals
would understate dependence among clips from one speaker.

### Calibration and decision rule — validation-only, with important limits

Temperature/threshold are fitted on validation, not test. However, that validation
set is also used for checkpoint selection and reporting ECE. The reported
ECE 0.177 to 0.022 is an in-fit calibration result, not independent proof of calibrated
confidence on new speakers or the rig. Use a separate calibration subset or grouped
cross-fitting in a future experiment. Test calibration was not saved in this audit.

Reference calibration uses clean validation only, while the fine-tuned model uses
clean plus simulated validation. Thus the final decision comparison includes both
weight adaptation and different calibration exposure. For a stronger ablation,
apply the same calibration protocol to every comparison model and report raw
argmax accuracy as well. Do not retune on the current test results.

### Unknown rejection — material weakness

Saved simulated test results: fine-tuned rejection of real non-digit speech is
36/60 = 60%; 24/60 non-digit utterances are falsely accepted as digits. Synthetic
noise rejection is 50/50. Their pooled 78.2% rejection masks this difference.
The four acceptance criteria do not impose a dedicated non-digit false-accept
limit. Balanced 11-class accuracy can improve even if unknown rejection worsens.
Define an application-appropriate false-accept limit before future evaluation and
collect diverse real rig noise and non-digit speech. Five successful curated replay
words do not override the broader 60-utterance test result.

### Per-class and per-speaker behavior — aggregate scores hide failures

Fine-tuned clean digit six accuracy is 69.4%; on the held-out `manit` recordings
it is **0/10**. Your overall clean accuracy drops from reference 87.0% to 85.9%
(92 clips). Simulated digit nine accuracy is 68.3%. Investigate with training and
validation examples first. Any further changes motivated by these test failures
must be assessed on a fresh held-out test set.

### Control experiment and causal claims — useful but not matched

The clean-only control selected epoch 6 with 6,391 clips/epoch (~1,200 optimizer
updates); the simulation run selected epoch 30 with 13,391 clips/epoch (~12,570
updates). The control used v1 simulated validation for checkpoint selection and
calibration; final fine-tuning used corrected v2 validation. Consequently it is
too strong to say the control proves all 27 points come only from simulation.
It supports that conclusion but does not isolate it completely. A future comparison
should match optimizer-update budget, validation/calibration conditions, seed
replicates and clean exposure. Preserve the existing control as exploratory evidence.

### Test independence across rounds — disclose reuse

There are round-1 and round-2 results on the same original test speakers. Replacing
defective simulated waveforms changes their inputs, not their speaker identities.
Settings frozen before each round support procedural discipline, but do not make
round 2 a globally untouched test set. No direct test-driven weight fitting was
found in the inspected code. Describe this as a corrected-simulator re-evaluation,
with no test-based hyperparameter selection claimed. Reserve fresh speakers or
recording sessions for the next confirmatory result.

### Physics, SNR and simulator transfer — not validated on hardware

The existing research review identifies mismatched SNR, hum, surface modes and
clipping priors. Further, `snr_db` is a sampled detector-stage parameter; it excludes
later hum, flicker, filtering and clipping. It is not a measured final in-band SNR.
Shot/electronic noise mixing also does not enforce that sampled value as exact total
noise power. Label the bins as simulator parameter bins. The <10 dB result cannot
be interpreted as an estimate of real-rig accuracy.

The simulator normalizes source peak and optical displacement and applies per-clip
ADC scaling. These are useful domain-randomization choices but do not preserve an
absolute physical mapping from laser watts to displacement, detector volts, or
speech SNR. It cannot substantiate Level 1 versus Level 3 laser-performance claims
without hardware specifications and calibration. No separate level comparison is
present in the inspected final metrics.

Only one simulation variant per test source is scored; training/test use the same
simulator and parameter distributions. Unseen-speaker success is not unseen-device,
unseen-surface, or out-of-distribution physics success. Use measured paired rig data
and held-out hardware/session conditions for transfer claims.

### Streaming behavior and deployment — regression checks pass, scope limited

The 17 event-handling regression tests pass locally. The saved larger replay suite
still reports one duplicate in 600 simulated clips: avoid universal one-event claims.
Replay normalizes prepared utterances to controlled amplitude and adds known gaps,
so it tests segmentation under controlled conditions, not real rig level stability.
Continuous above-gate noise still forces periodic classifications.

Offline simulated preprocessing resamples to 8 kHz before band-pass/enhancement;
live filtering runs on capture-rate blocks before resampling, with different block
edges/windowing. These paths are approximately aligned, not numerically identical.
The replay comparison is useful evidence of the difference; GUI/ADC/hardware and
Jetson performance remain unverified. Changing to causal filtering requires matching
training and validation preprocessing, not merely swapping a deployment filter.

## Recommended report wording

"On speaker-separated simulated laser audio, digit accuracy improved from 53.7%
to 80.5%. Matched train/validation checks showed gaps of 2.8 points on clean digits
and 2.3 points on simulated digits, with no sustained late validation-loss increase.
These findings do not show severe overfitting within the tested simulation
distribution. Real-laser performance remains unvalidated. Non-digit speech rejection
and recognition of six on one held-out team speaker remain significant limitations."

Prefer "40 voice-speaker IDs plus five synthetic-noise groups" to "45 unseen
speakers", and "corrected-simulator re-evaluation" to an unqualified claim that
the test set was used only once across the entire development process.

## Primary guidance checked

- scikit-learn, data leakage and consistent preprocessing:
  https://scikit-learn.org/stable/common_pitfalls.html
- scikit-learn, training/validation curves and model selection:
  https://scikit-learn.org/stable/modules/learning_curve.html
- Google ML Crash Course, interpreting overfitting loss curves:
  https://developers.google.com/machine-learning/crash-course/overfitting/interpreting-loss-curves

Existing physics references were reviewed as project evidence; this audit does
not independently certify every numerical physical claim in that research report.
