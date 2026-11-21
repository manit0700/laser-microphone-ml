# Sprint 8 follow-up: handoff

Scope: independent check of the Sprint 8 model-principles audit, plus validation-only diagnostics.
No model was retrained. The Sprint 8 test split was not read. Deployed and frozen checkpoints are unchanged.

## 1. Reproduce what was run (≈ 6 min total on 2 CPU cores; logits are cached after the first run)

```bash
cd /home/claude/lmml                       # cloud copy of the project; on the Mac, the project root
PY=/home/claude/.venv-lmml/bin/python      # any python with torch, numpy, scipy, soundfile, pandas, pytest
$PY scripts/sprint8_followup/diagnose_overfit.py finetuned_v2 reference
$PY scripts/sprint8_followup/calibration_crossfit.py reference continued_clean finetuned_v2
$PY scripts/sprint8_followup/unknown_rejection_dev.py finetuned_v2
LMML_OUTPUT_DIR=$PWD/results/sprint8_followup/logs $PY -m pytest scripts/sprint8/test_replay_events.py -q
```
Logits cache: `results/sprint8_followup/cache/` (regenerable, not part of the transfer). Environment: torch 2.15.0 CPU,
Python 3.13, `cpu_count` 2. Seeds: none needed for the deterministic evaluations; training seed 42 in the earlier runs.

## 2. Report-ready summary (verified unless marked)

- Pooled train-to-validation digit gap is small for the frozen fine-tuned model (92.6→89.8 clean, 83.7→81.4 simulated).
  Speaker-equal gap is larger (86.0→79.1 clean, 80.4→72.0 simulated; 271 vs 58 speakers, partly confounded).
- One validation speaker supplies 39% of digits; pooled accuracy overstates equal-weight accuracy by about 10 points.
- Out-of-fold (speaker-group cross-fit, validation only) ECE of the fine-tuned model is 0.032, versus 0.022 quoted
  earlier from an in-fit estimate. The fine-tuned model has the lowest NLL under both calibration protocols.
- Unknown rejection: the frozen rule rejects 74% of real non-digit speech on validation (8 speakers), 60% on the
  test split (24/60 false accepts). No validation-selectable rule beat it. Synthetic noise rejection is 98–100%
  and should not be used to describe speech rejection.
- Digit "six" errors are speaker-concentrated (e.g. one speaker 0/5 correct), mostly predicted as "eight".
  Sample rate does not explain them. Label correctness is not yet checked by listening.
- Level 1 / Level 3 definitions are not in the project. No Level 1 vs Level 3 claim is made.

Recommended wording: "On speaker-separated simulated laser audio, the frozen fine-tuned model reached 80.5% digit
accuracy on held-out speakers (pooled; 40 voice speakers plus 5 synthetic-noise groups in the test split). The
pooled train-validation gap was 2.8 points on clean digits; equal-weight speaker accuracy was lower. Non-digit speech
was rejected 60% of the time on the test split, and the evidence does not show this generalizes to real speech. No
real-laser performance is claimed."

## 3. Remaining work that needs compute (not started)

| item | what to run | rough cost on 2 cores |
|---|---|---|
| Seed replicates (training variance) | `scripts/sprint8/finetune_sim.py --manifest results/sprint8/data/manifest_v2.csv --epochs 30 --seed <S> --out-ckpt results/sprint8_followup/models/ft_seed<S>.pt --out-dir results/sprint8_followup/train/ft_seed<S>` (`--seed` exists; default epochs is 15, so pass 30; write only to NEW paths, never to `results/sprint8/models/`) | ~12–26 min per run |
| Update-matched control | clean-only continuation with the same optimizer-update count as the fine-tune (≈12,570 updates, not 1,200) | ~25 min |
| Grouped speaker CV for validation | k-fold over speaker groups on the train+val pool (not the test split) | several hours |
| Reliability diagrams | plot from `calibration/calibration_crossfit_val.json` (add a plotting step) | minutes |

Do these before any new claim about overfitting or matched gains. Each arm needs its own hypothesis entry in `docs/EXPERIMENTS.md`.

## 4. Remaining data work (needs people, not compute)

- Listen to the 29 "six→eight" validation clips and the 16 other six errors to confirm labels, clipping and trimming.
- Collect non-digit speech from more speakers (current validation evidence: 8 speakers) and realistic rig noise,
  hum, taps and silence. Only then can unknown rejection be claimed beyond the existing 60 test utterances.
- A fresh confirmatory set (new speakers or recording sessions) is required for any confirmatory claim. The Sprint 8
  test split is no longer pristine (Round 1 and Round 2 used it).

## 5. Blocked inputs (exact need)

| blocked item | input needed | how to resume |
|---|---|---|
| Level 1 / Level 3 meaning | the team's definition: safety class, device power setting, or team-defined config. If a safety class, the datasheet with wavelength, power limit and class | Send the definition or the datasheet; then add a hypothetical scenario table in `research/` |
| Real laser performance | recordings from the rig (`scripts/record_daq_wav.py`, see `results/sprint8/HANDOFF.md` §5) | unchanged from the earlier handoff |
| Real-speech unknown rejection | new non-digit speech and rig noise | record and add to the manifest as a new split; do not reuse Sprint 8 test speakers |
| Jetson / FP16 / TensorRT | the Nano and the export script | unchanged |

## 6. Caveats

- Frozen settings were not changed. Calibration and rejection results here are validation-only and do not replace the
  Round 2 test result, which is historical.
- Compute runtime (≈ 6 min container time) is separate from human work hours, which are not recorded.
- Round 2 test metrics are not re-scored here. Speaker-equal test metrics are not reported for the same reason.
- No commits, pushes, submissions or report edits were made.

## 7. Update (after seed replicates, matched control, physics re-check, revised report)

New scripts (all under scripts/sprint8_followup/):
- `run_seed_queue.sh`: update-matched controls (seeds 42–44, 63 epochs, early stop off) and fine-tune seeds 43–44. Runs finished with rc=0 (logs/seed_queue.log).
- `reliability_plots.py`: out-of-fold reliability diagrams (figures/, calibration/reliability_bins_oof_val.json).
- `export_six_review.py`: listening playlist and blank label sheet (listening_six_review/). No labels changed.
- `seed_comparison_val.py`: 3-seed validation comparison with speaker-clustered bootstrap (diagnostics/seed_comparison_val.json). Validation only.

Reproduce the seed queue: `bash scripts/sprint8_followup/run_seed_queue.sh` (about 1 hour on 2 cores; writes only under results/sprint8_followup/).

Revised report: `/home/claude/report8/Manit_Dankhara_Sprint_8_Followup_Report.tex` (and .pdf). The original report is unchanged. Sprint dates, actual hours and peer-review fields are still open, as in the original.

Still open:
- Listening review of the six clips (human task; labels unchanged until verified).
- Grouped speaker retraining (not run).
- Level 1 / Level 3 definitions (blocked on the team).
- Real rig data and a fresh confirmatory test set (blocked).
- Speaker-level test metrics (not computed; no saved per-clip test predictions, and the test split was not re-read).
