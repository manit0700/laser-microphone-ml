# Seed replicates and update-matched control (validation only)

Runs (all under results/sprint8_followup/, nothing overwritten):
- Fine-tune (sim + clean replay, 30-epoch cap, patience 4): seed 42 = existing `finetuned_v2_cnn.pt`; seeds 43, 44 = `models/ft_s43.pt`, `models/ft_s44.pt`.
- Update-matched clean-only control (clean replay only, 63 epochs = ~12,600 updates, early stop disabled): seeds 42, 43, 44 = `models/control_matched_s42/43/44.pt`.
- Update budget check: fine-tune 30 x 419 = 12,570 updates; control 63 x 200 = 12,600 updates. Clean exposure per epoch is identical (6,391 clean replay clips); the control sees about 2x more clean passes in total because it runs more epochs.
- Logs: logs/seed_queue.log, train/ft_s*/ and train/control_matched_s*/ histories.

Protocol: validation only; temperature fitted by speaker-group 2-fold cross-fit (P_mix); frozen threshold 0.26; out-of-fold predictions scored. Script: scripts/sprint8_followup/seed_comparison_val.py -> diagnostics/seed_comparison_val.json. The test split was not read.

| arm, seed | digit acc pooled | digit acc speaker-equal | speaker-equal 95% CI (speaker bootstrap) | digit rejected | real non-digit rejected | synthetic noise rejected | OOF NLL | OOF ECE |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| fine-tune, s42 | 85.6 | 75.5 | 66.4–83.6 | 4.3 | 73.9 | 98.0 | 0.512 | 0.032 |
| fine-tune, s43 | 84.3 | 75.2 | 65.9–83.2 | 4.5 | 70.4 | 97.0 | 0.544 | 0.029 |
| fine-tune, s44 | 84.2 | 73.2 | 64.0–81.3 | 4.2 | 67.3 | 97.0 | 0.546 | 0.028 |
| control (matched), s42 | 74.5 | 62.3 | 54.6–69.6 | 8.2 | 75.7 | 90.0 | 0.792 | 0.037 |
| control (matched), s43 | 73.9 | 60.6 | 53.0–68.0 | 9.7 | 81.9 | 91.0 | 0.811 | 0.045 |
| control (matched), s44 | 74.7 | 62.4 | 54.8–69.7 | 8.5 | 77.9 | 90.0 | 0.796 | 0.043 |

(Accuracy values here are on the combined clean + simulated validation set, so they are NOT directly comparable with the clean-only validation figures in PRINCIPLES_AUDIT.md. Digit accuracy and rejection in %. Real non-digit: 226 clips from 8 speakers; synthetic noise: 100 clips.)

Paired difference, fine-tune minus matched control, speaker-equal digit accuracy (speaker-clustered bootstrap 95% CI):
- seed 42: +13.2 points [9.3, 16.6]
- seed 43: +14.6 points [10.8, 18.2]
- seed 44: +10.7 points [7.3, 14.3]
- Arm means over 3 seeds: fine-tune 74.6 (SD 1.3), control 61.8 (SD 1.0).

Findings:
1. Under an update-matched design, fine-tuning on simulated audio still gives a large gain on validation digit accuracy (speaker-equal and pooled), and the gain is consistent across seeds. The earlier claim that the gain is "from the simulation, not extra training" now has update-matched support, on validation only.
2. The matched control rejects more real non-digit speech (76–82% vs 67–74%), but it also rejects more valid digits (8–10% vs 4%). Its higher rejection is not a better trade-off; it is a more conservative threshold behaviour. Do not report the control as better at unknown rejection.
3. The fine-tuned model's real non-digit rejection drops across seeds (73.9 → 70.4 → 67.3), so this rate is seed-sensitive. Three seeds are too few for a precise estimate.
4. Calibration (OOF NLL 0.51–0.55 fine-tuned vs 0.79–0.81 control) follows the same ranking.
5. The fine-tune's advantage holds on validation, which was used for checkpoint selection in both arms (in-sample for selection; applies to both). It is not test evidence.

Not done: reference-model seed replicates (the reference was trained once), grouped speaker retraining (folds are used only for calibration and bootstrap), and any test-split evaluation of the new seeds.
