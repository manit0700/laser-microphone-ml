# Sprint 8 follow-up completion table

Status: **done** = completed and verified by a saved artifact; **partial** = done with a stated gap;
**not done** = not attempted in this pass; **blocked** = needs something outside this environment.
Compute runtime is the container wall time of each step; human hours are not recorded here.

| # | Requirement (from the follow-up request) | Status | Evidence (paths under results/sprint8_followup/ unless noted) |
|---|---|---|---|
| 1 | PLAN.md with ownership, dependencies, acceptance criteria | **done** | `PLAN.md` |
| 2 | PRINCIPLES_AUDIT.md with evidence-backed findings and resolutions | **done** | `PRINCIPLES_AUDIT.md` (V/A/D/X tags on every claim) |
| 3 | Verify the audit's data-integrity and count claims | **partial** | manifest counts, speaker groups, 40 voice IDs, yweweler share, rejection counts verified; 19,027-file hash re-check and cross-split overlap re-run accepted from the audit, not re-run |
| 4 | Overfitting diagnostics: train/val, eval mode, clean and sim, per-class, per-speaker, per-dataset | **done** (1 seed) | `diagnostics/overfit_finetuned_v2.json`, `diagnostics/overfit_reference.json`; `scripts/sprint8_followup/diagnose_overfit.py` |
| 5 | Multiple training seeds and grouped validation folds | **partial** | 3 seeds per arm done (`SEED_AND_CONTROL_RESULTS.md`); grouped speaker retraining **not** done (folds used for calibration/bootstrap only) |
| 6 | Matched-control comparison (update-matched, exposure-matched) | **done** (update-matched, 3 seeds, validation) | `diagnostics/seed_comparison_val.json`; exposure differs per epoch by design (noted) |
| 7 | Calibration verified independently (cross-fit, NLL, Brier, ECE, same protocol for all models) | **done** | `calibration/calibration_crossfit_val.json`; `scripts/sprint8_followup/calibration_crossfit.py`; reliability diagrams **not** produced (JSON only) |
| 8 | Unknown rejection: compare rules on validation, pre-declared requirement | **done** (null result) | `unknown/unknown_rejection_dev_finetuned_v2.json`; `scripts/sprint8_followup/unknown_rejection_dev.py`; no rule beat the frozen baseline |
| 9 | Unknown rejection: broaden non-digit speech and rig noise | **not done** | requires new speech and real recordings (HANDOFF §4); seed variation measured (67–74%) |
| 10 | "Six" failure investigation | **partial** (listening playlist exported, not yet listened) | confusion, speaker concentration, duration and sample-rate checks done (PRINCIPLES_AUDIT §D); label correctness by listening **not** done; preprocessing/trimming check limited to duration and peak |
| 11 | Speaker-equal and per-source test metrics | **partial** | speaker-equal for train/val and 3-seed validation with speaker-clustered bootstrap; test not re-read (no saved per-clip predictions) |
| 12 | Physics assumptions review with original sources | **done** (highest-impact items) | `PHYSICS_RECHECK.md` (PCM1808 datasheet, Walker & Saxena 2022, power-line interference paper) |
| 13 | Level 1 / Level 3 definitions | **blocked** (missing specification) | no definition found in project or reports (search recorded in PRINCIPLES_AUDIT §E); definition needed from the team |
| 14 | Streaming and replay regression | **done** (unit-level only) | `logs/pytest_replay_regression.log` (17 passed); no replay re-run needed (no segmentation changes) |
| 15 | Reproducible scripts and exact commands | **done** | `scripts/sprint8_followup/{common,diagnose_overfit,calibration_crossfit,unknown_rejection_dev}.py`; `COMMANDS.md` |
| 16 | Separate checkpoints | **not done** (no new checkpoints) | none trained; frozen checkpoint hashes unchanged (`logs/checkpoint_hashes_after_followup.txt`) |
| 17 | Learning curves and validation-loss trends | **accepted** | from earlier `results/sprint8/train/finetune_v2/` (not re-plotted) |
| 18 | Reliability diagrams and calibration plots | **done** | `figures/reliability_oof_val.png`, `calibration/reliability_bins_oof_val.json` |
| 19 | COMPLETION_TABLE.md | **done** | this file |
| 20 | HANDOFF.md | **done** | `HANDOFF.md` |
| 21 | Concise report-ready summary | **done** | `PRINCIPLES_AUDIT.md` §Recommended wording (see HANDOFF §2) |
| 22 | Preserve deployed models and Sprint 8 evidence; no test-set reads | **done** | checksums unchanged; `results/sprint8/` not modified; scripts in this follow-up do not read `split == test` |
| 23 | Sprint 8 report revision | **not done** | not revised in this pass; the original report is preserved |
| — | Blocked: real laser recordings, Jetson/TensorRT timing, live dashboard, rig calibration | **blocked** (external) | HANDOFF §5 |
