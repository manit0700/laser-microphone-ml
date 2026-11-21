# Sprint 8 final: completion table (as of this session)

| # | task | status | evidence |
|---|---|---|---|
| A1 | Playlist/source/prediction match (57 clips) | DONE | audio_review/label_review_status.csv; objective_checks (57/57) |
| A2 | Duration, sample rate, clipping, waveform, spectrogram checks | DONE | audio_review/REVIEW.md; inspection PNG |
| A3 | Compare "six→eight" vs correct controls | DONE (descriptive) | audio_review/error_vs_control_structure.json (not separable) |
| A4 | Listening / human labels | **BLOCKED on human** | review_status = PENDING_HUMAN_LISTEN for all 57 |
| A5 | Numbered playlist and review sheet | DONE (CSV); HTML tool optional, not built | audio_review/ |
| B1 | Existing 17 replay regression tests | DONE | replay/pytest_regression_final.log |
| B2 | Scenario replay, 8 configs × 2 chunk sizes, 30 scenarios | DONE (validation clips) | replay/scenarios_finetuned_v2_val.json; expected_vs_observed_events.csv |
| B3 | Expected-vs-observed event table | DONE | replay/expected_vs_observed_events.md |
| B4 | Preprocessing consistency map | DONE (documented in REVIEW.md and MODEL_CARD.md) | package/MODEL_CARD.md |
| B5 | Focused code fixes | NONE needed; replay additive `--pool-split` only | scripts/sprint8/replay_harness.py |
| C1 | Intended model identified and frozen | DONE (historical frozen finetuned_v2, not a new selection) | package/config/model_config.json |
| C2 | Checksum, label order, feature, filter, T, threshold, deps | DONE | package/SHA256SUMS; config |
| C3 | Fresh-process load | DONE | package/VERIFICATION_LOG.md |
| C4 | Relative paths from outside repo | DONE | VERIFICATION_LOG.md |
| C5 | No /home/claude paths in source | DONE for package code; the repository code is still required | VERIFICATION_LOG.md, README.md |
| C6 | Pinned checkpoint, not deployed default | DONE (sha check; env override ignored) | VERIFICATION_LOG.md |
| C7 | Agreement with repository pipeline (20 clips) | DONE: max |dp| = 0, argmax 20/20 | VERIFICATION_LOG.md |
| C8 | Error messages (missing input, checkpoint, checksum, config, repo, usage) | DONE | VERIFICATION_LOG.md |
| C9 | MODEL_CARD.md and README.md | DONE | package/ |
| D1 | Hardware rig / Jetson access | **BLOCKED**: not reachable from this container; device tools reach only the Mac | see HANDOFF.md |
| D3 | Laser class wording ("grade level 1", "grade level 3") | RECORDED as UNCONFIRMED; labels, datasheets and Class 3R/3B status requested | HARDWARE_SPEC_MISSING.md |
| D2 | Level 1 / Level 3 definitions | **MISSING SPECIFICATION**: not defined in project files; not invented | see HANDOFF.md |
| E1 | Final report (Manit_Dankhara_Sprint_8_Final_Review .tex/.pdf) | DONE: 7 pages, rendered and inspected; human fields left open | CSE4316_SeniorDesign/ on Mac; checksums verified |
| E2 | Transfer to Mac + checksum verification | DONE: 371 files extracted, all OK; report checksums OK | laser_microphone_ml/transfer/ |
| E3 | Hours, sprint dates, peer scores, due date | **MISSING** (human input) | see HANDOFF.md |
