# Sprint 8 — agent C (prototype verification) — commands run

All from `/home/claude/lmml` with `. /home/claude/.venv-lmml/bin/activate` and
`export LMML_OUTPUT_DIR=/home/claude/lmml/results/sprint8/replay` (so autosave /
`predictions_log.csv` / `captures/` land under `results/sprint8/replay/results/`,
never in the project's `results/` root). Seed for every harness run: `--seed 8`
(default). Machine: cloud dev container, 2 vCPU, CPU-only PyTorch 2.15; torch
pinned to 1 intra-op thread (`--threads 1`, default) because agents A/B trained
on the same CPUs at the same time.

```bash
pip install pytest                      # was missing from the venv (pytest 9.2.0)

# --- baseline, ORIGINAL code (kept under baseline/) -----------------------------
python scripts/sprint8/replay_harness.py --tag deployed_baseline --out-dir results/sprint8/replay/baseline
python scripts/sprint8/replay_phase_sweep.py results/sprint8/replay/baseline/phase_sweep_before.json

# --- word-window variant comparison on the VALIDATION split (decision evidence) --
# (inline script, see CHANGES.md fix 2; writes window_variants_val.json)

# --- final runs, fixed code -------------------------------------------------------
python scripts/sprint8/replay_harness.py --tag deployed --out-dir results/sprint8/replay
python scripts/sprint8/replay_harness.py --model-path results/sprint8/models/finetuned_cnn.pt --tag finetuned_cnn --configs laser_word --no-autosave --out-dir results/sprint8/replay
python scripts/sprint8/replay_harness.py --model-path results/sprint8/models/reference_cnn.pt --tag reference_cnn --configs laser_word --no-autosave --out-dir results/sprint8/replay
python scripts/sprint8/replay_phase_sweep.py results/sprint8/replay/phase_sweep_after.json
python scripts/sprint8/replay_timing.py --out results/sprint8/replay/timing_cnn_256ms.json
python scripts/sprint8/replay_timing.py --chunk-ms 50 --out results/sprint8/replay/timing_cnn_50ms.json
python scripts/sprint8/replay_timing.py --autosave --out results/sprint8/replay/timing_cnn_256ms_autosave.json
python scripts/sprint8/replay_timing.py --model ensemble --n 20 --out results/sprint8/replay/timing_ensemble_256ms.json

# recognition accuracy (after results/sprint8/models/MODELS_READY appeared, 12:19)
python scripts/sprint8/replay_harness.py --accuracy --tag deployed --out-dir results/sprint8/replay
python scripts/sprint8/replay_harness.py --accuracy --model-path results/sprint8/models/finetuned_cnn.pt --tag finetuned_cnn --out-dir results/sprint8/replay
python scripts/sprint8/replay_harness.py --accuracy --model-path results/sprint8/models/reference_cnn.pt --tag reference_cnn --out-dir results/sprint8/replay

# Round 2 (corrected sims, manifest_v2.csv) -- after the lead's follow-up
python scripts/sprint8/replay_harness.py --accuracy --manifest results/sprint8/data/manifest_v2.csv --tag deployed --out-name accuracy_v2_deployed --out-dir results/sprint8/replay
python scripts/sprint8/replay_harness.py --accuracy --manifest results/sprint8/data/manifest_v2.csv --model-path results/sprint8/models/reference_cnn.pt --tag reference_cnn --out-name accuracy_v2_reference_cnn --out-dir results/sprint8/replay
python scripts/sprint8/replay_harness.py --accuracy --manifest results/sprint8/data/manifest_v2.csv --model-path results/sprint8/models/finetuned_v2_cnn.pt --tag finetuned_v2_cnn --out-name accuracy_v2_finetuned_v2_cnn --out-dir results/sprint8/replay   # only if the checkpoint appeared
python scripts/sprint8/replay_click_evidence.py results/sprint8/replay/click_evidence_v1_v2.json

# After Fix 4 (hold-until-quiet) every top-level result file was regenerated with the final code
# (same commands as above); the pre-Fix-4 files are kept in results/sprint8/replay/round1_fix3_snapshot/.

# regression checks
python -m pytest scripts/sprint8/test_replay_events.py -q

# report
python scripts/sprint8/replay_report.py
```

Scripts (all owned by C): `scripts/sprint8/replay_harness.py`,
`replay_phase_sweep.py`, `replay_timing.py`, `replay_report.py`,
`test_replay_events.py`.
