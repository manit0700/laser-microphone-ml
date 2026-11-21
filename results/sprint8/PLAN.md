# Sprint 8 — simulation & software-prototype plan (lead-owned)

Project copy: /home/claude/lmml  (snapshot of /Users/manitdankhara/laser_microphone_ml, 2026-11-08)
Python env:   /home/claude/.venv-lmml  (activate: `. /home/claude/.venv-lmml/bin/activate`)
Compute:      this container, CPU only, 2 cores, 7 GB RAM. Name it "cloud dev container (2 vCPU, CPU-only PyTorch 2.15)" in any timing.
Sample rate:  config.SAMPLE_RATE = 8000 (from models/model_meta.json). Do not change.

## Ground rules for every agent
- Other agents work in this same tree at the same time. Only edit files you own (below). Never `git checkout`/reset. Never delete anything under data/raw, models/, results/ that you did not create.
- Never overwrite models/best_model*.pt or models/backups/*. New checkpoints go under results/sprint8/models/.
- Set `LMML_OUTPUT_DIR=/home/claude/lmml/results/sprint8/<yourarea>` when running train.py/finetune.py so models/ and results/ writes land there.
- Record exact commands you ran in results/sprint8/<yourarea>/COMMANDS.md and seeds in your outputs.
- Never tune anything on the test split. Test is read once, after settings are frozen (see B).
- Report negative results as they are. Do not invent completion.

## Shared interfaces
### Manifest  (owner A → consumers B, C)
results/sprint8/data/manifest.csv, one row per clip, columns exactly:
  file            path relative to project root (data/raw/... or data/sprint8_sim/...)
  label           0-9 or unknown
  speaker         splits.speaker_of() value (SC hash, fsdd name, 'manit', etc.)
  split           train | val | test
  kind            clean | sim
  dataset         fsdd | speech_commands | team | sc_unknown | noise | silence
  synthetic_noise 0/1  (1 for generated noise/silence 'unknown' clips)
  sha256          content hash of this file
  source_file     for sim: the clean file it was made from (relative path); for clean: same as file
  source_sha256   hash of source_file
  sim_seed        integer per-clip seed (empty for clean)
  sim_params      JSON string of simulate() params (surface, depth, offset, snr_db, hum_db, hp_hz, lp_hz, headroom)
  snr_db          float for sim (empty for clean)
  condition       clean | sim_good (snr>=20) | sim_medium (10<=snr<20) | sim_poor (snr<10)
Marker file results/sprint8/data/SPLIT_VERIFIED is written by A only after split_integrity.json shows 0 overlaps. B waits for it.

### Checkpoints (owner B)
results/sprint8/models/reference_cnn.pt  (clean audio, from scratch, train split only)
results/sprint8/models/finetuned_cnn.pt  (copy of reference, fine-tuned on train-split sim + clean replay)
Both are standard project checkpoints (loadable by model_from_checkpoint, with labels/feature/temperature/threshold keys) so predict.py / _LiveClassifier can use them. B writes results/sprint8/models/MODELS_READY when final.

### Replay harness (owner C)
scripts/sprint8/replay_harness.py — takes --model-path; C runs deterministic event checks with the deployed model first, then re-runs accuracy checks with results/sprint8/models/finetuned_cnn.pt when MODELS_READY exists.

## File ownership
A: scripts/sprint8/build_dataset.py, scripts/sprint8/check_split.py, data/sprint8_sim/**, results/sprint8/data/**
B: scripts/sprint8/train_reference.py, scripts/sprint8/finetune_sim.py, scripts/sprint8/evaluate_sprint8.py, scripts/sprint8/plots.py, results/sprint8/models/**, results/sprint8/eval/**, results/sprint8/train/**
C: scripts/sprint8/replay_harness.py, scripts/sprint8/test_replay_*.py, results/sprint8/replay/**, and (only for confirmed bugs) signal_backend.py, src/predict.py, src/enhance.py, src/preprocess.py — announce any src/ change in results/sprint8/replay/CHANGES.md with before/after regression evidence.
R: results/sprint8/research/**
Lead: docs/EXPERIMENTS.md, results/sprint8/SPRINT8_SUMMARY.md, COMPLETION_TABLE.md, HANDOFF.md, the Sprint 8 report.
