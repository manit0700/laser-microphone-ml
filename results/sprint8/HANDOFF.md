# Sprint 8 handoff — how to reproduce, what changed, how to resume blocked work

Work was executed on 2026-11-08 in a cloud dev container (2 vCPU, CPU-only PyTorch 2.15, Python 3.13)
on a snapshot of `/Users/manitdankhara/laser_microphone_ml`, then copied back into the project under
`results/sprint8/`, `scripts/sprint8/`, `data/sprint8_sim*/` plus the three source changes listed below.
Nothing was committed, pushed or submitted; the deployed `models/best_model*.pt` are unchanged.

## 1. Environment

```bash
cd ~/laser_microphone_ml
python3 -m venv .venv-sprint8 && . .venv-sprint8/bin/activate
pip install torch torchaudio numpy scipy scikit-learn matplotlib pandas soundfile tabulate pytest
# versions used: torch 2.15.0, torchaudio 2.11.1, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, soundfile 0.14.0
```
`config.SAMPLE_RATE` resolves to 8000 from `models/model_meta.json`. Every Sprint 8 script writes only under
`results/sprint8/`; set `LMML_OUTPUT_DIR` as shown so nothing lands in `models/` or `results/` root.
Seeds: split 42, simulation run seed 2026 (per-clip `sim_seed` stored in the manifest), training 42.

## 2. Reproduce end to end (≈ 1 h on a 2-core CPU; faster with more cores or a GPU)

```bash
. .venv-sprint8/bin/activate
export LMML_OUTPUT_DIR=$PWD/results/sprint8/train

# Data: dedupe + speaker split of originals + simulated variants (click-free simulator) + manifest + integrity
python scripts/sprint8/build_dataset.py --train-sim-target 7000 --sim-dir data/sprint8_sim_v2 --manifest-suffix _v2
python scripts/sprint8/check_split.py --manifest-suffix _v2          # writes SPLIT_VERIFIED_V2 only if 0 overlaps

# Reference CNN from scratch (train clean only; val clean for early stopping)
python scripts/sprint8/train_reference.py --manifest results/sprint8/data/manifest_v2.csv

# Fine-tune a copy on train-split simulated clips + clean replay (30-epoch cap, val-selected)
python scripts/sprint8/finetune_sim.py --manifest results/sprint8/data/manifest_v2.csv --epochs 30 \
    --out-ckpt results/sprint8/models/finetuned_v2_cnn.pt --out-dir results/sprint8/train/finetune_v2
# Control (same schedule, no simulated clips)
python scripts/sprint8/finetune_sim.py --manifest results/sprint8/data/manifest_v2.csv --no-sim \
    --out-ckpt results/sprint8/models/continued_clean_cnn.pt --out-dir results/sprint8/train/control

# Calibration + unknown threshold on VALIDATION only, freeze, then ONE test run
python scripts/sprint8/evaluate_sprint8.py calibrate --manifest results/sprint8/data/manifest_v2.csv --tag v2 \
    --carry reference continued_clean deployed --extra finetuned_v2=results/sprint8/models/finetuned_v2_cnn.pt
python scripts/sprint8/evaluate_sprint8.py test --manifest results/sprint8/data/manifest_v2.csv --tag v2 --ft-name finetuned_v2
python scripts/sprint8/evaluate_sprint8.py environment

# Prototype software replay (no mic/ADC/GUI) + regression checks
export LMML_OUTPUT_DIR=$PWD/results/sprint8/replay
python scripts/sprint8/replay_harness.py --tag deployed --out-dir results/sprint8/replay
python scripts/sprint8/replay_harness.py --accuracy --manifest results/sprint8/data/manifest_v2.csv \
    --model-path results/sprint8/models/finetuned_v2_cnn.pt --tag finetuned_v2_cnn --out-name accuracy_v2_finetuned_v2_cnn --out-dir results/sprint8/replay
python -m pytest scripts/sprint8/test_replay_events.py -q
```
Exact command logs as run (including Round 1 with the pre-fix simulator and the scratch unit tests):
`results/sprint8/{data,train,replay,research}/COMMANDS.md`. `evaluate_sprint8.py test` refuses to re-score
a tag that already has results unless `--force` is given, to keep "test scored once" honest.

## 3. Source files changed outside `scripts/sprint8/` (all uncommitted; review before committing)

| file | change | evidence |
|---|---|---|
| `scripts/physics_laser_sim.py` | `electronics()` now starts the AC-coupling and amplifier filters in steady state (`sosfilt_zi`) instead of zero state. Removes a full-scale ~20 ms start-up click that previously set the normalisation peak of every simulated clip. No random numbers consumed → identical seeds give identical parameters. | `results/sprint8/data/sim_fix_before_after.png`, `sim_v1_vs_v2.json`; effect on accuracy in `docs/EXPERIMENTS.md` Exp 7 |
| `signal_backend.py` (`_LiveClassifier`) | Fix 1 gate scans all pushed audio in 0.15 s frames every 0.05 s (`_gate_frames`); Fix 2 model window anchored to the word (`_word_window`); Fix 3 click rejection on 20 ms active frames, `WORD_MIN_SEC` 0.12 → 0.05 (`_active_seconds`); Fix 4 hold-until-quiet after a `WORD_MAX_SEC` cut-off; new optional `on_decision=`, `clock=`, counters `n_decisions`/`n_inferences`/`last_decision`. Backward compatible; sliding mode unchanged except the injectable clock. | `results/sprint8/replay/CHANGES.md`, `replay_results.md`, `scripts/sprint8/test_replay_events.py` |
| `src/predict.py` | `set_checkpoint_path(model_type, path)`, `checkpoint_path()`, env `LMML_CKPT_<TYPE>` override (explicit > env > `config.model_checkpoint()`); default behaviour unchanged. Lets the dashboard try a Sprint 8 checkpoint without touching `models/`: `LMML_CKPT_CNN=results/sprint8/models/finetuned_v2_cnn.pt python3 signal_dashboard2.py`. | `replay/CHANGES.md` |

Pre-existing uncommitted changes from October (`notebooks/kaggle_train.ipynb`, `scripts/kaggle_push*.sh`,
`src/evaluate.py`, `src/splits.py`, `scripts/add_modified_commands.py`, `scripts/kaggle_train_with_mod.sh`) were
preserved untouched. A copy of the October diff as it was at the start of this work is `.git_uncommitted.diff`.

## 4. Where everything is

```
results/sprint8/
  PLAN.md  SPRINT8_SUMMARY.md  COMPLETION_TABLE.md  HANDOFF.md
  data/      manifest_v2.csv  manifest_summary_v2.md  split_integrity_v2.json  SPLIT_VERIFIED_V2  duplicates.csv
             sim_examples_v2.png  snr_histogram_v2.png  sim_fix_before_after.png  sim_v1_vs_v2.json  (v1 files alongside)
  models/    reference_cnn.pt  finetuned_v2_cnn.pt  continued_clean_cnn.pt  finetuned_cnn.pt  finetuned_cnn_long.pt  checksums.txt
  train/     reference/ finetune/ control/ finetune_long/ finetune_v2/ (history, curves, logs)  COMMANDS.md  ENVIRONMENT.md
  eval/      FROZEN_SETTINGS.json  calibration_val_v2.json  test_metrics_v2.{md,json}  plots_v2/  (Round 1: test_metrics.*, plots/)
  replay/    replay_results.{md,json}  CHANGES.md  COMMANDS.md  scenarios_*.json  accuracy_v2_*.json  click_evidence_v1_v2.json  timing_*.json  baseline/
  research/  assumptions_review.md  sources.md  simulator_vs_rig.md  COMMANDS.md
data/sprint8_sim_v2/<split>/<label>/   9,818 click-free simulated clips (16 kHz PCM_16)   [data/sprint8_sim/ = Round 1, clicky]
scripts/sprint8/   build_dataset.py check_split.py train_reference.py finetune_sim.py evaluate_sprint8.py plots.py
                   replay_harness.py replay_phase_sweep.py replay_timing.py replay_click_evidence.py replay_report.py test_replay_events.py
docs/EXPERIMENTS.md → Exp 7
~/Downloads/CSE4316_SeniorDesign/Manit_Dankhara_Sprint_8_Report.{tex,pdf}
```
Which model is "the" result: `results/sprint8/models/finetuned_v2_cnn.pt` (Round 2, click-free sims, 30 epochs),
compared against `reference_cnn.pt` and the `continued_clean_cnn.pt` control. Round 1 files (`finetuned_cnn.pt`,
`finetuned_cnn_long.pt`, `test_metrics.md`) are kept to document the simulator artifact's effect.

## 5. Blocked tasks and how to resume them

| blocked item | why | command / input needed to resume |
|---|---|---|
| Score on real laser audio | No laser recordings exist in the project; rig not reachable from here | On the Jetson: `python3 scripts/record_daq_wav.py --all --channel laser --speaker laser<name> --takes 10` (and `--unknown 50`), then score them two ways: (a) direct, `for f in data/raw/*_laser*_*.wav; do LMML_CKPT_CNN=results/sprint8/models/finetuned_v2_cnn.pt python3 src/predict.py "$f" --model cnn; done` (compare the label to the filename's first token); (b) through the dashboard path by building a manifest with those files as `kind=sim, split=test` rows (same columns as `manifest_v2.csv`) and running `python3 scripts/sprint8/replay_harness.py --accuracy --manifest <that manifest> --model-path results/sprint8/models/finetuned_v2_cnn.pt --tag real_laser --out-name accuracy_real_laser` (the harness has no dedicated real-laser flag yet; the manifest route reuses the sim path, which is the laser channel's band-pass + enhance). Report detection rate and accuracy separately. |
| Jetson / FP16 / TensorRT latency | Hardware not reachable | On the Nano: `python3 scripts/export_jetson.py`, then `python3 scripts/sprint8/replay_timing.py --out results/sprint8/replay/timing_jetson.json` and label it as Jetson. |
| Live one-event-per-word confirmation on the dashboard | GUI and PCM1808 not reachable | On the Nano: `python3 signal_dashboard2.py` with the laser channel in word mode; speak 10 digits with pauses; compare `results/predictions_log.csv` rows to the words spoken. The fixed `_LiveClassifier` logs a decision per word via `on_decision`. |
| Simulator calibration against the rig | Needs paired std/laser recordings | `python3 scripts/physics_laser_sim.py --calibrate-from <speaker>` after `record_daq_wav.py --channel both`; also the eight measurements in `research/simulator_vs_rig.md` §C. |
| Sprint report actual hours | Needs the Sprint 8 actual-hours spreadsheet | Fill the "Actual work" column in `Manit_Dankhara_Sprint_8_Report.tex`, then `pdflatex` twice. |

## 6. Known caveats to carry forward

- Simulated SNR prior (0–30 dB) is optimistic vs the rig (≈3–7.5 dB in-band): quote the `sim_poor` column.
- Deployed checkpoint numbers are information only (training speakers not shown disjoint from the test set).
- Fine-tunes stopped at the epoch cap while improving; the local reference set is 5× smaller than the Kaggle set.
- Real non-digit speech rejection is 55–68 %; digit "6" is the weakest class.
- Replayed WAVs are peak-normalised (`save_wav`), so absolute gate levels differ from live audio.
