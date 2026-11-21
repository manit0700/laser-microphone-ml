# Sprint 8 — Agent B (training & evaluation) command log

All commands run from `/home/claude/lmml` with
```
. /home/claude/.venv-lmml/bin/activate
export LMML_OUTPUT_DIR=/home/claude/lmml/results/sprint8/train
```
Compute: cloud dev container (2 vCPU, CPU-only PyTorch 2.15). Seed 42 everywhere
(python `random`, numpy, torch; DataLoader shuffle generator seeded 42; num_workers 0 so
augmentation randomness is reproducible in-process). Checkpoints are only written under
`results/sprint8/models/`; `models/` is never touched.

## Phase 0 — unit tests on a fake manifest (scratch, 2026-11-08)

Fake manifest built from a few `data/raw` files (clean + the same files declared as `kind=sim`
so the laser path is exercised; 150 unknown clips), written to
`results/sprint8/train/scratch/fake_manifest.csv` by an inline Python snippet (speakers
jackson/george -> train, nicolas -> val, theo/manit -> test; 1037 rows).

```
S=results/sprint8/train/scratch
python scripts/sprint8/train_reference.py --manifest $S/fake_manifest.csv --out-ckpt $S/models/reference_cnn.pt --out-dir $S/reference --epochs 3 --patience 2
python scripts/sprint8/finetune_sim.py   --manifest $S/fake_manifest.csv --base $S/models/reference_cnn.pt --out-ckpt $S/models/finetuned_cnn.pt --out-dir $S/finetune --epochs 3
python scripts/sprint8/evaluate_sprint8.py calibrate --manifest $S/fake_manifest.csv --reference $S/models/reference_cnn.pt --finetuned $S/models/finetuned_cnn.pt --eval-dir $S/eval
python scripts/sprint8/evaluate_sprint8.py test      --manifest $S/fake_manifest.csv --reference $S/models/reference_cnn.pt --finetuned $S/models/finetuned_cnn.pt --eval-dir $S/eval
```
Result: all four stages run end to end (~1 s/epoch for 316 clips => ~3 ms per clip incl.
augmentation; 23 PNGs written). Fixed on the way: checkpoints must stay loadable with
`torch.load` default `weights_only=True` (torch >= 2.6), so `torch.__version__` is stored
as a string and every save is re-loaded as a check (`check_loadable`).

## Phase 1a — reference CNN (manifest sha256 d81907fda23702ae…, SPLIT_VERIFIED present)

```
nohup python scripts/sprint8/train_reference.py --manifest results/sprint8/data/manifest.csv > results/sprint8/train/reference/train.log 2>&1 &
```
Defaults = the recipe in the script docstring (dropout 0.4, Adam 1e-3, plateau, SpecAugment,
noise 0-30 dB SNR, label smoothing 0.1, batch 32, max 40 epochs, patience 6, seed 42).
Outcome: 40 epochs, best epoch 38 (val loss 0.9696, val acc 0.8416, "GOOD FIT"), wall 1418 s.
Uncontended epochs take 11-13 s; several epochs took 60-200 s because other agents were running
concurrently (load average ~4 on 2 vCPU), not because of augmentation cost.

## Phase 1b — fine-tune on train-split sim + clean replay

```
nohup python scripts/sprint8/finetune_sim.py --manifest results/sprint8/data/manifest.csv > results/sprint8/train/finetune/finetune.log 2>&1 &
```
Defaults: base results/sprint8/models/reference_cnn.pt, lr 1e-4, batch 32, up to 15 epochs,
patience 4 on combined val loss, replay ratio 1.0 (one clean train clip per sim clip, per label),
label smoothing 0.1, waveform augmentation (noise 8-30 dB), no SpecAugment, laser path
(band-pass 200-3800 Hz order 4 + enhance) applied to sim clips in train and eval, seed 42.
Outcome: ran all 15 epochs (patience never triggered; still improving at the cap), selected epoch 15.
Argmax val accuracy: sim 0.4962 -> 0.7376, clean 0.8416 -> 0.8828 (clean floor 0.8116). Wall 736 s
(preload 52 s; uncontended epochs 16-18 s, contended ones up to 111 s). With replay ratio 1.0 every
clean train clip (6391) is replayed because each label has more sim than clean clips.

## Phase 1c — calibration + threshold on VALIDATION, freeze, MODELS_READY

```
python scripts/sprint8/evaluate_sprint8.py calibrate --manifest results/sprint8/data/manifest.csv
```
Outcome (97 s): reference T 0.581, thr 0.26 (fit on val clean, ECE 0.149 -> 0.028); fine-tuned
T 0.581, thr 0.26 (fit on val clean + val sim, ECE 0.167 -> 0.021). Wrote
results/sprint8/eval/{calibration_val.json,FROZEN_SETTINGS.json,threshold_sweep_*.csv}, plots/,
results/sprint8/models/{checksums.txt,MODELS_READY}. Deployed models/best_model_cnn.pt keeps its
own stored T=0.498 / thr=0.26 (not re-fitted, read-only).

## Phase 1d — single test run (after FROZEN_SETTINGS.json)

```
python scripts/sprint8/evaluate_sprint8.py test --manifest results/sprint8/data/manifest.csv
python scripts/sprint8/evaluate_sprint8.py environment
```
Outcome (test, 17 s + environment): all four acceptance criteria PASS. Digit accuracy
reference / fine-tuned / deployed: clean 89.1 / 92.8 / 94.1 %, sim_all 48.4 / 77.4 / 80.5 %,
sim_good 51.9 / 80.0 / 81.8, sim_medium 51.0 / 81.0 / 84.1, sim_poor 41.9 / 70.8 / 75.2,
manit clean 87.0 / 88.0 / 90.2. Full tables: results/sprint8/eval/test_metrics.md. The test split
was scored exactly once; `test` refuses to run again unless --force is passed (not used).

Scratch unit-test outputs remain under results/sprint8/train/scratch/ (fake manifest, 3-epoch
models, eval); they are not part of the results.

## Follow-up (coordinator request, pre-registered before any test look): control + long fine-tune

Both start from results/sprint8/models/reference_cnn.pt with the fine-tune recipe (lr 1e-4, batch 32,
patience 4 on combined val loss, seed 42, identical clean replay sample, laser path on sim clips).
Run sequentially so neither slows the other:
```
python scripts/sprint8/finetune_sim.py --manifest results/sprint8/data/manifest.csv --no-sim \
    --out-ckpt results/sprint8/models/continued_clean_cnn.pt --out-dir results/sprint8/train/control   # control: no sim clips, max 15 epochs
python scripts/sprint8/finetune_sim.py --manifest results/sprint8/data/manifest.csv --epochs 30 \
    --out-ckpt results/sprint8/models/finetuned_cnn_long.pt --out-dir results/sprint8/train/finetune_long  # same recipe, max 30 epochs
python scripts/sprint8/evaluate_sprint8.py calibrate --manifest results/sprint8/data/manifest.csv \
    --extra continued_clean=results/sprint8/models/continued_clean_cnn.pt --extra finetuned_long=results/sprint8/models/finetuned_cnn_long.pt
python scripts/sprint8/evaluate_sprint8.py test --manifest results/sprint8/data/manifest.csv --models continued_clean finetuned_long
python scripts/sprint8/evaluate_sprint8.py environment
```
finetuned_cnn.pt and the original test rows are left untouched; `test --models` only scores the new
models and merges them into test_metrics.json/.md.
Outcomes: control (continued_clean_cnn.pt) early-stopped at epoch 10, selected epoch 6 — val argmax
clean 0.842 -> 0.865, sim 0.496 -> 0.488 (no sim gain without sim exposure); wall 324 s. Long
fine-tune (finetuned_cnn_long.pt) ran all 30 epochs, selected epoch 30 (still at the cap) — val
sim 0.496 -> 0.784, clean 0.842 -> 0.892; wall 1556 s (epochs 16-20 s uncontended, several 90-310 s
under load from other agents). Epochs 1-15 reproduced the first fine-tune run exactly (seeded).
Calibration (97 s): continued_clean T 0.771 thr 0.22; finetuned_long T 0.576 thr 0.24.
Test (85 s, two new models only): continued_clean clean 90.9 % digits / sim_all 47.8 %;
finetuned_long clean 91.6 % / sim_all 79.3 %. AC1-AC4 still PASS (evaluated against finetuned_cnn.pt).

## Round 2 — regenerated sim clips without the start-up click (manifest_v2, SPLIT_VERIFIED_V2)

```
python scripts/sprint8/finetune_sim.py --manifest results/sprint8/data/manifest_v2.csv --epochs 30 \
    --out-ckpt results/sprint8/models/finetuned_v2_cnn.pt --out-dir results/sprint8/train/finetune_v2
python scripts/sprint8/evaluate_sprint8.py calibrate --manifest results/sprint8/data/manifest_v2.csv --tag v2 \
    --carry reference continued_clean deployed --extra finetuned_v2=results/sprint8/models/finetuned_v2_cnn.pt
python scripts/sprint8/evaluate_sprint8.py test --manifest results/sprint8/data/manifest_v2.csv --tag v2 --ft-name finetuned_v2
python scripts/sprint8/evaluate_sprint8.py environment
```
`--carry` copies reference / continued_clean / deployed into the round with their Round-1 frozen
T/threshold unchanged (not re-fitted on v2 val); only finetuned_v2 is calibrated on v2 val.
Outputs go to eval/calibration_val_v2.json, eval/test_metrics_v2.{json,md}, eval/plots_v2/; Round 1
files are untouched.
Outcomes (Round 2): reference argmax val sim 0.5478 on v2 (0.4962 on v1). finetuned_v2 ran all 30
epochs, selected epoch 30 (at the cap) — val sim 0.548 -> 0.811, val clean 0.842 -> 0.892; wall 737 s.
Calibration v2 (10 s): finetuned_v2 T 0.572, thr 0.26. Test v2 (77 s): reference sim_all digits
53.7 % (v1: 48.4 %, so the click artifact cost the reference 5.3 pts); finetuned_v2 sim_all 80.5 %,
clean 91.5 %; AC1-AC4 PASS (finetuned_v2 vs reference). Outputs: eval/test_metrics_v2.{json,md},
eval/plots_v2/ (24 PNGs), eval/calibration_val_v2.json, FROZEN_SETTINGS.json -> rounds.v2.
