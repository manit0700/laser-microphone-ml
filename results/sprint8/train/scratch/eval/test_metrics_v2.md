# Sprint 8 test results (split=test, scored once) - round v2

Acceptance criteria evaluated with `ft2` as the fine-tuned model.

Manifest sha256 `f7d11e6a585f6392…`; test clean clips 249, test sim clips 60; 35 test speakers. Compute: cloud dev container (2 vCPU, CPU-only PyTorch 2.15). Wall time 20 s.

Decision rule: softmax(logits / T), argmax, `unknown` if max prob < threshold. Sim clips are fed through the dashboard laser path (band-pass 200-3800 Hz + enhance); clean clips through the plain path.

## Models

| model | checkpoint | epoch | T | threshold | note |
|---|---|---|---|---|---|
| reference | `results/sprint8/train/scratch/models/reference_cnn.pt` | 3 | 1.000 | 0.58 |  |
| deployed | `models/best_model_cnn.pt` | 36 | 0.498 | 0.26 | deployed checkpoint - training provenance not established as speaker-disjoint from this test set (trained on a Kaggle fold of a superset of these sources), information only |
| control | `results/sprint8/train/scratch/models/control.pt` | 2 | 0.507 | 0.20 |  |
| ft2 | `results/sprint8/train/scratch/models/finetuned_cnn.pt` | 3 | 1.000 | 0.58 |  |

## Headline table (percent)

| model | condition | n digit | n unknown (synth / speech) | digit acc | digits rejected | unknown rejected (synth / speech) | overall 11-class | balanced acc |
|---|---|---|---|---|---|---|---|---|
| reference | clean | 212 | 37 (13 / 24) | 0.0 | 100.0 | 100.0 (100.0 / 100.0) | 14.9 | 9.1 |
| reference | sim_good | 20 | 0 (0 / 0) | 0.0 | 100.0 | - (- / -) | 0.0 | 0.0 |
| reference | sim_medium | 20 | 0 (0 / 0) | 0.0 | 100.0 | - (- / -) | 0.0 | 0.0 |
| reference | sim_poor | 20 | 0 (0 / 0) | 0.0 | 100.0 | - (- / -) | 0.0 | 0.0 |
| reference | sim_all | 60 | 0 (0 / 0) | 0.0 | 100.0 | - (- / -) | 0.0 | 0.0 |
| reference | manit_clean | 92 | 0 (0 / 0) | 0.0 | 100.0 | - (- / -) | 0.0 | 0.0 |
| deployed | clean | 212 | 37 (13 / 24) | 94.3 | 0.5 | 70.3 (38.5 / 87.5) | 90.8 | 92.1 |
| deployed | sim_good | 20 | 0 (0 / 0) | 90.0 | 0.0 | - (- / -) | 90.0 | 89.6 |
| deployed | sim_medium | 20 | 0 (0 / 0) | 100.0 | 0.0 | - (- / -) | 100.0 | 100.0 |
| deployed | sim_poor | 20 | 0 (0 / 0) | 95.0 | 5.0 | - (- / -) | 95.0 | 87.5 |
| deployed | sim_all | 60 | 0 (0 / 0) | 95.0 | 1.7 | - (- / -) | 95.0 | 94.7 |
| deployed | manit_clean | 92 | 0 (0 / 0) | 90.2 | 0.0 | - (- / -) | 90.2 | 90.9 |
| control | clean | 212 | 37 (13 / 24) | 5.7 | 87.3 | 97.3 (100.0 / 95.8) | 19.3 | 14.6 |
| control | sim_good | 20 | 0 (0 / 0) | 0.0 | 90.0 | - (- / -) | 0.0 | 0.0 |
| control | sim_medium | 20 | 0 (0 / 0) | 0.0 | 95.0 | - (- / -) | 0.0 | 0.0 |
| control | sim_poor | 20 | 0 (0 / 0) | 0.0 | 100.0 | - (- / -) | 0.0 | 0.0 |
| control | sim_all | 60 | 0 (0 / 0) | 0.0 | 95.0 | - (- / -) | 0.0 | 0.0 |
| control | manit_clean | 92 | 0 (0 / 0) | 12.0 | 84.8 | - (- / -) | 12.0 | 13.0 |
| ft2 | clean | 212 | 37 (13 / 24) | 0.0 | 100.0 | 100.0 (100.0 / 100.0) | 14.9 | 9.1 |
| ft2 | sim_good | 20 | 0 (0 / 0) | 0.0 | 100.0 | - (- / -) | 0.0 | 0.0 |
| ft2 | sim_medium | 20 | 0 (0 / 0) | 0.0 | 100.0 | - (- / -) | 0.0 | 0.0 |
| ft2 | sim_poor | 20 | 0 (0 / 0) | 0.0 | 100.0 | - (- / -) | 0.0 | 0.0 |
| ft2 | sim_all | 60 | 0 (0 / 0) | 0.0 | 100.0 | - (- / -) | 0.0 | 0.0 |
| ft2 | manit_clean | 92 | 0 (0 / 0) | 0.0 | 100.0 | - (- / -) | 0.0 | 0.0 |

## Acceptance criteria (defined in FROZEN_SETTINGS.json before testing)

| id | statement | result | values |
|---|---|---|---|
| AC1 | fine-tuned test sim_all digit_acc >= reference test sim_all digit_acc + 0.05 | **FAIL** | reference_sim_all_digit_acc=0.0000, finetuned_sim_all_digit_acc=0.0000, delta=0.0000 |
| AC2 | fine-tuned test clean digit_acc >= reference test clean digit_acc - 0.03 | **PASS** | reference_clean_digit_acc=0.0000, finetuned_clean_digit_acc=0.0000, delta=0.0000 |
| AC3 | fine-tuned test sim_all balanced_acc >= reference test sim_all balanced_acc | **PASS** | reference_sim_all_balanced_acc=0.0000, finetuned_sim_all_balanced_acc=0.0000 |
| AC4 | both models: test clean digit_rejection_rate <= 0.05 | **FAIL** | reference_clean_digit_rejection=1.0000, finetuned_clean_digit_rejection=1.0000 |

## Per-class accuracy (percent; digits = correct digit after threshold, unknown = rejected)

### reference

| condition | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | unknown |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 |
| sim_good | - | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - | 0.0 | 0.0 | - |
| sim_medium | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - | 0.0 | - |
| sim_poor | - | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - | 0.0 | 0.0 | 0.0 | - |
| sim_all | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| manit_clean | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - |

### deployed

| condition | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | unknown |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 94.7 | 60.0 | 100.0 | 92.3 | 96.0 | 70.3 |
| sim_good | - | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | - | 66.7 | 50.0 | - |
| sim_medium | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | - | 100.0 | - |
| sim_poor | - | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | - | 100.0 | 100.0 | 0.0 | - |
| sim_all | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 80.0 | 66.7 | - |
| manit_clean | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 88.9 | 20.0 | 100.0 | 100.0 | 100.0 | - |

### control

| condition | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | unknown |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 30.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32.0 | 97.3 |
| sim_good | - | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - | 0.0 | 0.0 | - |
| sim_medium | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - | 0.0 | - |
| sim_poor | - | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - | 0.0 | 0.0 | 0.0 | - |
| sim_all | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| manit_clean | 50.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 80.0 | - |

### ft2

| condition | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | unknown |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 |
| sim_good | - | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - | 0.0 | 0.0 | - |
| sim_medium | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - | 0.0 | - |
| sim_poor | - | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - | 0.0 | 0.0 | 0.0 | - |
| sim_all | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| manit_clean | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - |

## Confusion matrices

PNG per model x condition under `results/sprint8/eval/plots/confusion_<model>_<condition>.png`; the raw 11x11 counts (rows = true, columns = predicted, order 0-9, unknown) are in `test_metrics.json`.
