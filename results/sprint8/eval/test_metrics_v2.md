# Sprint 8 test results (split=test, scored once) - round v2

Acceptance criteria evaluated with `finetuned_v2` as the fine-tuned model.

Manifest sha256 `4095986e12c17de7…`; test clean clips 1385, test sim clips 1385; 45 test speakers. Compute: cloud dev container (2 vCPU, CPU-only PyTorch 2.15). Wall time 77 s.

Decision rule: softmax(logits / T), argmax, `unknown` if max prob < threshold. Sim clips are fed through the dashboard laser path (band-pass 200-3800 Hz + enhance); clean clips through the plain path.

## Models

| model | checkpoint | epoch | T | threshold | note |
|---|---|---|---|---|---|
| reference | `results/sprint8/models/reference_cnn.pt` | 38 | 0.581 | 0.26 |  |
| continued_clean | `results/sprint8/models/continued_clean_cnn.pt` | 6 | 0.771 | 0.22 |  |
| deployed | `models/best_model_cnn.pt` | 36 | 0.498 | 0.26 | deployed checkpoint - training provenance not established as speaker-disjoint from this test set (trained on a Kaggle fold of a superset of these sources), information only |
| finetuned_v2 | `results/sprint8/models/finetuned_v2_cnn.pt` | 30 | 0.572 | 0.26 |  |

## Headline table (percent)

| model | condition | n digit | n unknown (synth / speech) | digit acc | digits rejected | unknown rejected (synth / speech) | overall 11-class | balanced acc |
|---|---|---|---|---|---|---|---|---|
| reference | clean | 1275 | 110 (50 / 60) | 89.1 | 2.1 | 82.7 (100.0 / 68.3) | 88.6 | 88.4 |
| reference | sim_good | 451 | 42 (21 / 21) | 55.9 | 15.5 | 69.0 (85.7 / 52.4) | 57.0 | 57.1 |
| reference | sim_medium | 416 | 38 (16 / 22) | 56.5 | 17.5 | 78.9 (93.8 / 68.2) | 58.4 | 58.5 |
| reference | sim_poor | 408 | 30 (13 / 17) | 48.5 | 24.8 | 83.3 (92.3 / 76.5) | 50.9 | 51.1 |
| reference | sim_all | 1275 | 110 (50 / 60) | 53.7 | 19.1 | 76.4 (90.0 / 65.0) | 55.5 | 55.6 |
| reference | manit_clean | 92 | 0 (0 / 0) | 87.0 | 0.0 | - (- / -) | 87.0 | 87.8 |
| continued_clean | clean | 1275 | 110 (50 / 60) | 90.9 | 2.4 | 83.6 (98.0 / 71.7) | 90.3 | 90.1 |
| continued_clean | sim_good | 451 | 42 (21 / 21) | 58.3 | 15.5 | 69.0 (85.7 / 52.4) | 59.2 | 59.1 |
| continued_clean | sim_medium | 416 | 38 (16 / 22) | 56.0 | 18.8 | 81.6 (93.8 / 72.7) | 58.1 | 58.1 |
| continued_clean | sim_poor | 408 | 30 (13 / 17) | 48.0 | 27.9 | 90.0 (92.3 / 88.2) | 50.9 | 51.3 |
| continued_clean | sim_all | 1275 | 110 (50 / 60) | 54.3 | 20.5 | 79.1 (90.0 / 70.0) | 56.2 | 56.3 |
| continued_clean | manit_clean | 92 | 0 (0 / 0) | 88.0 | 0.0 | - (- / -) | 88.0 | 88.7 |
| deployed | clean | 1275 | 110 (50 / 60) | 94.1 | 0.6 | 45.5 (14.0 / 71.7) | 90.3 | 89.7 |
| deployed | sim_good | 451 | 42 (21 / 21) | 83.8 | 1.1 | 28.6 (0.0 / 57.1) | 79.1 | 78.6 |
| deployed | sim_medium | 416 | 38 (16 / 22) | 87.7 | 1.9 | 26.3 (0.0 / 45.5) | 82.6 | 82.2 |
| deployed | sim_poor | 408 | 30 (13 / 17) | 81.6 | 1.5 | 20.0 (0.0 / 35.3) | 77.4 | 76.1 |
| deployed | sim_all | 1275 | 110 (50 / 60) | 84.4 | 1.5 | 25.5 (0.0 / 46.7) | 79.7 | 79.1 |
| deployed | manit_clean | 92 | 0 (0 / 0) | 90.2 | 0.0 | - (- / -) | 90.2 | 90.9 |
| finetuned_v2 | clean | 1275 | 110 (50 / 60) | 91.5 | 1.3 | 81.8 (98.0 / 68.3) | 90.8 | 90.4 |
| finetuned_v2 | sim_good | 451 | 42 (21 / 21) | 80.7 | 4.4 | 81.0 (100.0 / 61.9) | 80.7 | 80.8 |
| finetuned_v2 | sim_medium | 416 | 38 (16 / 22) | 84.4 | 2.4 | 73.7 (100.0 / 54.5) | 83.5 | 83.4 |
| finetuned_v2 | sim_poor | 408 | 30 (13 / 17) | 76.5 | 5.6 | 80.0 (100.0 / 64.7) | 76.7 | 76.7 |
| finetuned_v2 | sim_all | 1275 | 110 (50 / 60) | 80.5 | 4.2 | 78.2 (100.0 / 60.0) | 80.4 | 80.3 |
| finetuned_v2 | manit_clean | 92 | 0 (0 / 0) | 85.9 | 0.0 | - (- / -) | 85.9 | 86.7 |

## Acceptance criteria (defined in FROZEN_SETTINGS.json before testing)

| id | statement | result | values |
|---|---|---|---|
| AC1 | fine-tuned test sim_all digit_acc >= reference test sim_all digit_acc + 0.05 | **PASS** | reference_sim_all_digit_acc=0.5373, finetuned_sim_all_digit_acc=0.8055, delta=0.2682 |
| AC2 | fine-tuned test clean digit_acc >= reference test clean digit_acc - 0.03 | **PASS** | reference_clean_digit_acc=0.8910, finetuned_clean_digit_acc=0.9153, delta=0.0243 |
| AC3 | fine-tuned test sim_all balanced_acc >= reference test sim_all balanced_acc | **PASS** | reference_sim_all_balanced_acc=0.5560, finetuned_sim_all_balanced_acc=0.8025 |
| AC4 | both models: test clean digit_rejection_rate <= 0.05 | **PASS** | reference_clean_digit_rejection=0.0212, finetuned_clean_digit_rejection=0.0133 |

## Per-class accuracy (percent; digits = correct digit after threshold, unknown = rejected)

### reference

| condition | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | unknown |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 91.2 | 96.3 | 96.1 | 87.6 | 96.1 | 87.1 | 59.7 | 91.6 | 95.5 | 88.1 | 82.7 |
| sim_good | 70.8 | 66.7 | 47.5 | 61.0 | 53.7 | 64.1 | 27.3 | 60.4 | 64.6 | 42.6 | 69.0 |
| sim_medium | 63.2 | 83.3 | 50.0 | 55.3 | 47.4 | 55.3 | 31.0 | 65.1 | 65.1 | 48.6 | 78.9 |
| sim_poor | 46.2 | 76.5 | 47.4 | 39.0 | 51.4 | 53.8 | 28.9 | 40.0 | 63.4 | 31.8 | 83.3 |
| sim_all | 60.8 | 75.6 | 48.4 | 51.9 | 51.2 | 57.8 | 29.0 | 55.7 | 64.4 | 40.5 | 76.4 |
| manit_clean | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 77.8 | 0.0 | 100.0 | 100.0 | 100.0 | - |

### continued_clean

| condition | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | unknown |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 93.6 | 97.0 | 90.6 | 86.8 | 96.1 | 85.3 | 82.3 | 91.6 | 93.2 | 91.3 | 83.6 |
| sim_good | 72.9 | 76.2 | 50.0 | 56.1 | 57.4 | 56.4 | 38.6 | 64.6 | 62.5 | 46.8 | 69.0 |
| sim_medium | 55.3 | 88.1 | 42.0 | 55.3 | 50.0 | 36.8 | 59.5 | 74.4 | 53.5 | 42.9 | 81.6 |
| sim_poor | 46.2 | 78.4 | 47.4 | 39.0 | 56.8 | 46.2 | 36.8 | 42.5 | 51.2 | 29.5 | 90.0 |
| sim_all | 59.2 | 80.7 | 46.1 | 50.4 | 55.0 | 46.6 | 45.2 | 61.1 | 56.1 | 39.7 | 79.1 |
| manit_clean | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 66.7 | 20.0 | 100.0 | 100.0 | 100.0 | - |

### deployed

| condition | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | unknown |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 93.6 | 97.0 | 97.7 | 90.7 | 94.6 | 94.8 | 88.7 | 99.2 | 93.9 | 90.5 | 45.5 |
| sim_good | 87.5 | 73.8 | 85.0 | 80.5 | 87.0 | 82.1 | 88.6 | 87.5 | 77.1 | 87.2 | 28.6 |
| sim_medium | 92.1 | 88.1 | 92.0 | 74.5 | 86.8 | 84.2 | 97.6 | 97.7 | 79.1 | 85.7 | 26.3 |
| sim_poor | 79.5 | 86.3 | 81.6 | 70.7 | 91.9 | 89.7 | 84.2 | 85.0 | 73.2 | 75.0 | 20.0 |
| sim_all | 86.4 | 83.0 | 86.7 | 75.2 | 88.4 | 85.3 | 90.3 | 90.1 | 76.5 | 82.5 | 25.5 |
| manit_clean | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 88.9 | 20.0 | 100.0 | 100.0 | 100.0 | - |

### finetuned_v2

| condition | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | unknown |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 92.8 | 98.5 | 97.7 | 93.8 | 95.3 | 81.0 | 69.4 | 99.2 | 93.9 | 91.3 | 81.8 |
| sim_good | 85.4 | 83.3 | 90.0 | 80.5 | 87.0 | 82.1 | 75.0 | 81.2 | 79.2 | 63.8 | 81.0 |
| sim_medium | 94.7 | 92.9 | 90.0 | 72.3 | 76.3 | 84.2 | 73.8 | 95.3 | 83.7 | 80.0 | 73.7 |
| sim_poor | 76.9 | 88.2 | 86.8 | 70.7 | 78.4 | 71.8 | 71.1 | 80.0 | 75.6 | 63.6 | 80.0 |
| sim_all | 85.6 | 88.1 | 89.1 | 74.4 | 81.4 | 79.3 | 73.4 | 85.5 | 79.5 | 68.3 | 78.2 |
| manit_clean | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 66.7 | 0.0 | 100.0 | 100.0 | 100.0 | - |

## Confusion matrices

PNG per model x condition under `results/sprint8/eval/plots_v2/confusion_<model>_<condition>.png`; the raw 11x11 counts (rows = true, columns = predicted, order 0-9, unknown) are in `test_metrics_v2.json`.
