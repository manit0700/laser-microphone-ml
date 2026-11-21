# Sprint 8 test results (split=test, scored once)

Acceptance criteria evaluated with `finetuned` as the fine-tuned model.

Manifest sha256 `d81907fda23702ae…`; test clean clips 1385, test sim clips 1385; 45 test speakers. Compute: cloud dev container (2 vCPU, CPU-only PyTorch 2.15). Wall time 85 s.

Decision rule: softmax(logits / T), argmax, `unknown` if max prob < threshold. Sim clips are fed through the dashboard laser path (band-pass 200-3800 Hz + enhance); clean clips through the plain path.

Later additions (pre-registered on validation, then scored once): 2026-11-08T18:57:49+00:00: continued_clean, finetuned_long

## Models

| model | checkpoint | epoch | T | threshold | note |
|---|---|---|---|---|---|
| reference | `results/sprint8/models/reference_cnn.pt` | 38 | 0.581 | 0.26 |  |
| finetuned | `results/sprint8/models/finetuned_cnn.pt` | 15 | 0.581 | 0.26 |  |
| deployed | `models/best_model_cnn.pt` | 36 | 0.498 | 0.26 | deployed checkpoint - training provenance not established as speaker-disjoint from this test set (trained on a Kaggle fold of a superset of these sources), information only |
| continued_clean | `results/sprint8/models/continued_clean_cnn.pt` | 6 | 0.771 | 0.22 |  |
| finetuned_long | `results/sprint8/models/finetuned_cnn_long.pt` | 30 | 0.576 | 0.24 |  |

## Headline table (percent)

| model | condition | n digit | n unknown (synth / speech) | digit acc | digits rejected | unknown rejected (synth / speech) | overall 11-class | balanced acc |
|---|---|---|---|---|---|---|---|---|
| reference | clean | 1275 | 110 (50 / 60) | 89.1 | 2.1 | 82.7 (100.0 / 68.3) | 88.6 | 88.4 |
| reference | sim_good | 451 | 42 (21 / 21) | 51.9 | 10.2 | 66.7 (85.7 / 47.6) | 53.1 | 53.3 |
| reference | sim_medium | 416 | 38 (16 / 22) | 51.0 | 12.5 | 71.1 (87.5 / 59.1) | 52.6 | 52.6 |
| reference | sim_poor | 408 | 30 (13 / 17) | 41.9 | 17.4 | 80.0 (92.3 / 70.6) | 44.5 | 44.5 |
| reference | sim_all | 1275 | 110 (50 / 60) | 48.4 | 13.3 | 71.8 (88.0 / 58.3) | 50.3 | 50.2 |
| reference | manit_clean | 92 | 0 (0 / 0) | 87.0 | 0.0 | - (- / -) | 87.0 | 87.8 |
| finetuned | clean | 1275 | 110 (50 / 60) | 92.8 | 1.3 | 76.4 (98.0 / 58.3) | 91.5 | 91.2 |
| finetuned | sim_good | 451 | 42 (21 / 21) | 80.0 | 3.5 | 69.0 (85.7 / 52.4) | 79.1 | 78.8 |
| finetuned | sim_medium | 416 | 38 (16 / 22) | 81.0 | 3.6 | 71.1 (100.0 / 50.0) | 80.2 | 80.4 |
| finetuned | sim_poor | 408 | 30 (13 / 17) | 70.8 | 6.4 | 76.7 (92.3 / 64.7) | 71.2 | 71.1 |
| finetuned | sim_all | 1275 | 110 (50 / 60) | 77.4 | 4.5 | 71.8 (92.0 / 55.0) | 77.0 | 76.8 |
| finetuned | manit_clean | 92 | 0 (0 / 0) | 88.0 | 0.0 | - (- / -) | 88.0 | 88.8 |
| deployed | clean | 1275 | 110 (50 / 60) | 94.1 | 0.6 | 45.5 (14.0 / 71.7) | 90.3 | 89.7 |
| deployed | sim_good | 451 | 42 (21 / 21) | 81.8 | 1.8 | 31.0 (0.0 / 61.9) | 77.5 | 76.9 |
| deployed | sim_medium | 416 | 38 (16 / 22) | 84.1 | 1.4 | 23.7 (0.0 / 40.9) | 79.1 | 78.6 |
| deployed | sim_poor | 408 | 30 (13 / 17) | 75.2 | 1.5 | 23.3 (0.0 / 41.2) | 71.7 | 70.2 |
| deployed | sim_all | 1275 | 110 (50 / 60) | 80.5 | 1.6 | 26.4 (0.0 / 48.3) | 76.2 | 75.5 |
| deployed | manit_clean | 92 | 0 (0 / 0) | 90.2 | 0.0 | - (- / -) | 90.2 | 90.9 |
| continued_clean | clean | 1275 | 110 (50 / 60) | 90.9 | 2.4 | 83.6 (98.0 / 71.7) | 90.3 | 90.1 |
| continued_clean | sim_good | 451 | 42 (21 / 21) | 52.1 | 12.0 | 69.0 (85.7 / 52.4) | 53.5 | 53.6 |
| continued_clean | sim_medium | 416 | 38 (16 / 22) | 50.5 | 13.5 | 76.3 (87.5 / 68.2) | 52.6 | 52.6 |
| continued_clean | sim_poor | 408 | 30 (13 / 17) | 40.4 | 20.6 | 80.0 (92.3 / 70.6) | 43.2 | 43.1 |
| continued_clean | sim_all | 1275 | 110 (50 / 60) | 47.8 | 15.2 | 74.5 (88.0 / 63.3) | 50.0 | 49.8 |
| continued_clean | manit_clean | 92 | 0 (0 / 0) | 88.0 | 0.0 | - (- / -) | 88.0 | 88.7 |
| finetuned_long | clean | 1275 | 110 (50 / 60) | 91.6 | 1.6 | 82.7 (98.0 / 70.0) | 90.9 | 90.6 |
| finetuned_long | sim_good | 451 | 42 (21 / 21) | 79.8 | 4.7 | 81.0 (100.0 / 61.9) | 79.9 | 79.8 |
| finetuned_long | sim_medium | 416 | 38 (16 / 22) | 84.4 | 2.2 | 73.7 (100.0 / 54.5) | 83.5 | 83.4 |
| finetuned_long | sim_poor | 408 | 30 (13 / 17) | 73.5 | 6.4 | 86.7 (100.0 / 76.5) | 74.4 | 74.6 |
| finetuned_long | sim_all | 1275 | 110 (50 / 60) | 79.3 | 4.4 | 80.0 (100.0 / 63.3) | 79.4 | 79.3 |
| finetuned_long | manit_clean | 92 | 0 (0 / 0) | 85.9 | 0.0 | - (- / -) | 85.9 | 86.7 |

## Acceptance criteria (defined in FROZEN_SETTINGS.json before testing)

| id | statement | result | values |
|---|---|---|---|
| AC1 | fine-tuned test sim_all digit_acc >= reference test sim_all digit_acc + 0.05 | **PASS** | reference_sim_all_digit_acc=0.4839, finetuned_sim_all_digit_acc=0.7741, delta=0.2902 |
| AC2 | fine-tuned test clean digit_acc >= reference test clean digit_acc - 0.03 | **PASS** | reference_clean_digit_acc=0.8910, finetuned_clean_digit_acc=0.9278, delta=0.0369 |
| AC3 | fine-tuned test sim_all balanced_acc >= reference test sim_all balanced_acc | **PASS** | reference_sim_all_balanced_acc=0.5018, finetuned_sim_all_balanced_acc=0.7682 |
| AC4 | both models: test clean digit_rejection_rate <= 0.05 | **PASS** | reference_clean_digit_rejection=0.0212, finetuned_clean_digit_rejection=0.0133 |

## Per-class accuracy (percent; digits = correct digit after threshold, unknown = rejected)

### reference

| condition | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | unknown |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 91.2 | 96.3 | 96.1 | 87.6 | 96.1 | 87.1 | 59.7 | 91.6 | 95.5 | 88.1 | 82.7 |
| sim_good | 79.2 | 69.0 | 55.0 | 73.2 | 44.4 | 35.9 | 15.9 | 56.2 | 52.1 | 38.3 | 66.7 |
| sim_medium | 76.3 | 81.0 | 50.0 | 57.4 | 23.7 | 34.2 | 19.0 | 60.5 | 53.5 | 51.4 | 71.1 |
| sim_poor | 53.8 | 74.5 | 39.5 | 51.2 | 37.8 | 23.1 | 23.7 | 32.5 | 43.9 | 29.5 | 80.0 |
| sim_all | 70.4 | 74.8 | 48.4 | 60.5 | 36.4 | 31.0 | 19.4 | 50.4 | 50.0 | 38.9 | 71.8 |
| manit_clean | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 77.8 | 0.0 | 100.0 | 100.0 | 100.0 | - |

### finetuned

| condition | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | unknown |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 93.6 | 97.0 | 95.3 | 93.0 | 96.1 | 87.1 | 83.1 | 95.4 | 93.2 | 92.9 | 76.4 |
| sim_good | 87.5 | 78.6 | 80.0 | 85.4 | 92.6 | 71.8 | 79.5 | 77.1 | 70.8 | 74.5 | 69.0 |
| sim_medium | 89.5 | 90.5 | 74.0 | 76.6 | 76.3 | 76.3 | 73.8 | 88.4 | 76.7 | 91.4 | 71.1 |
| sim_poor | 74.4 | 84.3 | 65.8 | 63.4 | 75.7 | 69.2 | 63.2 | 70.0 | 80.5 | 59.1 | 76.7 |
| sim_all | 84.0 | 84.4 | 73.4 | 75.2 | 82.9 | 72.4 | 72.6 | 78.6 | 75.8 | 73.8 | 71.8 |
| manit_clean | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 77.8 | 10.0 | 100.0 | 100.0 | 100.0 | - |

### deployed

| condition | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | unknown |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 93.6 | 97.0 | 97.7 | 90.7 | 94.6 | 94.8 | 88.7 | 99.2 | 93.9 | 90.5 | 45.5 |
| sim_good | 87.5 | 73.8 | 75.0 | 85.4 | 87.0 | 76.9 | 84.1 | 87.5 | 68.8 | 89.4 | 31.0 |
| sim_medium | 92.1 | 90.5 | 86.0 | 80.9 | 76.3 | 73.7 | 90.5 | 93.0 | 69.8 | 88.6 | 23.7 |
| sim_poor | 76.9 | 88.2 | 76.3 | 78.0 | 81.1 | 69.2 | 71.1 | 80.0 | 51.2 | 77.3 | 23.3 |
| sim_all | 85.6 | 84.4 | 79.7 | 81.4 | 82.2 | 73.3 | 82.3 | 87.0 | 63.6 | 84.9 | 26.4 |
| manit_clean | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 88.9 | 20.0 | 100.0 | 100.0 | 100.0 | - |

### continued_clean

| condition | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | unknown |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 93.6 | 97.0 | 90.6 | 86.8 | 96.1 | 85.3 | 82.3 | 91.6 | 93.2 | 91.3 | 83.6 |
| sim_good | 77.1 | 81.0 | 52.5 | 65.9 | 44.4 | 28.2 | 25.0 | 58.3 | 50.0 | 38.3 | 69.0 |
| sim_medium | 68.4 | 90.5 | 40.0 | 53.2 | 28.9 | 18.4 | 40.5 | 74.4 | 41.9 | 45.7 | 76.3 |
| sim_poor | 51.3 | 80.4 | 39.5 | 46.3 | 37.8 | 17.9 | 28.9 | 35.0 | 34.1 | 22.7 | 80.0 |
| sim_all | 66.4 | 83.7 | 43.8 | 55.0 | 38.0 | 21.6 | 31.5 | 56.5 | 42.4 | 34.9 | 74.5 |
| manit_clean | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 66.7 | 20.0 | 100.0 | 100.0 | 100.0 | - |

### finetuned_long

| condition | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | unknown |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 91.2 | 98.5 | 97.7 | 91.5 | 96.1 | 82.8 | 74.2 | 97.7 | 93.2 | 91.3 | 82.7 |
| sim_good | 85.4 | 81.0 | 77.5 | 80.5 | 87.0 | 79.5 | 72.7 | 83.3 | 75.0 | 74.5 | 81.0 |
| sim_medium | 89.5 | 90.5 | 90.0 | 74.5 | 73.7 | 81.6 | 81.0 | 88.4 | 83.7 | 91.4 | 73.7 |
| sim_poor | 71.8 | 84.3 | 84.2 | 70.7 | 78.4 | 66.7 | 63.2 | 77.5 | 78.0 | 59.1 | 86.7 |
| sim_all | 82.4 | 85.2 | 84.4 | 75.2 | 80.6 | 75.9 | 72.6 | 83.2 | 78.8 | 73.8 | 80.0 |
| manit_clean | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 66.7 | 0.0 | 100.0 | 100.0 | 100.0 | - |

## Confusion matrices

PNG per model x condition under `results/sprint8/eval/plots/confusion_<model>_<condition>.png`; the raw 11x11 counts (rows = true, columns = predicted, order 0-9, unknown) are in `test_metrics.json`.
