# Replay analysis: lmml-sprint8-finetuned_v2 (validation clips, controlled stream)

Source: scenarios_finetuned_v2_val.json (run by scripts/sprint8/replay_harness.py, `--pool-split val`, manifest_v2.csv). Flattened table: expected_vs_observed_events.csv / .md (240 rows). Summary script: scripts/sprint8_final/replay_summary.py.

## Scope
- Development (validation) clips only, one controlled stream at 48 kHz. This is software replay, **not** physical hardware reliability.
- Word-mode configurations reflect the decisions the streaming classifier emits. Sliding configurations emit repeated decisions by design.

## Aggregate counts (word scenarios; 48 spoken words per configuration, 42 digits and 6 unknown words)
| config | chunk | events | missed | duplicate | digits correct | digits wrong | digits rejected | unknown rejected / 6 | unknown false accept |
|---|---|---|---|---|---|---|---|---|---|
| laser_word | 256 ms | 46 | 2 | 0 | 35/42 | 5 | 0 | 6 | 0 |
| laser_word | 50 ms | 47 | 2 | 1 | 35/42 | 5 | 0 | 6 | 0 |
| std_ensemble_word | 256 ms | 46 | 5 | 1 | 31/42 | 1 | 5 | 6 | 0 |
| std_ensemble_word | 50 ms | 46 | 5 | 1 | 31/42 | 1 | 5 | 6 | 0 |
| laser_sliding | 256 ms | 43 | 17 | 12 | 21/42 | 6 | 2 | 1 / 6 | 1 |
| laser_sliding | 50 ms | 56 | 13 | 21 | 20/42 | 10 | 0 | 0 / 6 | 5 |
| std_ensemble_sliding | 256 ms | 33 | 19 | 3 | 25/42 | 1 | 2 | 1 / 6 | 0 |
| std_ensemble_sliding | 50 ms | 32 | 20 | 3 | 23/42 | 1 | 3 | 1 / 6 | 0 |

"Missed" = spoken word with no event. "Duplicate" = an extra event for an already-decided word. Counts come from the saved event-handling fields.

## Noise-only and silence scenarios (15 rows per configuration)
| config | noise-only events | spurious unknown | false activations |
|---|---|---|---|
| laser_word (both chunks) | 13 / 14 | 12 / 13 | 1 |
| std_ensemble_word (both) | 8 | 8 | 0 |
| laser_sliding (both) | 6 | 0 | 6 |
| std_ensemble_sliding (both) | 6 | 6 | 0 |

Silence (5 s) produced no events in every configuration.

## Observations (verified from the JSON)
1. laser_word is the best-counted configuration here: 2 missed words (both in the 0.01-RMS low-level scenario), 0-1 duplicate, 5 wrong digits of 42, 6/6 unknown words rejected.
2. Low-level speech at RMS 0.01 is missed in word modes (gate silence_rms 0.02 for laser_word). This matches the gate setting but has not been separately tested against a gate sweep.
3. Noise at RMS ≥ 0.02-0.08 (real and synthetic) produces spurious `unknown` events in word modes. These are counted as spurious unknown and are not digits. Whether a downstream dashboard should suppress them is an open decision.
4. Sliding configurations produce many repeated decisions (up to 21 duplicates) and many misses. They are not suitable for per-word counting. This is a design limitation, not a bug confirmed in the code.
5. The `decision_latency_s` field is negative in several sliding rows. The field's reference point has not been checked against the code, so latency is not reported here.

## Not established
- Hardware reliability, rig SNR, or accuracy on new speakers.
- The cause of the 5 wrong digits in laser_word has not been individually listened to (see the audio review, which is pending human listening).
- A gate or threshold sweep was not run, and no tuning was done against these results.

## Regression
- scripts/sprint8/test_replay_events.py: 17 passed (results/sprint8_final/replay/pytest_regression_final.log).
