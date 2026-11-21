# Sprint 8 data manifest_v2 summary (agent A)

Generated 2026-11-08 12:32:47 by scripts/sprint8/build_dataset.py (split seed 42, sim run seed 2026); sim audio under data/sprint8_sim_v2, manifest results/sprint8/data/manifest_v2.csv.

Rows: 19027 = 9209 clean + 9818 sim. Originals dropped as duplicates: 473 (see duplicates.csv).

## Clip counts per split x kind

| split   |   clean |   sim |   All |
|:--------|--------:|------:|------:|
| train   |    6391 |  7000 | 13391 |
| val     |    1433 |  1433 |  2866 |
| test    |    1385 |  1385 |  2770 |
| All     |    9209 |  9818 | 19027 |

Clean clips per split: train 6391 (69.4%), val 1433 (15.6%), test 1385 (15.0%)

## Clean clips per split x dataset

| dataset         |   train |   val |   test |   All |
|:----------------|--------:|------:|-------:|------:|
| fsdd            |    2000 |   500 |    500 |  3000 |
| noise           |     174 |    38 |     38 |   250 |
| sc_unknown      |     627 |   113 |     60 |   800 |
| silence         |      56 |    12 |     12 |    80 |
| speech_commands |    3534 |   770 |    683 |  4987 |
| team            |       0 |     0 |     92 |    92 |
| All             |    6391 |  1433 |   1385 |  9209 |

## Sim clips per split x dataset

| dataset         |   train |   val |   test |   All |
|:----------------|--------:|------:|-------:|------:|
| fsdd            |    2230 |   500 |    500 |  3230 |
| noise           |     174 |    38 |     38 |   250 |
| sc_unknown      |     627 |   113 |     60 |   800 |
| silence         |      56 |    12 |     12 |    80 |
| speech_commands |    3913 |   770 |    683 |  5366 |
| team            |       0 |     0 |     92 |    92 |
| All             |    7000 |  1433 |   1385 |  9818 |

## Clips per split x kind x label

|                    |   0 |   1 |   2 |   3 |   4 |   5 |   6 |   7 |   8 |   9 |   unknown |
|:-------------------|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----------:|
| ('train', 'clean') | 557 | 546 | 554 | 554 | 563 | 561 | 552 | 552 | 548 | 547 |       857 |
| ('train', 'sim')   | 615 | 615 | 607 | 631 | 624 | 614 | 614 | 604 | 608 | 611 |       857 |
| ('val', 'clean')   | 123 | 126 | 124 | 124 | 117 | 132 | 131 | 128 | 131 | 134 |       163 |
| ('val', 'sim')     | 123 | 126 | 124 | 124 | 117 | 132 | 131 | 128 | 131 | 134 |       163 |
| ('test', 'clean')  | 125 | 135 | 128 | 129 | 129 | 116 | 124 | 131 | 132 | 126 |       110 |
| ('test', 'sim')    | 125 | 135 | 128 | 129 | 129 | 116 | 124 | 131 | 132 | 126 |       110 |

## Speakers per split (clean)

- **train**: 276 groups: fsdd 4, noise 4, sc_unknown 62, silence 1, speech_commands 267
  - fsdd: ['jackson', 'lucas', 'nicolas', 'theo']; team: []; generated groups: ['gen_noise_bandlimited_train', 'gen_noise_brown_train', 'gen_noise_hum_click_train', 'gen_noise_white_train', 'gen_silence_white_train']
- **val**: 63 groups: fsdd 1, noise 4, sc_unknown 8, silence 1, speech_commands 57
  - fsdd: ['george']; team: []; generated groups: ['gen_noise_bandlimited_val', 'gen_noise_brown_val', 'gen_noise_hum_click_val', 'gen_noise_white_val', 'gen_silence_white_val']
- **test**: 45 groups: fsdd 1, noise 4, sc_unknown 5, silence 1, speech_commands 38, team 1
  - fsdd: ['yweweler']; team: ['manit']; generated groups: ['gen_noise_bandlimited_test', 'gen_noise_brown_test', 'gen_noise_hum_click_test', 'gen_noise_white_test', 'gen_silence_white_test']

Speech Commands hashes appearing in both digits and unknown words: 75; all such hashes are in a single split (verified by check_split.py).

## Condition histogram (sim)

| split   |   sim_good |   sim_medium |   sim_poor |   All |
|:--------|-----------:|-------------:|-----------:|------:|
| train   |       2338 |         2388 |       2274 |  7000 |
| val     |        487 |          492 |        454 |  1433 |
| test    |        493 |          454 |        438 |  1385 |
| All     |       3318 |         3334 |       3166 |  9818 |

snr_db (sim): min 0.0, median 15.1, max 30.0. Bins per PLAN.md: sim_good snr>=20, sim_medium 10<=snr<20, sim_poor snr<10.

## Decisions

- **Team speaker `manit`** (92 unique recordings after dropping 460 `_dupN` copies, 48 kHz): team recording, recorder unknown. Placed ENTIRELY in test; never in train or val.
- **FSDD**: 6 speakers -> 4 train / 1 val / 1 test, shuffled with seed 42.
- **Speech Commands**: digit clips and 'unknown' word clips of the same 8-hex speaker hash are one group (splits.speaker_of gives the hash for both) and land in the same split.
- **noise_*/silence_* unknown clips are SYNTHETIC (synthetic_noise=1).** Evidence: every clip is exactly 1.000 s and peak-normalised to 1.0; `silence_*` (8 kHz) is Gaussian white noise (flat spectrum, kurtosis 3.0, zero-crossing rate 0.5); `noise_*` (16 kHz) falls into four generator families by spectral slope / kurtosis / zero-crossing rate: {'brown': 58, 'white': 59, 'hum_click': 68, 'bandlimited': 65} (brown = 1/f^2 spectrum with a random DC offset, i.e. a random walk, files 0-58; white = flat, files 59-117; hum_click = 60/180 Hz hum plus impulsive clicks, kurtosis 5-200, files ~118-178; bandlimited = coloured noise peaking at ~450-1500 Hz, files ~179-249; the families were written in index order, the classifier is a statistics heuristic so the two noisy families' boundary is approximate). Adjacent files are uncorrelated (|r| < 0.13, tail/head r ~ 0), so they are not chunks of one longer recording. No generator script exists in the repo; the statistics alone show they are generated, not recorded.
- **Grouping of generated clips**: there is no speaker identity to leak, and sharing a noise *distribution* between splits is the same as sharing it with the deployment environment. Each family is cut into contiguous index blocks (~70/15/15), one block per split, so every split has some noise and silence and any hypothetical temporal adjacency stays inside a split. `speaker` for these rows is the block id (e.g. `gen_noise_white_train`), which is unique to its split by construction.
- **Sim seeds**: sim_seed = int(sha256(f'{source_sha256}:{variant}:2026')[:15], 16); rng = numpy.random.default_rng(sim_seed); simulate(clean, rng) from scripts/physics_laser_sim.py unchanged (domain-randomised, no calibration). Output via preprocess.save_capture at 16 kHz.
- **Sim counts**: 1 variant per kept clip in every split, plus a seeded random subset of train digit clips gets a 2nd variant so the train sim total hits the 6,000-8,000 cap.
