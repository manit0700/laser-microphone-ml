# Laser Microphone digit classifier: package README (lmml-sprint8-finetuned_v2)

Status: **experimental historical Sprint 8 candidate.** Not validated on the laser rig or Jetson. Read MODEL_CARD.md first.

## Contents
| path | what |
|---|---|
| model/finetuned_v2_cnn.pt | checkpoint (copy of results/sprint8/models/finetuned_v2_cnn.pt, same bytes) |
| config/model_config.json | labels, sample rate, feature, temperature, threshold, mode definitions |
| SHA256SUMS | checksum of the checkpoint (`sha256sum -c SHA256SUMS` from this folder) |
| predict_package.py | standalone predictor (clean or laser mode, JSON output, self-test) |
| VERIFICATION_LOG.md / verification.json | results of scripts/sprint8_final/verify_package.py |

## Dependencies
- Python 3.10+ with torch, torchaudio, numpy, scipy, soundfile (verified with torch 2.15 CPU in /home/claude/.venv-lmml).
- The predictor imports the repository code (`src/` and `scripts/sprint8/`). It does **not** bundle that code. Copy the repository, or point `--repo` at a copy.

## Commands (run from this folder unless noted)
```bash
sha256sum -c SHA256SUMS                                  # checksum
python predict_package.py --selftest                      # 12 clean + 12 laser validation examples vs repository pipeline
python predict_package.py clean  path/to/digit.wav       # file as recorded (clean path)
python predict_package.py laser  path/to/laser.wav --json out.json   # laser path: band-pass 200-3800 Hz + enhancement
python predict_package.py clean a.wav b.wav --repo /path/to/laser_microphone_ml
```
Environment: `LMML_REPO=/path/to/laser_microphone_ml` is equivalent to `--repo`.

## Input / output example
```
$ python predict_package.py clean <val clean wav>
<file>	<label>	conf=0.812	unknown=False
```
Each result has: `file`, `mode`, `label` (0-9 or `unknown`), `confidence` (max probability after temperature scaling), `rejected_as_unknown`, `checkpoint_sha256`. Output is an 11-way decision on one clip of at most 1.0 s at 8 kHz.

## Error behaviour (verified; exit codes)
| condition | exit | message starts with |
|---|---|---|
| config missing or invalid JSON | 2 | `config not found` / `config is not valid JSON` |
| repository src/ missing | 2 | `repository src/ not found` |
| bad usage | 2 | `usage` |
| checkpoint missing | 3 | `checkpoint not found` |
| checkpoint checksum mismatch | 3 | `checkpoint checksum mismatch` |
| input WAV missing | 4 | `input file not found` |

## Known limitations
- Development demo only. No hardware, real-laser, or Jetson accuracy is claimed.
- The historical test results used speaker-grouped data; this package adds no new evaluation.
- Single-clip decisions. Streaming behaviour (windows, gate, repeats) is in the replay analysis, not in this predictor.
- The "laser" mode reproduces the training and dashboard filter path. It does not model the physical laser, its power, or the detector.

See MODEL_CARD.md for provenance, intended use, and the decision rule.
