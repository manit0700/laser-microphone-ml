# Sprint 8 final: handoff

## What is ready
- Package: results/sprint8_final/package/ (README.md, MODEL_CARD.md, SHA256SUMS, config, predict_package.py, VERIFICATION_LOG.md).
- Replay: results/sprint8_final/replay/ (REPLAY_ANALYSIS.md, expected_vs_observed_events.csv/.md, scenario JSON, regression log).
- Audio review: results/sprint8_final/audio_review/ (REVIEW.md, label_review_status.csv). **No label has been human-confirmed.**
- Scripts: scripts/sprint8_final/ (package_model.py, verify_package.py, audio_review_objective.py, replay_summary.py).

## Verified now
- Package self-test and 14 verification checks pass (VERIFICATION_LOG.md).
- Replay counts: laser_word is the best-counted configuration (2 missed / 48 words, 5 wrong of 42 digits, 6/6 unknown rejected). Sliding configurations are not suitable for word counting.

## Exact commands
```bash
cd /home/claude/lmml/results/sprint8_final/package
sha256sum -c SHA256SUMS
python predict_package.py --selftest
python predict_package.py clean <wav>            # laser: use 'laser' mode
python /home/claude/lmml/scripts/sprint8_final/verify_package.py     # regenerate VERIFICATION_LOG.md (creates scratch in /tmp)
python /home/claude/lmml/scripts/sprint8_final/replay_summary.py     # regenerate replay tables
```
Note: verify_package.py creates its scratch folder under the system temp directory.

## Blocked and missing
1. **Listening review:** 57 clips need human listening. Fill the human_* columns in label_review_status.csv. Do not infer labels from the machine predictions.
2. **Hardware (rig and Jetson):** not reachable from this container. Resume steps: (a) connect the rig and Jetson to the Mac project; (b) record the configuration without changing laser power; (c) evaluate the frozen package before any new recordings; (d) measure latency and FP16/TensorRT only if supported, labelling each result Jetson or CPU.
3. **Level 1 / Level 3:** no definition in the project. Please state what these mean (safety class, device setting, or team configuration).
4. **Human fields:** submission due date and timezone, actual hours, sprint dates, peer-review scores. Left blank; not invented.
5. **Final report:** Manit_Dankhara_Sprint_8_Final_Review.tex/.pdf (7 pages) is in the reports folder. Open items are listed in its Section 12.
6. **Laser specification:** the hardware teammate's wording ("grade level 1", "grade level 3") is unconfirmed. Requested: model numbers, datasheets, wavelength, output power, and whether the second laser is Class 3R or 3B. See HARDWARE_SPEC_MISSING.md.
7. **Transfer:** done. The archive sprint8_final_r3.tar.gz and sprint8_final_files.sha256 are in laser_microphone_ml/transfer/.
