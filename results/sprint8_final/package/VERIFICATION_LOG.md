# Package verification log (lmml-sprint8-finetuned_v2)

Overall: **PASS** (15/15 checks)

| check | result |
|---|---|
| fresh process, clean mode, relative paths, cwd outside repo | PASS |
| fresh process, laser mode, relative paths, cwd outside repo | PASS |
| built-in self-test (labels vs repository pipeline, 12 clean + 12 laser) | PASS |
| repository predictor (predict._probs / load_live_path) probabilities written | PASS |
| package probabilities written (same process type as CLI) | PASS |
| numeric agreement clean (20 validation clips) | PASS |
| numeric agreement laser (20 validation clips) | PASS |
| env override LMML_CKPT_CNN pointing to another file is ignored | PASS |
| output reports the pinned checkpoint sha (not deployed models/) | PASS |
| missing input file -> exit 4 with message | PASS |
| missing checkpoint -> exit 3 with message | PASS |
| corrupted checkpoint -> exit 3 checksum message | PASS |
| invalid config -> exit 2 message | PASS |
| missing repository -> exit 2 message | PASS |
| no mode/file -> usage message | PASS |
