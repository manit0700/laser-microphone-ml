# Sprint 8 final: summary

**Software:** the frozen historical checkpoint finetuned_v2 is packaged with a checksum, labels, sample rate, filter path, temperature 0.572, and threshold 0.26. The standalone predictor passes its self-test and 14 verification checks in fresh processes, run from outside the repository, and reproduces the repository pipeline exactly on 20 validation clips. Error paths are understandable.

**Replay (validation clips only):** laser_word counts 2 missed words, 0-1 duplicates, and 5 wrong digits out of 42 digits, with 6/6 unknown words rejected. Noise-only input produces spurious unknown events. Sliding modes are unsuitable for word counting. This is software replay, not hardware evidence.

**Audio:** objective checks are complete (57/57 match sources, predictions, and labels; no flat-top clipping). No listening has been done; every label remains unconfirmed.

**Not done:** human listening, hardware (blocked), Level 1 / Level 3 definitions (missing), the final report, transfer to the Mac, and the human fields (due date, hours, sprint dates, peer scores).

**Status:** the model is an experimental historical candidate, not a validated laser-rig system.
