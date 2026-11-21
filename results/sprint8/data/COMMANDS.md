# Sprint 8 — agent A (data & simulation) commands

Environment: cloud dev container (2 vCPU, CPU-only PyTorch 2.15), `. /home/claude/.venv-lmml/bin/activate`,
project root /home/claude/lmml. `pip install tabulate` was added to the venv (used only for the markdown
tables in manifest_summary.md).

Seeds: split seed 42 (`SPLIT_SEED` in build_dataset.py; also seeds the choice of which train digit clips get a
2nd sim variant); sim run seed 2026 (`SIM_RUN_SEED`); per-clip `sim_seed = int(sha256(f"{source_sha256}:{variant}:2026")[:15], 16)`
stored in manifest.csv; check_split.py samples the 20 sanity-listen clips with numpy seed 42 and the 3 figure clips with seed 7.

```bash
cd /home/claude/lmml && . /home/claude/.venv-lmml/bin/activate

# dry run: provenance + split + sim plan only (no audio written), ~5 s
python scripts/sprint8/build_dataset.py --dry-run

# full build: duplicates.csv, split, 9,818 simulated clips -> data/sprint8_sim/, manifest.csv, manifest_summary.md (~75 s)
nohup python scripts/sprint8/build_dataset.py --train-sim-target 7000 > results/sprint8/data/build_dataset.log 2>&1 &

# integrity checks, sanity-listen numbers, figures, SPLIT_VERIFIED marker (~10 s)
python scripts/sprint8/check_split.py
```

Outputs: results/sprint8/data/{manifest.csv, manifest_summary.md, duplicates.csv, split_integrity.json, SPLIT_VERIFIED,
sanity_listen.json, sim_examples.png, snr_histogram.png, build_dataset.log, _hash_cache.json (splits.dedupe_indices md5 cache)}
and data/sprint8_sim/<split>/<label>/<source_stem>__sim<k>.wav (16 kHz PCM_16, 358 MB).

## v2 — regeneration after the simulator start-up-click fix (lead changed `physics_laser_sim.electronics()` to start its filters in steady state)

Identical split, per-clip seeds and params (electronics() consumes no random numbers, so every `sim_seed`/`sim_params`
row is byte-identical to v1; only the sim audio and its sha256 change). v1 (`data/sprint8_sim/`, `manifest.csv`) is kept as
Round 1 evidence and was NOT deleted.

```bash
cd /home/claude/lmml && . /home/claude/.venv-lmml/bin/activate

# regenerate: data/sprint8_sim_v2/, manifest_v2.csv, manifest_summary_v2.md (~105 s)
nohup python scripts/sprint8/build_dataset.py --train-sim-target 7000 --sim-dir data/sprint8_sim_v2 --manifest-suffix _v2 \
      > results/sprint8/data/build_dataset_v2.log 2>&1 &

# checks on v2 + v1-vs-v2 comparison on 200 shared sim clips (numpy seed 2026):
#   split_integrity_v2.json, SPLIT_VERIFIED_V2, sanity_listen_v2.json, sim_examples_v2.png, snr_histogram_v2.png,
#   sim_v1_vs_v2.json, sim_fix_before_after.png
python scripts/sprint8/check_split.py --manifest-suffix _v2 --compare-with results/sprint8/data/manifest.csv
```

Consumers (B, C) should use `manifest_v2.csv` / `SPLIT_VERIFIED_V2`; `manifest.csv` / `SPLIT_VERIFIED` describe the v1 audio with the click.
