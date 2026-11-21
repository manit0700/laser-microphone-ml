#!/usr/bin/env python
"""
replay_phase_sweep.py  (Sprint 8, agent C)
-----------------------------------------
Targeted check for the end-of-word gate: a SHORT real digit (0.22-0.26 s) is
placed at 16 onset offsets (16 ms steps) relative to the dashboard's 256 ms
push boundary, and we count how many of the utterances produce a decision.
A gate that only inspects the last 0.15 s of each pushed chunk has a dead zone
of chunk - 0.15 s, so detection depends on where the word falls.

    LMML_OUTPUT_DIR=results/sprint8/replay python scripts/sprint8/replay_phase_sweep.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "sprint8"))
os.environ.setdefault("LMML_OUTPUT_DIR", str(PROJECT_ROOT / "results" / "sprint8" / "replay"))

import replay_harness as rh  # noqa: E402
import predict as pm  # noqa: E402


def run(chunks=(256, 50), n_clips=6, max_len=0.26, out=None):
    pm.set_checkpoint_path("cnn", PROJECT_ROOT / "models" / "best_model_cnn.pt")
    pool = rh.ClipPool(PROJECT_ROOT / "data/raw", PROJECT_ROOT / "results/sprint8/data/manifest.csv", 8)
    rate = rh.DASH_RATE
    short = []
    for p in pool.digits:
        if any(s in p.stem for s in ("yweweler", "theo", "jackson", "nicolas", "lucas")):
            x = rh.prepare_word(rh.load_clip(p, rate), rate, 0.05)
            if x.size / rate < max_len:
                short.append((p, x))
        if len(short) >= n_clips:
            break
    rep = rh.Replayer("cnn", True, rh.LASER_BANDPASS, rh.LASER_GATE_DEFAULT, autosave=False)
    res = {"clips": [(p.name, round(x.size / rate, 3)) for p, x in short], "chunks": {}}
    for chunk in chunks:
        tot = det = 0
        missed = []
        for p, x in short:
            xw = x
            for k in range(16):
                off = 1.0 + k * 0.016
                s = rh.Stream(rate)
                s.add_silence(off)
                s.add_word(xw, p.stem[0], str(p), "")
                s.add_silence(1.5)
                rep.run(s.render(), rate, chunk)
                tot += 1
                det += int(len(rep.decisions) == 1)
                if len(rep.decisions) != 1:
                    missed.append((p.name, round(off, 3), len(rep.decisions)))
        res["chunks"][str(chunk)] = {"detected": det, "total": tot, "missed": missed}
        print(f"chunk {chunk} ms: exactly one event for {det}/{tot} short-word placements")
    rep.close()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    run(out=sys.argv[1] if len(sys.argv) > 1 else None)
