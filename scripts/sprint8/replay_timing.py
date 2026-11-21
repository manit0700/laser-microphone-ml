#!/usr/bin/env python
"""
replay_timing.py  (Sprint 8, agent C)
------------------------------------
Per-stage wall-clock timing of the live classifier path, measured on this
machine ("cloud dev container, 2 vCPU, CPU-only PyTorch"). NOT Jetson latency.

Replays N single-word streams through _LiveClassifier (laser channel settings:
CNN, 200-3800 Hz band-pass, end-of-word mode) at the dashboard's 256 ms push
size and records, per word decision: band-pass of the word window, enhance,
preprocess (+ features), model forward, autosave, and the whole predict() call;
plus the per-tick cost of the gate (band-pass of the new audio) on idle ticks.

    LMML_OUTPUT_DIR=results/sprint8/replay python scripts/sprint8/replay_timing.py \
        --out results/sprint8/replay/timing.json [--model ensemble] [--threads 1]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "sprint8"))
os.environ.setdefault("LMML_OUTPUT_DIR", str(PROJECT_ROOT / "results" / "sprint8" / "replay"))

import torch  # noqa: E402
import replay_harness as rh  # noqa: E402
import predict as pm  # noqa: E402


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn", choices=["cnn", "ensemble"])
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--chunk-ms", type=float, default=256.0)
    ap.add_argument("--autosave", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    pm.set_checkpoint_path("cnn", Path(args.model_path) if args.model_path else PROJECT_ROOT / "models" / "best_model_cnn.pt")
    pm.set_checkpoint_path("lstm", PROJECT_ROOT / "models" / "best_model.pt")

    pool = rh.ClipPool(PROJECT_ROOT / "data/raw", PROJECT_ROOT / "results/sprint8/data/manifest.csv", 8)
    rate = rh.DASH_RATE
    bp = rh.LASER_BANDPASS if args.model == "cnn" else None
    gate = rh.LASER_GATE_DEFAULT if args.model == "cnn" else rh.STD_GATE_DEFAULT
    rep = rh.Replayer(args.model, True, bp, gate, autosave=args.autosave, source_label="replay-timing")
    # warm-up (first call pays filter design / lazy imports)
    x = rh.prepare_word(rh.load_clip(pool.pick_digit("5"), rate), rate, 0.05)
    s = rh.Stream(rate); s.add_silence(1.0); s.add_word(x, "5", "", ""); s.add_silence(1.2)
    rep.run(s.render(), rate, args.chunk_ms)
    warm = dict(rep.decisions[0]["stages_ms"]) if rep.decisions else {}

    stages: dict[str, list] = {}
    idle = []
    load_before = os.getloadavg()
    t0 = time.time()
    for i in range(args.n):
        p = pool.pick_digit(str(i % 10))
        x = rh.prepare_word(rh.load_clip(p, rate), rate, 0.05)
        s = rh.Stream(rate); s.add_silence(1.0); s.add_word(x, p.stem[0], "", ""); s.add_silence(1.2)
        st = rep.run(s.render(), rate, args.chunk_ms)
        idle.append(st["idle_tick_gate_filter_ms_mean"])
        for d in rep.decisions:
            for k, v in d["stages_ms"].items():
                stages.setdefault(k, []).append(v)
    load_after = os.getloadavg()
    rep.close()

    summary = {k: {"n": len(v), "min_ms": float(np.min(v)), "median_ms": float(np.median(v)),
                   "p95_ms": pct(v, 95), "max_ms": float(np.max(v))} for k, v in sorted(stages.items())}
    out = {
        "machine": rh.MACHINE, "note": "software replay on the dev container; not Jetson/hardware latency",
        "python": platform.python_version(), "torch": torch.__version__, "torch_threads": args.threads,
        "cpu_count": os.cpu_count(), "loadavg_1m_before": load_before[0], "loadavg_1m_after": load_after[0],
        "contended": max(load_before[0], load_after[0]) > 1.5,
        "model": args.model, "chunk_ms": args.chunk_ms, "stream_rate": rate, "n_words": args.n,
        "autosave": args.autosave, "wall_s": time.time() - t0,
        "first_call_warmup_ms": warm,
        "per_word_stage_ms": summary,
        "idle_tick_gate_filter_ms_mean": float(np.mean(idle)),
    }
    print(json.dumps({k: v for k, v in out.items() if k != "per_word_stage_ms"}, indent=1))
    for k, v in summary.items():
        print(f"{k:<18} median {v['median_ms']:8.2f} ms   p95 {v['p95_ms']:8.2f} ms   min {v['min_ms']:8.2f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
