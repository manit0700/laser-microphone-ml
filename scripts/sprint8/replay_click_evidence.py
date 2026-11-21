#!/usr/bin/env python
"""
replay_click_evidence.py  (Sprint 8, agent C)
--------------------------------------------
How much of the "second bogus event per simulated clip" problem was the
simulator's start-up transient and how much was the live segmentation?

Runs the same 600 test-split sim clips (same seed/selection as the accuracy
set) through the laser-channel replay with
  (a) the FIXED _LiveClassifier, and
  (b) the pre-fix segmentation emulated by monkey-patching the old
      single-frame gate, tick-anchored window and never-firing click test,
for both the v1 (manifest.csv) and v2 (manifest_v2.csv) simulated clips,
and counts clips with 0 / 1 / 2+ events.

    LMML_OUTPUT_DIR=results/sprint8/replay python scripts/sprint8/replay_click_evidence.py \
        results/sprint8/replay/click_evidence_v1_v2.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "sprint8"))
os.environ.setdefault("LMML_OUTPUT_DIR", str(PROJECT_ROOT / "results" / "sprint8" / "replay"))

import torch  # noqa: E402
import replay_harness as rh  # noqa: E402
import predict as pm  # noqa: E402
from signal_backend import _LiveClassifier, BUFFER_SECONDS  # noqa: E402

NEW_GATE, NEW_WIN, NEW_ACT = _LiveClassifier._gate_frames, _LiveClassifier._word_window, _LiveClassifier._active_seconds


def old_gate(self):
    """Pre-fix: loudness of the last 0.15 s of the buffer only; GUI-tick times."""
    rate = self.rate
    now = self._stream_time()
    ctx = self._buf[-int(0.6 * rate):]
    if ctx.size < int(0.4 * rate):
        return None
    f = self._filter(ctx)
    seg = f[-int((0.15 + 0.03) * rate): -int(0.03 * rate)]
    level = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0
    if level >= self.silence_rms:
        if not self._in_word:
            self._in_word = True
            self._word_start = now - 0.15
        self._last_active = now
    return now


def old_window(self, now):
    span = min(now - self._word_start + 0.25, BUFFER_SECONDS)
    return self._buf[-int(max(span, 1.0) * self.rate):]


def old_active(self, x, frame_sec=0.02):
    return 1.0          # the old length test (>= 0.15 s vs 0.12 s) could never reject


def set_mode(old: bool):
    _LiveClassifier._gate_frames = old_gate if old else NEW_GATE
    _LiveClassifier._word_window = old_window if old else NEW_WIN
    _LiveClassifier._active_seconds = old_active if old else NEW_ACT


def main(out=None):
    torch.set_num_threads(1)
    pm.set_checkpoint_path("cnn", PROJECT_ROOT / "models" / "best_model_cnn.pt")
    rate = rh.DASH_RATE
    res = {"machine": rh.MACHINE, "model": "models/best_model_cnn.pt", "n_per_set": 600, "sets": {}}
    for ver, manifest in (("v1", "results/sprint8/data/manifest.csv"), ("v2", "results/sprint8/data/manifest_v2.csv")):
        pool = rh.ClipPool(PROJECT_ROOT / "data/raw", PROJECT_ROOT / manifest, 8)
        _, sim, _ = rh.select_accuracy_rows(PROJECT_ROOT / manifest, pool, 400, 600, 8)
        res["sets"][ver] = {"manifest": manifest}
        for mode in ("fixed_code", "prefix_code_emulated"):
            set_mode(mode != "fixed_code")
            rep = rh.Replayer("cnn", True, rh.LASER_BANDPASS, rh.LASER_GATE_DEFAULT, autosave=False)
            counts = {"0": 0, "1": 0, "2+": 0}
            first_is_click = 0
            by_cond = {}
            for r in sim:
                xw = rh.prepare_word(rh.load_clip(PROJECT_ROOT / r["file"], rate), rate, 0.05)
                s = rh.Stream(rate); s.add_silence(1.0); s.add_word(xw, r["label"], r["file"], ""); s.add_silence(1.2)
                rep.run(s.render(), rate, 256)
                n = len(rep.decisions)
                key = "0" if n == 0 else ("1" if n == 1 else "2+")
                counts[key] += 1
                c = by_cond.setdefault(r["condition"], {"0": 0, "1": 0, "2+": 0})
                c[key] += 1
                # a decision that ends within 0.4 s of the clip start = the head transient
                if n >= 2 and rep.decisions[0]["word_end"] <= s.events[0]["t_start"] + 0.4:
                    first_is_click += 1
            rep.close()
            res["sets"][ver][mode] = {"events_per_clip": counts, "multi_with_head_click": first_is_click,
                                      "by_condition": by_cond}
            print(ver, mode, counts, "head-click multis:", first_is_click)
    set_mode(False)
    if out:
        Path(out).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
