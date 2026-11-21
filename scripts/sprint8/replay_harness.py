#!/usr/bin/env python
"""
replay_harness.py  (Sprint 8, agent C)
=====================================
Software replay of the dashboard's live classifier path WITHOUT any microphone,
ADC or GUI.

It builds synthetic audio streams from REAL clips in data/raw (digits, non-digit
words, noise), feeds them to `signal_backend._LiveClassifier` in chunks exactly
the way signal_dashboard2.py does (push() a chunk, then predict(), once per GUI
tick), and counts PREDICTION EVENTS explicitly through the classifier's
`on_decision` callback -- not by comparing returned (label, confidence) tuples,
which are identical for two consecutive identical words.

Two kinds of metrics are kept apart on purpose:
  * event handling  : words vs. events (missed words, duplicate events, false
                      activations, decision latency) -- the deterministic part;
  * recognition     : whether the label of a matched event is right, and whether
                      non-speech / non-digit words are rejected as Unknown.

Usage (from the project root, inside the lmml venv):
    LMML_OUTPUT_DIR=results/sprint8/replay python scripts/sprint8/replay_harness.py \
        --out-dir results/sprint8/replay            # deterministic scenarios, deployed CNN
    ... --model-path results/sprint8/models/finetuned_cnn.pt --tag finetuned --accuracy

See results/sprint8/replay/replay_results.md for what each scenario means.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# LMML_OUTPUT_DIR must be set BEFORE config is imported (it decides where
# autosave/log files go). Default it to results/sprint8/replay so nothing ever
# lands in the project's results/ root.
os.environ.setdefault("LMML_OUTPUT_DIR", str(PROJECT_ROOT / "results" / "sprint8" / "replay"))

import torch  # noqa: E402
import torchaudio.functional as AF  # noqa: E402
import soundfile as sf  # noqa: E402

import signal_backend as sb  # noqa: E402
from signal_backend import _LiveClassifier  # noqa: E402
import predict as predict_mod  # noqa: E402
import utils as utils_mod  # noqa: E402
from config import SAMPLE_RATE  # noqa: E402
import splits  # noqa: E402

MACHINE = f"cloud dev container, 2 vCPU, CPU-only PyTorch {torch.__version__.split('+')[0]}"

# --- dashboard constants mirrored here (signal_dashboard2.py) ------------------
DASH_RATE = 48000              # PCM1808 stream rate the dashboard pushes at
DASH_CHUNK = 4096 * 3          # frames per hardware read -> 256 ms per push
LASER_BANDPASS = (200.0, 3800.0)
LASER_GATE_DEFAULT = 0.02      # LMML_LASER_SILENCE_RMS default when band-passed
STD_GATE_DEFAULT = sb.DAQ_SILENCE_RMS  # 0.03, std-mic channel (no band-pass)

DIGITS = [str(d) for d in range(10)]


# =============================================================================
# Clip loading / stream construction
# =============================================================================
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 16), b""):
            h.update(blk)
    return h.hexdigest()


def load_clip(path: Path, rate: int) -> np.ndarray:
    """Mono float32 samples of a WAV at `rate` (resampled with torchaudio)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = torch.from_numpy(np.asarray(data, dtype=np.float32).mean(axis=1))
    if sr != rate:
        x = AF.resample(x, orig_freq=int(sr), new_freq=int(rate))
    return x.numpy().astype(np.float32)


def trim_word(x: np.ndarray, rate: int, rel_db: float = -40.0, margin_s: float = 0.03) -> np.ndarray:
    """Cut a clip to its active region (frame energy within rel_db of the peak frame).

    Gives the harness tight word boundaries so the expected event timeline is
    accurate. Trailing/leading room noise in the clip is dropped; the harness
    adds its own gaps and (optionally) background noise.
    """
    frame = int(0.02 * rate)
    if x.size < 2 * frame:
        return x
    n = x.size // frame
    e = (x[: n * frame].reshape(n, frame) ** 2).mean(axis=1)
    peak = e.max()
    if peak <= 0:
        return x
    thr = peak * 10 ** (rel_db / 10.0)
    idx = np.where(e >= thr)[0]
    a = max(0, int(idx[0]) * frame - int(margin_s * rate))
    b = min(x.size, (int(idx[-1]) + 1) * frame + int(margin_s * rate))
    return x[a:b]


def set_rms(x: np.ndarray, target: float) -> np.ndarray:
    r = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    return (x * (target / r)).astype(np.float32) if r > 0 else x


_SOS_CACHE: dict = {}


def speech_band(x: np.ndarray, rate: int, band=LASER_BANDPASS) -> np.ndarray:
    """Zero-phase 200-3800 Hz band-pass (same shape as _LiveClassifier._filter)."""
    from scipy.signal import butter, sosfiltfilt
    key = (rate, band)
    if key not in _SOS_CACHE:
        _SOS_CACHE[key] = butter(4, [band[0], min(band[1], rate / 2 * 0.95)], btype="band", fs=rate, output="sos")
    return sosfiltfilt(_SOS_CACHE[key], x).astype(np.float32) if x.size > 64 else x


def prepare_word(x: np.ndarray, rate: int, speech_rms: float) -> np.ndarray:
    """Cut a clip to its active region and set its level, both measured in the
    speech band (200-3800 Hz) -- the band the laser channel's gate and model see.

    `speech_rms` is therefore "band RMS of the word", comparable to the
    dashboard's LMML_LASER_SILENCE_RMS (quiet ~0.013 / speech ~0.03 measured on
    the real rig after the same band-pass). Out-of-band hum or rumble in a clip
    (simulated laser audio, some Speech Commands recordings) does not inflate
    the level. The returned samples are the UNFILTERED clip: the classifier
    does its own filtering.
    """
    xb = speech_band(x, rate)
    frame = int(0.02 * rate)
    n = xb.size // frame
    if n < 2:
        return set_rms(x, speech_rms)
    e = (xb[: n * frame].reshape(n, frame) ** 2).mean(axis=1)
    peak = e.max()
    if peak <= 0:
        return x
    idx = np.where(e >= peak * 10 ** (-40 / 10.0))[0]
    a = max(0, int(idx[0]) * frame - int(0.03 * rate))
    b = min(x.size, (int(idx[-1]) + 1) * frame + int(0.03 * rate))
    r = float(np.sqrt(np.mean(xb[a:b] ** 2)))
    return (x[a:b] * (speech_rms / r)).astype(np.float32) if r > 0 else x[a:b]


class ClipPool:
    """Picks real clips from data/raw, optionally restricted by a Sprint 8 manifest."""

    def __init__(self, raw_dir: Path, manifest: Path | None, seed: int, split: str = "test"):
        self.raw_dir = raw_dir
        self.rng = random.Random(seed)
        self.manifest_used = False
        self.split = split
        all_digits = sorted(p for p in raw_dir.glob("[0-9]_*.wav") if "_dup" not in p.stem)
        self.unknown_words = sorted(p for p in (raw_dir / "unknown").glob("*.wav")
                                    if not p.stem.startswith(("noise", "silence")))
        self.noise_clips = sorted((raw_dir / "unknown").glob("noise_*.wav"))
        if manifest is not None and manifest.exists():
            keep = set()
            with open(manifest, newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("split") == split and row.get("kind") == "clean" and \
                            row.get("label") in DIGITS and row.get("dataset") in ("fsdd", "speech_commands", "team"):
                        keep.add((PROJECT_ROOT / row["file"]).resolve())
            restricted = [p for p in all_digits if p.resolve() in keep]
            if restricted:
                all_digits = restricted
                self.manifest_used = True
        self.digits = all_digits
        self.by_label: dict[str, list[Path]] = {d: [] for d in DIGITS}
        for p in self.digits:
            self.by_label[p.stem.split("_")[0]].append(p)

    def speaker(self, p: Path) -> str:
        return splits.speaker_of(p)

    def pick_digit(self, label: str, speakers_used: set | None = None) -> Path:
        cands = self.by_label[label]
        if speakers_used is not None:
            fresh = [p for p in cands if self.speaker(p) not in speakers_used]
            if fresh:
                cands = fresh
        p = self.rng.choice(cands)
        if speakers_used is not None:
            speakers_used.add(self.speaker(p))
        return p

    def pick_unknown_word(self) -> Path:
        return self.rng.choice(self.unknown_words)

    def pick_noise(self) -> Path:
        return self.rng.choice(self.noise_clips)


class Stream:
    """A synthetic stream + the timeline of words it contains."""

    def __init__(self, rate: int):
        self.rate = rate
        self.parts: list[np.ndarray] = []
        self.events: list[dict] = []
        self.t = 0.0

    def add_silence(self, sec: float):
        n = int(round(sec * self.rate))
        self.parts.append(np.zeros(n, dtype=np.float32))
        self.t += n / self.rate

    def add_word(self, x: np.ndarray, label: str, source: str, speaker: str):
        self.events.append({"t_start": round(self.t, 4), "t_end": round(self.t + x.size / self.rate, 4),
                            "label": label, "source": source, "speaker": speaker})
        self.parts.append(x.astype(np.float32))
        self.t += x.size / self.rate

    def render(self, noise: np.ndarray | None = None, noise_rms: float = 0.0, seed: int = 0) -> np.ndarray:
        sig = np.concatenate(self.parts) if self.parts else np.zeros(0, np.float32)
        if noise_rms > 0:
            if noise is None:                              # synthetic white noise
                g = np.random.default_rng(seed)
                nz = g.standard_normal(sig.size).astype(np.float32)
            else:                                          # tiled real noise clip
                reps = int(np.ceil(sig.size / max(1, noise.size)))
                nz = np.tile(noise, reps)[: sig.size]
            sig = sig + set_rms(nz, noise_rms)
        return np.clip(sig, -1.0, 1.0).astype(np.float32)


# =============================================================================
# Replayer: drives _LiveClassifier like the dashboard timer does
# =============================================================================
class StageTimer:
    """Wraps the pipeline stages the classifier calls, so each decision gets
    per-stage wall-clock timings (ms). Restored by .close()."""

    def __init__(self, clf: _LiveClassifier):
        self.clf = clf
        self.cur: dict[str, float] = {}
        self._orig = {}
        self._hooks = []

        def wrap(mod, name, key):
            fn = getattr(mod, name)
            self._orig[(mod, name)] = fn

            def w(*a, **k):
                t0 = time.perf_counter()
                out = fn(*a, **k)
                self.cur[key] = self.cur.get(key, 0.0) + (time.perf_counter() - t0) * 1e3
                return out
            setattr(mod, name, w)

        wrap(sb, "enhance_waveform", "enhance_ms")
        wrap(predict_mod, "preprocess_waveform", "preprocess_ms")
        wrap(predict_mod, "extract_features", "features_ms")
        wrap(utils_mod, "log_prediction", "autosave_ms")
        # band-pass: instance-level wrap of _filter; calls made from inside the
        # gate (_gate_frames) are booked as gate_filter_ms, the rest (the word
        # window / sliding window) as word_filter_ms.
        orig_filter = clf._filter
        self._in_gate = False

        def filt(x):
            t0 = time.perf_counter()
            out = orig_filter(x)
            key = "gate_filter_ms" if self._in_gate else "word_filter_ms"
            self.cur[key] = self.cur.get(key, 0.0) + (time.perf_counter() - t0) * 1e3
            return out
        clf._filter = filt
        self._orig_filter = orig_filter
        if hasattr(clf, "_gate_frames"):
            orig_gate = clf._gate_frames

            def gate():
                self._in_gate = True
                try:
                    return orig_gate()
                finally:
                    self._in_gate = False
            clf._gate_frames = gate
            self._orig_gate = orig_gate
        else:
            self._orig_gate = None
        # model forward(s)
        for mt in (["lstm", "cnn"] if clf.model == "ensemble" else [clf.model]):
            m = predict_mod._load(mt)["model"]
            st = {}
            h1 = m.register_forward_pre_hook(lambda mod, inp, st=st: st.__setitem__("t0", time.perf_counter()))
            h2 = m.register_forward_hook(
                lambda mod, inp, out, st=st: self.cur.__setitem__(
                    "forward_ms", self.cur.get("forward_ms", 0.0) + (time.perf_counter() - st["t0"]) * 1e3))
            self._hooks += [h1, h2]

    def take(self) -> dict:
        d, self.cur = self.cur, {}
        return d

    def close(self):
        for (mod, name), fn in self._orig.items():
            setattr(mod, name, fn)
        self.clf._filter = self._orig_filter
        if self._orig_gate is not None:
            self.clf._gate_frames = self._orig_gate
        for h in self._hooks:
            h.remove()


class Replayer:
    def __init__(self, model: str, word_mode: bool, bandpass, silence_rms: float,
                 autosave: bool = True, source_label: str = "replay"):
        self.decisions: list[dict] = []
        self.stream_time = 0.0
        self.clf = _LiveClassifier(model=model, silence_rms=silence_rms, bandpass=bandpass,
                                   word_mode=word_mode, autosave=autosave, source_label=source_label,
                                   on_decision=self._on_decision, clock=lambda: self.stream_time)
        if not self.clf.model_available:
            raise RuntimeError(self.clf.model_error)
        self.word_mode = word_mode
        self.timer = StageTimer(self.clf)
        self.returned: list[tuple] = []     # (stream_time, predict() return) per tick

    def _on_decision(self, info: dict):
        self.decisions.append(dict(info))

    def run(self, signal: np.ndarray, rate: int, chunk_ms: float) -> dict:
        self.clf.reset()
        # reset() keeps the sliding-mode throttle stamp; our clock restarts at 0 for
        # every stream (the real dashboard's time.monotonic never goes backwards).
        self.clf._last_predict_t = 0.0
        self.stream_time = 0.0
        self.decisions.clear()
        self.returned.clear()
        n = int(round(rate * chunk_ms / 1000.0))
        t_wall0 = time.perf_counter()
        pred_ms, gate_ms = [], []
        self.timer.take()
        for i in range(0, signal.size, n):
            chunk = signal[i:i + n]
            self.clf.push(chunk, rate)
            self.stream_time = self.clf._stream_time()
            n_before = len(self.decisions)
            t0 = time.perf_counter()
            r = self.clf.predict()
            dt = (time.perf_counter() - t0) * 1e3
            stages = self.timer.take()          # stage times spent inside THIS tick
            if len(self.decisions) > n_before:   # a decision was made in this tick
                stages["predict_call_ms"] = dt
                self.decisions[-1]["stages_ms"] = stages
            else:
                gate_ms.append(stages.get("gate_filter_ms", 0.0))
            pred_ms.append(dt)
            self.returned.append((round(self.stream_time, 4), r))
        return {"ticks": len(pred_ms), "wall_s": time.perf_counter() - t_wall0,
                "predict_call_ms_mean": float(np.mean(pred_ms)) if pred_ms else 0.0,
                "predict_call_ms_max": float(np.max(pred_ms)) if pred_ms else 0.0,
                "idle_tick_gate_filter_ms_mean": float(np.mean(gate_ms)) if gate_ms else 0.0}

    def close(self):
        self.timer.close()


# =============================================================================
# Matching events to the expected timeline
# =============================================================================
def match_events(expected: list[dict], decisions: list[dict], word_mode: bool,
                 window_sec: float = sb.PREDICT_WINDOW_SEC, tol: float = 0.15) -> dict:
    """Attribute each decision to at most one expected word; score both ways."""
    per_word = [dict(e, decisions=[]) for e in expected]
    unmatched = []
    for d in decisions:
        if word_mode and "word_start" in d:
            a, b = d["word_start"] - tol, d["word_end"] + tol
        else:
            b = d["stream_time"] + tol
            a = d["stream_time"] - window_sec - tol
        # candidate words overlapping [a, b]; pick the one with the largest overlap
        best, best_ov = None, 0.0
        for w in per_word:
            ov = min(b, w["t_end"]) - max(a, w["t_start"])
            if ov > best_ov:
                best, best_ov = w, ov
        if best is None:
            unmatched.append(d)
        else:
            best["decisions"].append(d)

    n_words = len(expected)
    n_digit_words = sum(1 for e in expected if e["label"] in DIGITS)
    n_unknown_words = n_words - n_digit_words
    missed = [w for w in per_word if not w["decisions"]]
    dup_events = sum(max(0, len(w["decisions"]) - 1) for w in per_word)
    false_act = [d for d in unmatched if d["label"] != "unknown"]
    spurious_unknown = [d for d in unmatched if d["label"] == "unknown"]

    # recognition on matched words: use the FIRST decision for the word
    correct = wrong = rejected = 0            # digit words
    unk_rejected = unk_accepted = 0           # non-digit words
    latencies = []
    rows = []
    for w in per_word:
        first = w["decisions"][0] if w["decisions"] else None
        got = first["label"] if first else None
        if first:
            latencies.append(first["stream_time"] - w["t_end"])
        if w["label"] in DIGITS:
            if got is None:
                outcome = "missed"
            elif got == w["label"]:
                correct += 1; outcome = "correct"
            elif got == "unknown":
                rejected += 1; outcome = "rejected_as_unknown"
            else:
                wrong += 1; outcome = "wrong_label"
        else:
            if got is None:
                outcome = "missed(no event)"
            elif got == "unknown":
                unk_rejected += 1; outcome = "correctly_rejected"
            else:
                unk_accepted += 1; outcome = "false_accept"
        rows.append({"t_start": w["t_start"], "t_end": w["t_end"], "expected": w["label"],
                     "speaker": w["speaker"], "n_events": len(w["decisions"]),
                     "observed": [(round(d["stream_time"], 3), d["label"], round(d["confidence"], 3))
                                  for d in w["decisions"]],
                     "outcome": outcome})
    return {
        "event_handling": {
            "n_words": n_words, "n_events": len(decisions),
            "n_words_with_event": n_words - len(missed), "n_missed_words": len(missed),
            "n_duplicate_events": dup_events,
            "n_false_activations": len(false_act),       # digit events with no word
            "n_spurious_unknown_events": len(spurious_unknown),  # Unknown events with no word
            "one_event_per_word": (len(missed) == 0 and dup_events == 0 and len(false_act) == 0),
            "decision_latency_s": {"mean": float(np.mean(latencies)) if latencies else None,
                                   "max": float(np.max(latencies)) if latencies else None},
        },
        "recognition": {
            "n_digit_words": n_digit_words, "correct": correct, "wrong_label": wrong,
            "rejected_as_unknown": rejected,
            "digit_accuracy_of_events": (correct / (correct + wrong + rejected)) if (correct + wrong + rejected) else None,
            "digit_accuracy_of_words": (correct / n_digit_words) if n_digit_words else None,
            "n_unknown_words": n_unknown_words, "unknown_rejected": unk_rejected,
            "unknown_false_accept": unk_accepted,
        },
        "words": rows,
        "unmatched_events": [(round(d["stream_time"], 3), d["label"], round(d["confidence"], 3)) for d in unmatched],
    }


# =============================================================================
# Scenarios
# =============================================================================
def build_scenarios(pool: ClipPool, rate: int, speech_rms: float, seed: int) -> list[dict]:
    """Return a list of scenario dicts: name, signal, events, params."""
    sc = []

    def digit_words(labels, speakers=True):
        used = set() if speakers else None
        out = []
        for lab in labels:
            p = pool.pick_digit(lab, used)
            x = prepare_word(load_clip(p, rate), rate, speech_rms)
            out.append((x, lab, str(p.relative_to(PROJECT_ROOT)), pool.speaker(p)))
        return out

    def make(name, words, gap, lead=1.0, tail=1.5, noise=None, noise_rms=0.0, noise_kind="none",
             note="", expect_merge=False, **extra):
        s = Stream(rate)
        s.add_silence(lead)
        for i, (x, lab, src, spk) in enumerate(words):
            s.add_word(x, lab, src, spk)
            s.add_silence(gap if i < len(words) - 1 else 0.0)
        s.add_silence(tail)
        sig = s.render(noise=noise, noise_rms=noise_rms, seed=seed)
        d = {"name": name, "signal": sig, "events": s.events, "duration_s": round(s.t, 3),
             "params": {"gap_s": gap, "speech_rms": speech_rms, "noise_kind": noise_kind,
                        "noise_rms": noise_rms,
                        "snr_db": (round(20 * np.log10(speech_rms / noise_rms), 1) if noise_rms > 0 else None),
                        **extra},
             "note": note, "expect_merge": expect_merge}
        sc.append(d)
        return d

    # 1. single digit
    make("single_digit", digit_words(["7"]), gap=0.0, note="one word, clean")
    # 2. same digit 3x, 0.6 s gaps (SAME clip each time: identical tuples)
    w = digit_words(["3"])[0]
    make("repeat_3x_gap0.6", [w, w, w], gap=0.6, note="identical clip repeated; predict() tuples identical")
    # 3. five different digits, several speakers
    make("five_digits_gap0.8", digit_words(["1", "4", "9", "0", "6"]), gap=0.8)
    # 4. long pauses
    make("long_pauses_3s", digit_words(["2", "8", "5", "1"]), gap=3.0)
    # 5. pure silence
    s = Stream(rate); s.add_silence(5.0)
    sc.append({"name": "silence_5s", "signal": s.render(), "events": [], "duration_s": 5.0,
               "params": {"speech_rms": 0, "noise_kind": "none", "noise_rms": 0.0, "snr_db": None},
               "note": "digital zero", "expect_merge": False})
    # 6. noise only (synthetic white + real noise clip), several levels, 5 s
    noise_a_path, noise_b_path = pool.pick_noise(), pool.pick_noise()
    noise_clip = load_clip(noise_a_path, rate)
    noise_clip_b = load_clip(noise_b_path, rate)
    for kind, nz, src in (("white_synthetic", None, "numpy standard_normal (synthetic)"),
                          ("real_noise_clipA", noise_clip, str(noise_a_path.relative_to(PROJECT_ROOT))),
                          ("real_noise_clipB", noise_clip_b, str(noise_b_path.relative_to(PROJECT_ROOT)))):
        for lvl in (0.005, 0.01, 0.02, 0.04, 0.08):
            s = Stream(rate); s.add_silence(5.0)
            sc.append({"name": f"noise_only_{kind}_rms{lvl}", "signal": s.render(nz, lvl, seed), "events": [],
                       "duration_s": 5.0, "params": {"speech_rms": 0, "noise_kind": kind, "noise_rms": lvl,
                                                     "noise_source": src, "snr_db": None},
                       "note": "no speech; any digit event is a false activation", "expect_merge": False})
    # 7. non-digit words (Speech Commands 'unknown' words)
    words = []
    for _ in range(5):
        p = pool.pick_unknown_word()
        x = prepare_word(load_clip(p, rate), rate, speech_rms)
        words.append((x, "unknown", str(p.relative_to(PROJECT_ROOT)), pool.speaker(p)))
    make("unknown_words_gap0.8", words, gap=0.8, note="expected: one Unknown event per word")
    # 8. low replay level around the gate
    for lvl in (0.03, 0.02, 0.015, 0.01):
        ws = digit_words(["4", "7", "2"])
        ws = [(x * (lvl / speech_rms), lab, src, spk) for (x, lab, src, spk) in ws]
        d = make(f"low_level_rms{lvl}", ws, gap=0.8, note="word RMS vs silence gate", level_override=lvl)
        d["params"]["speech_rms"] = lvl
    # 9. back-to-back, 0.2 s gap (< WORD_END_QUIET_SEC) -> may merge
    make("back_to_back_gap0.2", digit_words(["5", "9", "3"]), gap=0.2, expect_merge=True,
         note="gap shorter than WORD_END_QUIET_SEC=0.25: merging is possible")
    # 10. digits in background noise at stated SNR (real noise clip)
    for snr in (20, 10, 5):
        nr = speech_rms / (10 ** (snr / 20))
        make(f"digits_in_noise_snr{snr}dB", digit_words(["6", "1", "8", "3"]), gap=0.8,
             noise=noise_clip, noise_rms=nr, noise_kind="real_noise_clip")
    # 11. mixed digits and non-digit words
    ws = digit_words(["0", "9"])
    p = pool.pick_unknown_word()
    ws.insert(1, (prepare_word(load_clip(p, rate), rate, speech_rms), "unknown",
                  str(p.relative_to(PROJECT_ROOT)), pool.speaker(p)))
    make("mixed_digit_unknown_digit", ws, gap=0.8)
    return sc


CONFIGS = {
    # name: (model, word_mode, bandpass, gate)  -- how the dashboard builds each channel
    "laser_word":   ("cnn", True, LASER_BANDPASS, LASER_GATE_DEFAULT),
    "laser_sliding": ("cnn", False, LASER_BANDPASS, LASER_GATE_DEFAULT),
    "std_ensemble_sliding": ("ensemble", False, None, STD_GATE_DEFAULT),
    # proposed (not what the dashboard does today): std channel in end-of-word mode
    "std_ensemble_word": ("ensemble", True, None, STD_GATE_DEFAULT),
}


def run_scenarios(args, pool: ClipPool, model_name: str) -> dict:
    out = {"machine": MACHINE, "model": model_name, "seed": args.seed, "stream_rate": args.stream_rate,
           "chunk_ms_list": args.chunk_ms, "configs": {}, "manifest_used": pool.manifest_used,
           "speech_rms": args.speech_rms}
    scenarios = build_scenarios(pool, args.stream_rate, args.speech_rms, args.seed)
    # dump the synthetic streams' timelines (and optionally the audio) for audit
    (args.out_dir / "streams").mkdir(parents=True, exist_ok=True)
    for s in scenarios:
        with open(args.out_dir / "streams" / f"{s['name']}.json", "w") as f:
            json.dump({k: v for k, v in s.items() if k != "signal"}, f, indent=1)
        if args.save_streams:
            sf.write(str(args.out_dir / "streams" / f"{s['name']}.wav"), s["signal"], args.stream_rate)

    for cfg_name in args.configs:
        model, word_mode, bp, gate = CONFIGS[cfg_name]
        if model == "cnn":
            model = args.model_type
        gate = args.silence_rms if (args.silence_rms is not None and cfg_name.startswith("laser")) else gate
        rep = Replayer(model, word_mode, bp, gate, autosave=not args.no_autosave,
                       source_label=f"replay-{cfg_name}")
        cfg_out = {"model": model, "word_mode": word_mode, "bandpass": bp, "silence_rms": gate, "runs": {}}
        for chunk_ms in args.chunk_ms:
            runs = {}
            for s in scenarios:
                stats = rep.run(s["signal"], args.stream_rate, chunk_ms)
                m = match_events(s["events"], rep.decisions, word_mode)
                m["run"] = stats
                m["decisions"] = [{k: v for k, v in d.items() if k != "committed"} for d in rep.decisions]
                m["params"] = s["params"]
                m["note"] = s["note"]
                m["expect_merge"] = s["expect_merge"]
                m["n_inferences"] = rep.clf.n_inferences
                runs[s["name"]] = m
                print(f"[{cfg_name} chunk={chunk_ms}ms] {s['name']:<38} words={m['event_handling']['n_words']:>2} "
                      f"events={m['event_handling']['n_events']:>2} missed={m['event_handling']['n_missed_words']} "
                      f"dup={m['event_handling']['n_duplicate_events']} falseact={m['event_handling']['n_false_activations']} "
                      f"spurUnk={m['event_handling']['n_spurious_unknown_events']} "
                      f"acc={m['recognition']['digit_accuracy_of_words']}")
            cfg_out["runs"][f"chunk_{chunk_ms}ms"] = runs
        rep.close()
        out["configs"][cfg_name] = cfg_out
    return out


# =============================================================================
# Recognition accuracy on single-word streams (clean test clips + sim clips)
# =============================================================================
def load_manifest_rows(manifest: Path) -> list[dict]:
    with open(manifest, newline="") as f:
        return list(csv.DictReader(f))


def accuracy_eval(args, model_name: str, rows_clean: list[dict], rows_sim: list[dict]) -> dict:
    """Word-mode replay of one clip per stream vs. direct predict.infer() on the file."""
    res = {"model": model_name, "machine": MACHINE, "sets": {}}
    fixed_gate = args.silence_rms if args.silence_rms is not None else LASER_GATE_DEFAULT
    rep = Replayer(args.model_type, True, LASER_BANDPASS, fixed_gate, autosave=False, source_label="replay-acc")
    chunk_ms = args.chunk_ms[0]
    rate = args.stream_rate
    for set_name, rows in (("clean_test", rows_clean), ("sim_test", rows_sim)):
        if not rows:
            res["sets"][set_name] = {"n": 0, "note": "no rows available"}
            continue
        per = []
        for r in rows:
            p = PROJECT_ROOT / r["file"]
            xw = prepare_word(load_clip(p, rate), rate, args.speech_rms)
            s = Stream(rate)
            gate = fixed_gate
            rep.clf.silence_rms = gate
            s.add_silence(1.0)
            s.add_word(xw, r["label"], r["file"], r.get("speaker", ""))
            s.add_silence(1.2)
            rep.run(s.render(), rate, chunk_ms)
            m = match_events(s.events, rep.decisions, True)
            ev = m["words"][0]
            got = ev["observed"][0][1] if ev["observed"] else None
            # direct path (no streaming/gate/word segmentation): the file itself
            from preprocess import load_audio
            direct, _ = predict_mod.infer(load_audio(p), model_type=args.model_type)
            per.append({"file": r["file"], "label": r["label"], "speaker": r.get("speaker", ""),
                        "condition": r.get("condition", "clean"), "snr_db": r.get("snr_db", ""), "gate": round(gate, 4),
                        "replay_pred": got, "replay_n_events": ev["n_events"],
                        "direct_pred": direct["prediction"], "direct_conf": round(direct["confidence"], 4)})
        n = len(per)
        digit = [q for q in per if q["label"] in DIGITS]
        unk = [q for q in per if q["label"] not in DIGITS]
        def acc(qs, key):
            return (sum(1 for q in qs if q[key] == q["label"]) / len(qs)) if qs else None
        by_cond = {}
        for c in sorted({q["condition"] for q in digit}):
            qs = [q for q in digit if q["condition"] == c]
            by_cond[c] = {"n": len(qs), "replay_acc": acc(qs, "replay_pred"), "direct_acc": acc(qs, "direct_pred"),
                          "replay_rejected_as_unknown": sum(1 for q in qs if q["replay_pred"] == "unknown"),
                          "replay_missed_no_event": sum(1 for q in qs if q["replay_pred"] is None),
                          "replay_multi_event": sum(1 for q in qs if q["replay_n_events"] > 1),
                          "gate_mean": float(np.mean([q["gate"] for q in qs]))}
        res["sets"][set_name] = {
            "n": n, "n_digit": len(digit), "n_unknown": len(unk),
            "replay_digit_accuracy": acc(digit, "replay_pred"), "direct_digit_accuracy": acc(digit, "direct_pred"),
            "replay_one_event_per_clip": sum(1 for q in per if q["replay_n_events"] == 1) / n if n else None,
            "replay_unknown_rejection": (sum(1 for q in unk if q["replay_pred"] == "unknown") / len(unk)) if unk else None,
            "direct_unknown_rejection": (sum(1 for q in unk if q["direct_pred"] == "unknown") / len(unk)) if unk else None,
            "by_condition": by_cond, "per_clip": per,
        }
        print(f"[accuracy {model_name}] {set_name}: n={n} replay_acc={res['sets'][set_name]['replay_digit_accuracy']} "
              f"direct_acc={res['sets'][set_name]['direct_digit_accuracy']}")
    rep.close()
    return res


def select_accuracy_rows(manifest: Path, pool: ClipPool, n_clean: int, n_sim: int, seed: int):
    rng = random.Random(seed)
    rows = load_manifest_rows(manifest) if manifest.exists() else []
    clean = [r for r in rows if r["split"] == "test" and r["kind"] == "clean"
             and r.get("synthetic_noise", "0") != "1"]
    sim = [r for r in rows if r["split"] == "test" and r["kind"] == "sim"]
    note = "manifest"
    if not clean:
        note = "no manifest: clean clips sampled from ALL speakers (not a held-out set)"
        clean = [{"file": str(p.relative_to(PROJECT_ROOT)), "label": p.stem.split("_")[0],
                  "speaker": pool.speaker(p), "condition": "clean"} for p in pool.digits]
    rng.shuffle(clean); rng.shuffle(sim)
    return clean[:n_clean], sim[:n_sim], note


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default=None, help="checkpoint for --model-type (default: deployed models/)")
    ap.add_argument("--model-type", default="cnn", choices=["cnn", "lstm"])
    ap.add_argument("--lstm-path", default=None, help="LSTM checkpoint for the ensemble config (default deployed)")
    ap.add_argument("--tag", default=None, help="name used in outputs (default: derived from model path)")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results/sprint8/replay")
    ap.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "results/sprint8/data/manifest.csv")
    ap.add_argument("--stream-rate", type=int, default=DASH_RATE)
    ap.add_argument("--chunk-ms", type=float, nargs="+", default=[256.0, 50.0],
                    help="push sizes; 256 ms = dashboard hardware read (12288 frames @48k), 50 ms = GUI-tick-like")
    ap.add_argument("--speech-rms", type=float, default=0.05, help="word RMS in the replayed stream")
    ap.add_argument("--silence-rms", type=float, default=None, help="laser gate (default: dashboard's 0.02)")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--no-autosave", action="store_true", help="disable log_prediction autosave (dashboard has it on)")
    ap.add_argument("--save-streams", action="store_true", help="also write the synthetic streams as WAV")
    ap.add_argument("--scenarios", action="store_true", help="run the deterministic scenarios (default if nothing else)")
    ap.add_argument("--accuracy", action="store_true", help="run the single-word recognition accuracy set")
    ap.add_argument("--out-name", default=None, help="basename for the --accuracy JSON (default accuracy_<tag>)")
    ap.add_argument("--n-clean", type=int, default=400)
    ap.add_argument("--n-sim", type=int, default=600)
    ap.add_argument("--threads", type=int, default=1,
                    help="torch intra-op threads (1: stable timings while other agents use the CPU)")
    args = ap.parse_args()
    if not args.accuracy:
        args.scenarios = True
    args.out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.set_num_threads(args.threads)

    # Point predict.py at explicit checkpoints. LMML_OUTPUT_DIR moves config.MODELS_DIR
    # away from models/, so the deployed checkpoints must be named explicitly too.
    deployed = PROJECT_ROOT / "models"
    cnn_path = Path(args.model_path) if args.model_path else deployed / "best_model_cnn.pt"
    lstm_path = Path(args.lstm_path) if args.lstm_path else deployed / "best_model.pt"
    predict_mod.set_checkpoint_path(args.model_type, cnn_path)
    predict_mod.set_checkpoint_path("lstm", lstm_path)
    if args.model_type == "lstm":
        predict_mod.set_checkpoint_path("cnn", deployed / "best_model_cnn.pt")
    tag = args.tag or (cnn_path.stem if args.model_path else "deployed")
    ckpt = torch.load(cnn_path, map_location="cpu")
    model_info = {"tag": tag, "path": str(cnn_path), "sha256": _sha256(cnn_path),
                  "threshold": ckpt.get("threshold"), "temperature": ckpt.get("temperature"),
                  "labels": ckpt.get("labels"), "lstm_path": str(lstm_path)}
    print(f"model {tag}: {cnn_path} thr={model_info['threshold']} T={model_info['temperature']}")

    pool = ClipPool(PROJECT_ROOT / "data/raw", args.manifest, args.seed)
    print(f"clip pool: {len(pool.digits)} digit clips, manifest_used={pool.manifest_used}, "
          f"{len(pool.unknown_words)} unknown words, {len(pool.noise_clips)} noise clips")

    result = {"model_info": model_info, "machine": MACHINE, "python": platform.python_version(),
              "torch": torch.__version__, "sample_rate_model": SAMPLE_RATE,
              "constants": {"BUFFER_SECONDS": sb.BUFFER_SECONDS, "PREDICT_WINDOW_SEC": sb.PREDICT_WINDOW_SEC,
                            "PREDICT_EVERY_SEC": sb.PREDICT_EVERY_SEC, "STABLE_CONF": sb.STABLE_CONF,
                            "STABLE_HITS": sb.STABLE_HITS,
                            "WORD_FRAME_SEC": _LiveClassifier.WORD_FRAME_SEC,
                            "WORD_END_QUIET_SEC": _LiveClassifier.WORD_END_QUIET_SEC,
                            "WORD_MIN_SEC": _LiveClassifier.WORD_MIN_SEC,
                            "WORD_MAX_SEC": _LiveClassifier.WORD_MAX_SEC,
                            "WORD_PAD_SEC": _LiveClassifier.WORD_PAD_SEC},
              "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}
    if args.scenarios:
        result["scenarios"] = run_scenarios(args, pool, tag)
        with open(args.out_dir / f"scenarios_{tag}.json", "w") as f:
            json.dump(result, f, indent=1, default=str)
        print(f"wrote {args.out_dir / f'scenarios_{tag}.json'}")
    if args.accuracy:
        clean, sim, note = select_accuracy_rows(args.manifest, pool, args.n_clean, args.n_sim, args.seed)
        acc = accuracy_eval(args, tag, clean, sim)
        acc["selection_note"] = note
        result["accuracy"] = acc
        acc["manifest"] = str(args.manifest)
        name = args.out_name or f"accuracy_{tag}"
        with open(args.out_dir / f"{name}.json", "w") as f:
            json.dump(result, f, indent=1, default=str)
        print(f"wrote {args.out_dir / f'{name}.json'}")


if __name__ == "__main__":
    main()
