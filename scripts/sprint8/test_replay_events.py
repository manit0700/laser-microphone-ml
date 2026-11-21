"""
Sprint 8 regression checks for the live classifier's event handling.

    python -m pytest scripts/sprint8/test_replay_events.py -q

Uses the deployed CNN (models/best_model_cnn.pt) and a few real clips from
data/raw through the same replay path as scripts/sprint8/replay_harness.py.
No microphone, ADC or GUI is involved. Autosave is off, so nothing is written.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "sprint8"))
os.environ.setdefault("LMML_OUTPUT_DIR", str(PROJECT_ROOT / "results" / "sprint8" / "replay"))

import replay_harness as rh  # noqa: E402
import predict as pm  # noqa: E402
import signal_backend as sb  # noqa: E402
from signal_backend import _LiveClassifier  # noqa: E402

RATE = rh.DASH_RATE
GATE = rh.LASER_GATE_DEFAULT
BP = rh.LASER_BANDPASS
CNN = PROJECT_ROOT / "models" / "best_model_cnn.pt"
LSTM = PROJECT_ROOT / "models" / "best_model.pt"

pytestmark = pytest.mark.skipif(not CNN.exists() or not (PROJECT_ROOT / "data" / "raw").exists(),
                                reason="needs the deployed CNN and data/raw")


@pytest.fixture(scope="module", autouse=True)
def _checkpoints():
    pm.set_checkpoint_path("cnn", CNN)
    pm.set_checkpoint_path("lstm", LSTM)
    yield


def _clip(name: str, rms: float = 0.05) -> np.ndarray:
    p = PROJECT_ROOT / "data" / "raw" / name
    if not p.exists():
        pytest.skip(f"missing clip {name}")
    return rh.prepare_word(rh.load_clip(p, RATE), RATE, rms)


def _stream(words, gap=0.6, lead=1.0, tail=1.5):
    s = rh.Stream(RATE)
    s.add_silence(lead)
    for i, (x, lab) in enumerate(words):
        s.add_word(x, lab, "", "")
        s.add_silence(gap if i < len(words) - 1 else 0.0)
    s.add_silence(tail)
    return s


def _run(sig, chunk_ms, word_mode=True, model="cnn", bandpass=BP, gate=GATE):
    rep = rh.Replayer(model, word_mode, bandpass, gate, autosave=False)
    try:
        rep.run(sig, RATE, chunk_ms)
        return list(rep.decisions), list(rep.returned)
    finally:
        rep.close()


# ---------------------------------------------------------------------------
# core guarantees: one event per spoken word, none without a word
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("chunk_ms", [256, 50])
def test_single_digit_one_event(chunk_ms):
    s = _stream([(_clip("7_jackson_0.wav"), "7")])
    dec, _ = _run(s.render(), chunk_ms)
    assert len(dec) == 1
    assert dec[0]["kind"] == "word"
    # decision falls after the word ended and within the end-of-word timer + one push
    assert s.events[0]["t_end"] - 0.2 <= dec[0]["stream_time"] <= s.events[0]["t_end"] + 0.25 + 0.15 + chunk_ms / 1000 + 0.1


@pytest.mark.parametrize("chunk_ms", [256, 50])
def test_repeated_identical_word_gives_one_event_each(chunk_ms):
    """Three identical clips -> three decisions, even though predict() returns
    the very same (label, confidence) tuple for each of them."""
    x = _clip("3_nicolas_0.wav")
    s = _stream([(x, "3")] * 3, gap=0.6)
    dec, _ = _run(s.render(), chunk_ms)
    assert len(dec) == 3, [(d["stream_time"], d["label"]) for d in dec]
    assert len({d["label"] for d in dec}) == 1          # same decision each time
    # one decision per word, in order
    for d, e in zip(dec, s.events):
        assert d["word_start"] <= e["t_end"] and d["word_end"] >= e["t_start"] - 0.1


@pytest.mark.parametrize("chunk_ms", [256, 50])
def test_five_words_five_events(chunk_ms):
    names = ["1_theo_0.wav", "4_lucas_0.wav", "9_jackson_1.wav", "0_nicolas_1.wav", "6_yweweler_2.wav"]
    s = _stream([(_clip(n), n[0]) for n in names], gap=0.8)
    dec, _ = _run(s.render(), chunk_ms)
    assert len(dec) == 5


def test_silence_and_sub_gate_noise_give_no_event():
    s = rh.Stream(RATE)
    s.add_silence(5.0)
    assert _run(s.render(), 256)[0] == []
    # white noise whose band-passed RMS stays under the gate
    assert _run(s.render(noise=None, noise_rms=0.01, seed=1), 256)[0] == []


def test_short_word_detected_at_every_push_phase():
    """Regression for the end-of-word gate dead zone: with 256 ms pushes the gate
    used to inspect only the last 0.15 s of each chunk, so a 0.2 s digit placed in
    the first ~0.1 s of a chunk was missed. Now every new frame is gated."""
    x = _clip("1_yweweler_1.wav")                     # ~0.22 s after trimming
    assert x.size / RATE < 0.3
    for k in range(16):
        s = rh.Stream(RATE)
        s.add_silence(1.0 + k * 0.016)
        s.add_word(x, "1", "", "")
        s.add_silence(1.5)
        dec, _ = _run(s.render(), 256)
        assert len(dec) == 1, f"onset offset {k * 16} ms: {len(dec)} events"


def test_word_start_is_tracked_from_frames_not_ticks():
    x = _clip("8_jackson_3.wav")
    s = _stream([(x, "8")], lead=1.1)
    dec, _ = _run(s.render(), 256)
    assert len(dec) == 1
    # true onset 1.1 s; word_start must be within one frame (0.15 s) + one hop, not one push (0.256 s)
    assert abs(dec[0]["word_start"] - 1.1) <= 0.2, dec[0]["word_start"]


def test_model_window_is_independent_of_push_size():
    """The samples handed to the model are anchored to the word, so the push
    size (GUI tick) does not change the classified window or the label."""
    names = ["2_theo_3.wav", "5_lucas_2.wav", "7_nicolas_4.wav"]
    s = _stream([(_clip(n), n[0]) for n in names], gap=1.0)
    d256, _ = _run(s.render(), 256)
    d50, _ = _run(s.render(), 50)
    assert len(d256) == len(d50) == 3
    for a, b in zip(d256, d50):
        assert a["label"] == b["label"]
        assert abs(a["span_sec"] - b["span_sec"]) <= 0.06
        assert abs(a["word_start"] - b["word_start"]) <= 0.06


def test_back_to_back_short_gap_never_duplicates():
    """0.2 s gaps (< WORD_END_QUIET_SEC) may merge words, but must never
    produce MORE events than words."""
    names = ["5_theo_1.wav", "9_lucas_1.wav", "3_jackson_2.wav"]
    s = _stream([(_clip(n), n[0]) for n in names], gap=0.2)
    dec, _ = _run(s.render(), 256)
    assert 1 <= len(dec) <= 3


# ---------------------------------------------------------------------------
# instrumentation contract
# ---------------------------------------------------------------------------
def test_decision_counter_and_callback_agree():
    seen = []
    clf = _LiveClassifier(model="cnn", silence_rms=GATE, bandpass=BP, word_mode=True,
                          autosave=False, on_decision=seen.append)
    x = _clip("4_jackson_0.wav")
    s = _stream([(x, "4"), (x, "4")], gap=0.7)
    sig = s.render()
    n = int(RATE * 0.256)
    for i in range(0, sig.size, n):
        clf.push(sig[i:i + n], RATE)
        clf.predict()
    assert clf.n_decisions == len(seen) == 2
    assert clf.n_inferences == 2
    assert clf.last_decision is seen[-1] or clf.last_decision == seen[-1]
    assert {"kind", "label", "confidence", "status", "stream_time", "word_start", "word_end"} <= set(seen[0])


def test_callback_exceptions_never_break_predict():
    def boom(info):
        raise RuntimeError("listener failure")
    clf = _LiveClassifier(model="cnn", silence_rms=GATE, bandpass=BP, word_mode=True,
                          autosave=False, on_decision=boom)
    s = _stream([(_clip("6_theo_2.wav"), "6")])
    sig = s.render()
    n = int(RATE * 0.256)
    for i in range(0, sig.size, n):
        clf.push(sig[i:i + n], RATE)
        clf.predict()
    assert clf.n_decisions == 1


def test_reset_clears_gate_state():
    clf = _LiveClassifier(model="cnn", silence_rms=GATE, bandpass=BP, word_mode=True, autosave=False)
    s = _stream([(_clip("2_jackson_0.wav"), "2")])
    sig = s.render()
    n = int(RATE * 0.256)
    for i in range(0, sig.size, n):
        clf.push(sig[i:i + n], RATE)
        clf.predict()
    assert clf.n_decisions == 1
    clf.reset()
    assert clf._n_pushed == 0 and clf._n_gated == 0 and not clf._in_word and clf._committed is None
    for i in range(0, sig.size, n):
        clf.push(sig[i:i + n], RATE)
        clf.predict()
    assert clf.n_decisions == 2


def test_sliding_mode_uses_injected_clock():
    """Sliding mode's throttle must follow the injected clock (replay), and the
    default must still be time.monotonic (dashboard)."""
    t = {"now": 0.0}
    clf = _LiveClassifier(model="cnn", silence_rms=GATE, bandpass=BP, word_mode=False,
                          autosave=False, clock=lambda: t["now"])
    s = _stream([(_clip("7_jackson_0.wav"), "7")], lead=0.5, tail=0.5)
    sig = s.render()
    n = int(RATE * 0.05)
    for i in range(0, sig.size, n):
        clf.push(sig[i:i + n], RATE)
        t["now"] = clf._stream_time()
        clf.predict()
    assert clf.n_inferences > 0
    assert clf.n_decisions == 1 and clf.last_decision["kind"] == "commit"
    import time
    assert _LiveClassifier(model="cnn", autosave=False)._clock is time.monotonic


# ---------------------------------------------------------------------------
# predict.py checkpoint hook
# ---------------------------------------------------------------------------
def test_checkpoint_override_and_env(monkeypatch, tmp_path):
    pm.set_checkpoint_path("cnn", CNN)
    assert pm.checkpoint_path("cnn") == CNN
    assert pm._load("cnn")["labels"][-1] == "unknown"
    # env fallback applies only when no explicit override is set
    pm.set_checkpoint_path("cnn", None)
    monkeypatch.setenv("LMML_CKPT_CNN", str(tmp_path / "nope.pt"))
    assert pm.checkpoint_path("cnn") == tmp_path / "nope.pt"
    with pytest.raises(FileNotFoundError):
        pm._load("cnn")
    monkeypatch.delenv("LMML_CKPT_CNN")
    # without override/env: config.model_checkpoint (unchanged behaviour)
    from config import model_checkpoint
    assert pm.checkpoint_path("cnn") == Path(model_checkpoint("cnn"))
    pm.set_checkpoint_path("cnn", CNN)          # restore for the other tests
    assert "cnn" in pm._cache or pm._load("cnn")


def test_continuous_activity_above_gate_gives_one_event():
    """Regression for the WORD_MAX_SEC cut-off: noise that never drops below the
    gate must not produce a new decision every 1.1 s; one per burst, re-armed by
    the first quiet frame."""
    s = rh.Stream(RATE)
    s.add_silence(5.0)
    loud = s.render(noise=None, noise_rms=0.08, seed=3)     # band-passed RMS ~0.03 > gate 0.02
    dec, _ = _run(loud, 256)
    assert len(dec) == 1, [(d["stream_time"], d["label"]) for d in dec]
    # and two bursts separated by real silence are two decisions
    s2 = rh.Stream(RATE)
    s2.add_silence(0.5)
    burst = rh.Stream(RATE); burst.add_silence(2.0)
    b = burst.render(noise=None, noise_rms=0.08, seed=4)
    sig = np.concatenate([np.zeros(int(0.5 * RATE), np.float32), b, np.zeros(int(1.0 * RATE), np.float32), b,
                          np.zeros(int(1.0 * RATE), np.float32)])
    dec2, _ = _run(sig, 256)
    assert len(dec2) == 2, [(d["stream_time"], d["label"]) for d in dec2]
