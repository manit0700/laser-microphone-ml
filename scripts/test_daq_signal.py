"""
Headless check that the PCM1808/DAQ hardware is actually reaching software.

No GUI, no model -- just opens signal_backend._DAQSource and prints the live RMS
level of EACH input channel (left = standard mic, right = laser receiver) a few
times a second. Run this before touching the full dashboard when bringing up new
hardware, and use it to check the laser channel really responds to speech.

While it runs, SPEAK (or tap / disturb the surface the laser is pointed at). A
channel only counts as "responding" if its level clearly jumps above its own idle
level -- steady background noise alone is not a response.

Usage (on the Jetson):
    python scripts/test_daq_signal.py
    python scripts/test_daq_signal.py --seconds 15
    LMML_DAQ_DEVICE=2 python scripts/test_daq_signal.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from signal_backend import _DAQSource  # noqa: E402

# A channel "responded" if its loudest reading is at least this many times its
# typical (median) reading AND the jump is big enough to matter.
JUMP_RATIO = 2.0
MIN_JUMP = 0.002


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def _bar(level: float) -> str:
    return "#" * min(30, int(level * 300))


def _responded(levels: list[float]) -> bool:
    if len(levels) < 3:
        return False
    median, peak = float(np.median(levels)), max(levels)
    return peak > JUMP_RATIO * median and (peak - median) > MIN_JUMP


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="How long to listen (default: 10s)")
    parser.add_argument("--window", type=float, default=0.5,
                        help="Seconds of audio per level reading (default: 0.5s)")
    args = parser.parse_args()

    daq = _DAQSource(buffer_seconds=2.0)

    if not daq.available:
        print(f"FAIL: DAQ source not available -> {daq.error}")
        print("Check: pyaudio installed, portaudio19-dev installed, "
              "PCM1808 plugged in, LMML_DAQ_DEVICE set correctly if auto-detect is wrong.")
        return 1

    print(f"Device: [{daq.device}] {daq.device_name}")
    print(f"Rate:   {daq.rate} Hz")
    print(f"Listening for {args.seconds:.0f}s, reading every {args.window:.1f}s "
          f"-- speak / disturb the sensor while it runs ...")
    print()

    std_levels: list[float] = []
    laser_levels: list[float] = []
    daq.start()
    try:
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            time.sleep(args.window)
            std, laser = daq.latest_channels(args.window)
            if std.size == 0:
                print("  (no samples yet)")
                continue
            s, l = _rms(std), _rms(laser)
            std_levels.append(s)
            laser_levels.append(l)
            print(f"  std   rms={s:.5f} {_bar(s):<30} | laser rms={l:.5f} {_bar(l)}")
    finally:
        daq.stop()

    print()
    for name, levels in (("standard mic (left)", std_levels), ("laser receiver (right)", laser_levels)):
        if not levels:
            print(f"{name}: no samples captured")
            continue
        verdict = "RESPONDED" if _responded(levels) else "flat (no clear response)"
        print(f"{name}: {verdict}  | idle~{np.median(levels):.5f}  peak={max(levels):.5f}")
    print()
    if _responded(laser_levels):
        print("OK: the laser channel's level jumped -- it is picking something up.")
    else:
        print("WARN: the laser channel stayed flat. If the standard mic responded, the laser "
              "path (optics / alignment / gain / wiring) isn't picking up the speech yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
