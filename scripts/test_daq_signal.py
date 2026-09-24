"""
Headless check that the PCM1808/DAQ hardware is actually reaching software.

No GUI, no model -- just opens signal_backend._DAQSource and prints the
live RMS/peak level a few times a second. Run this before touching the full
dashboard when bringing up new hardware or a new Jetson.

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    print(f"Listening for {args.seconds:.0f}s, reading every {args.window:.1f}s ...")
    print()

    daq.start()
    try:
        end = time.monotonic() + args.seconds
        saw_signal = False
        while time.monotonic() < end:
            time.sleep(args.window)
            buf = daq.latest(args.window)
            if buf.size == 0:
                print("  (no samples yet)")
                continue
            rms = float(np.sqrt(np.mean(buf ** 2)))
            peak = float(np.max(np.abs(buf)))
            bar = "#" * min(50, int(rms * 500))
            print(f"  rms={rms:.5f}  peak={peak:.5f}  {bar}")
            if rms > 0.001:
                saw_signal = True
    finally:
        daq.stop()

    print()
    if saw_signal:
        print("OK: non-zero signal reached software from the hardware.")
        return 0
    print("WARN: capture ran but level stayed near zero the whole time.")
    print("Check the physical signal, wiring, and gain -- software path is open.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
