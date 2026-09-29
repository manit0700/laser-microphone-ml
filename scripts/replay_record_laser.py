"""
replay_record_laser.py
======================
Build a REAL laser-microphone training set automatically -- no one has to speak.

HOW IT WORKS
------------
Put a speaker (phone / Bluetooth / USB speaker plugged into the Jetson) right
next to the surface the laser is pointed at. This script then, over and over:

    1. picks a labeled spoken-digit clip already in data/raw/
       (FSDD, AudioMNIST, Speech Commands, your team's recordings),
    2. plays it out of the speaker,
    3. records what the PCM1808 hears on BOTH channels while it plays,
    4. checks the laser channel actually picked the sound up (SNR check),
    5. saves the laser recording with the same digit label:

           data/raw/<digit>_<speaker>_laser_<n>.wav     e.g. 7_laserplay_laser_12.wav

Because the script knows which digit it just played, every clip is labeled
for free. A few hundred clips per digit = thousands of real laser examples
from many different voices, collected unattended (roughly 2 s per clip).

Clips where the laser channel did NOT rise clearly above its own background
noise are skipped (not saved), so a bad alignment doesn't fill the training
set with hum labeled as digits.

REQUIRES
--------
  - PCM1808 connected (same as record_daq_wav.py), laser aligned
  - a speaker the Jetson can play through (default output device, or
    LMML_PLAY_DEVICE=<index>; list devices with --list-devices)
  - base clips in data/raw/ (scripts/download_fsdd.py and/or AudioMNIST)

USAGE (project root)
--------------------
  Check the setup first -- plays 10 clips, prints how well the laser hears
  them, saves nothing:
      python3 scripts/replay_record_laser.py --test

  Collect 200 laser clips per digit (~70 min) plus 300 'unknown' words:
      python3 scripts/replay_record_laser.py --per-digit 200 --unknown 300

  Then fine-tune on them:
      python3 scripts/finetune.py --speakers laserplay

  Useful options:
      --volume 0.9          playback loudness (0-1)
      --min-snr 2.0         how far above background the laser must rise to keep a clip
      --sources audiomnist  only replay clips whose path contains this text
      --list-devices        show audio devices and their indexes
      LMML_DAQ_DEVICE=4     force the PCM1808 input (same as the other scripts)
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import soundfile as sf  # noqa: E402
from scipy.signal import butter, resample_poly, sosfiltfilt  # noqa: E402

from config import DIGIT_LABELS, RAW_DATA_DIR, SAMPLE_RATE, UNKNOWN_LABEL  # noqa: E402
from preprocess import load_waveform_from_array, save_capture, save_wav  # noqa: E402,F401
from signal_backend import _DAQSource  # noqa: E402

PRE_SEC = 0.4    # background captured before playback (noise reference)
POST_SEC = 0.4   # tail captured after playback
SKIP_TAGS = ("_laser_", "laserplay", "lasersynth")  # never replay laser/derived clips


def _speech_band(x: np.ndarray, rate: int) -> np.ndarray:
    """300-3400 Hz band: where speech energy is, and well away from 60 Hz hum."""
    if x.size < 64:
        return x
    sos = butter(4, [300 / (rate / 2), min(3400, rate / 2 * 0.95) / (rate / 2)],
                 btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def _snr(signal: np.ndarray, rate: int, pre_n: int) -> float:
    """Speech-band loudness during playback divided by the pre-playback background."""
    band = _speech_band(signal.astype(np.float64), rate)
    noise = _rms(band[:pre_n])
    active = _rms(band[pre_n:])
    return active / noise if noise > 0 else float("inf")


def _list_devices():
    import pyaudio
    pa = pyaudio.PyAudio()
    try:
        default_out = pa.get_default_output_device_info().get("index")
    except Exception:  # noqa: BLE001
        default_out = None
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        tags = []
        if d.get("maxInputChannels", 0) > 0:
            tags.append(f"in:{d['maxInputChannels']}")
        if d.get("maxOutputChannels", 0) > 0:
            tags.append(f"out:{d['maxOutputChannels']}")
        star = "  <- default output" if i == default_out else ""
        print(f"[{i}] {d['name']}  ({', '.join(tags)}, {int(d.get('defaultSampleRate', 0))} Hz){star}")
    pa.terminate()


class _Player:
    """Plays a mono float clip on an output device (PyAudio, blocking)."""

    def __init__(self, device_index=None):
        import pyaudio
        self._pyaudio = pyaudio
        self.pa = pyaudio.PyAudio()
        if device_index is None:
            info = self.pa.get_default_output_device_info()
        else:
            info = self.pa.get_device_info_by_index(device_index)
        self.index = int(info["index"])
        self.name = info["name"]
        self.rate = int(info.get("defaultSampleRate") or 48000)
        self.channels = 2 if info.get("maxOutputChannels", 1) >= 2 else 1
        self.stream = self.pa.open(format=pyaudio.paFloat32, channels=self.channels,
                                   rate=self.rate, output=True, output_device_index=self.index)

    def play(self, samples: np.ndarray, rate: int, volume: float):
        x = samples.astype(np.float32)
        if rate != self.rate:
            g = np.gcd(int(rate), int(self.rate))
            x = resample_poly(x, self.rate // g, int(rate) // g).astype(np.float32)
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak > 0:
            x = x / peak * volume
        if self.channels == 2:
            x = np.repeat(x[:, None], 2, axis=1).reshape(-1)
        self.stream.write(x.astype(np.float32).tobytes())

    def close(self):
        try:
            self.stream.stop_stream()
            self.stream.close()
        finally:
            self.pa.terminate()


def _pick_clips(raw_dir: Path, per_digit: int, n_unknown: int, source_filter, rng):
    digit_pool = {d: [] for d in DIGIT_LABELS}
    unknown_pool = []
    for p in raw_dir.rglob("*.wav"):
        s = str(p)
        if any(t in p.name for t in SKIP_TAGS):
            continue
        if source_filter and source_filter not in s:
            continue
        if p.parent.name == UNKNOWN_LABEL:
            unknown_pool.append(p)
            continue
        label = p.stem.split("_")[0]
        if label in digit_pool:
            digit_pool[label].append(p)
    jobs = []
    for d, items in digit_pool.items():
        rng.shuffle(items)
        jobs += [(d, p) for p in items[:per_digit]]
    rng.shuffle(unknown_pool)
    jobs += [(UNKNOWN_LABEL, p) for p in unknown_pool[:n_unknown]]
    rng.shuffle(jobs)  # interleave digits so slow drift in alignment affects all labels equally
    counts = {d: len(v) for d, v in digit_pool.items()}
    return jobs, counts, len(unknown_pool)


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay labeled digit clips through a speaker and "
                                             "record them through the laser to build laser training data.")
    ap.add_argument("--per-digit", type=int, default=100, help="clips to replay per digit (default 100)")
    ap.add_argument("--unknown", type=int, default=0, help="non-digit clips to replay for the 'unknown' class")
    ap.add_argument("--speaker", default="laserplay", help="speaker tag in saved filenames (default laserplay)")
    ap.add_argument("--volume", type=float, default=0.9, help="playback level 0-1 (default 0.9)")
    ap.add_argument("--min-snr", type=float, default=2.0,
                    help="keep a clip only if the laser's speech-band level during playback is at least "
                         "this many times its background level (default 2.0)")
    ap.add_argument("--save-std", action="store_true",
                    help="also save the standard-mic channel (as <digit>_<speaker>ref_std_<n>.wav)")
    ap.add_argument("--sources", default=None, help="only replay clips whose path contains this text")
    ap.add_argument("--test", action="store_true", help="play 10 clips, report laser pickup, save nothing")
    ap.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.list_devices:
        _list_devices()
        return 0

    raw_dir = Path(RAW_DATA_DIR)
    rng = random.Random(args.seed if not args.test else int(time.time()))
    per_digit = 1 if args.test else args.per_digit
    jobs, pool_counts, n_unk_pool = _pick_clips(raw_dir, per_digit, 0 if args.test else args.unknown,
                                                args.sources, rng)
    if args.test:
        jobs = jobs[:10]
    if not jobs:
        print(f"No source clips found in {raw_dir}. Run scripts/download_fsdd.py (or add AudioMNIST) first.")
        return 1

    daq = _DAQSource(buffer_seconds=6.0)
    if not daq.available:
        print(f"ERROR: DAQ not available: {daq.error}")
        print("Close signal_dashboard2.py / other recorders, or set LMML_DAQ_DEVICE=<index>.")
        return 1

    play_dev = os.environ.get("LMML_PLAY_DEVICE")
    try:
        player = _Player(int(play_dev) if play_dev is not None else None)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: could not open a playback device: {type(e).__name__}: {e}")
        print("Connect a speaker, then run --list-devices and set LMML_PLAY_DEVICE=<index>.")
        return 1

    print(f"Input : [{daq.device}] {daq.device_name} @ {daq.rate} Hz")
    print(f"Output: [{player.index}] {player.name} @ {player.rate} Hz, volume {args.volume}")
    print(f"Source clips available per digit: {pool_counts} | unknown: {n_unk_pool}")
    mode = "TEST (nothing saved)" if args.test else f"saving to {raw_dir} as *_{args.speaker}_laser_*.wav"
    print(f"Replaying {len(jobs)} clips -- {mode}. Keep the room quiet. Ctrl+C stops safely.\n")

    unk_dir = raw_dir / UNKNOWN_LABEL
    next_idx = {}

    def idx_for(label: str, tag: str) -> int:
        key = (label, tag)
        if key not in next_idx:
            folder = unk_dir if label == UNKNOWN_LABEL else raw_dir
            prefix = "unk" if label == UNKNOWN_LABEL else label
            next_idx[key] = len(list(folder.glob(f"{prefix}_{tag}_*.wav")))
        i = next_idx[key]
        next_idx[key] = i + 1
        return i

    kept = skipped = 0
    laser_snrs, std_snrs = [], []
    daq.start()
    time.sleep(1.0)  # let the input buffer fill
    try:
        for n, (label, src) in enumerate(jobs, 1):
            audio, sr = sf.read(str(src), dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            dur = len(audio) / sr

            time.sleep(PRE_SEC)
            player.play(audio, sr, args.volume)
            time.sleep(POST_SEC)
            std, laser = daq.latest_channels(PRE_SEC + dur + POST_SEC)
            pre_n = int(PRE_SEC * daq.rate)
            if laser.size < pre_n + int(0.3 * daq.rate):
                print(f"[{n}/{len(jobs)}] '{label}' too few samples captured, skipped")
                skipped += 1
                continue

            s_laser = _snr(laser, daq.rate, pre_n)
            s_std = _snr(std, daq.rate, pre_n)
            laser_snrs.append(s_laser)
            std_snrs.append(s_std)
            ok = s_laser >= args.min_snr
            verdict = "KEEP" if ok else "skip (laser didn't hear it)"
            line = (f"[{n}/{len(jobs)}] '{label}' {src.name:<28} laser SNR {s_laser:5.2f} | "
                    f"std SNR {s_std:5.2f} -> {verdict}")

            if ok and not args.test:
                tag = f"{args.speaker}_laser"
                if label == UNKNOWN_LABEL:
                    out = unk_dir / f"unk_{tag}_{idx_for(label, tag)}.wav"
                else:
                    out = raw_dir / f"{label}_{tag}_{idx_for(label, tag)}.wav"
                save_capture(laser[pre_n // 2:], daq.rate, out)
                line += f"  {out.name}"
                if args.save_std:
                    rtag = f"{args.speaker}ref_std"
                    folder = unk_dir if label == UNKNOWN_LABEL else raw_dir
                    prefix = "unk" if label == UNKNOWN_LABEL else label
                    save_capture(std[pre_n // 2:], daq.rate,
                                 folder / f"{prefix}_{rtag}_{idx_for(label, rtag)}.wav")
            print(line)
            kept += int(ok)
            skipped += int(not ok)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        daq.stop()
        player.close()

    print("\n" + "=" * 70)
    if laser_snrs:
        print(f"Laser SNR   median {np.median(laser_snrs):.2f}  (>= {args.min_snr} kept)")
        print(f"Std mic SNR median {np.median(std_snrs):.2f}  (should be well above 2 -- proves the speaker is audible)")
    print(f"Kept {kept}, skipped {skipped}.")
    if laser_snrs and np.median(std_snrs) < 2:
        print("HINT: the standard mic barely heard the playback either -> turn the speaker up / move it closer.")
    elif laser_snrs and np.median(laser_snrs) < args.min_snr:
        print("HINT: the speaker is audible but the laser isn't picking it up -> fix alignment "
              "(spot half on the photodiode), use a thinner surface, reduce hum. See the 1 kHz tone test.")
    elif not args.test and kept:
        print(f"Next:  python3 scripts/finetune.py --speakers {args.speaker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
