"""
record_daq_wav.py
==================
Record spoken digits through the PCM1808/laser DAQ hardware and save them as
WAV files into data/raw/, ready for training. Same idea as record_wav.py, but
captures from the real hardware (signal_backend._DAQSource) instead of a
laptop microphone, so the model can learn what a laser-transduced digit looks
like instead of only ever seeing clean voice recordings.

The PCM1808 is stereo: the LEFT channel is the standard reference mic and the
RIGHT channel is the laser receiver (same mapping as signal_dashboard2.py).
By default each take saves BOTH, sample-aligned, as a pair of clips of the same
utterance -- the laser clip is the laser-domain training example, the std clip
is a matching clean-voice example (and lets you compare the two channels).

Files are named in the FSDD style the dataset already understands:

        <digit>_<speaker>_std_<index>.wav      e.g.  7_manit_std_0.wav
        <digit>_<speaker>_laser_<index>.wav    e.g.  7_manit_laser_0.wav

so `dataset.py` picks up the label (the first token) automatically -- just
run `python src/train.py` afterward to fold these into training.

REQUIRES
--------
    pip install pyaudio
(needs the PCM1808 connected; see docs/laser_daq_interface.md)

USAGE
-----
Record 5 paired takes of digit 7 as speaker "manit":
    python scripts/record_daq_wav.py --digit 7 --speaker manit --takes 5

Record 10 paired takes of every digit 0-9:
    python scripts/record_daq_wav.py --all --speaker manit --takes 10

Record 50 NON-digit takes (silence, taps, bumps, coughs, other words) for the
'unknown' class -- the model learns what to REJECT:
    python scripts/record_daq_wav.py --unknown 50 --speaker manit --channel both

Options:
    --digit N        which digit to record (0-9)
    --unknown N      record N non-digit takes into data/raw/unknown/
    --all            loop over all digits 0-9 instead of a single --digit
    --speaker NAME   speaker label used in the filename (default: "me")
    --takes K        how many recordings per digit (default: 3)
    --seconds S      length of each recording in seconds (default: 1.5)
    --channel C      both (default: std + laser pair) | laser | std | mix
                     ("mix" averages the two channels -- avoid it for laser data)

After each take it prints each channel's level (RMS). If the laser channel sits
near zero while the std mic responds, the laser path isn't picking up the speech
-- fix the optics/wiring before recording more, no model can learn from silence.

Wrong device picked up? Same override as the rest of the DAQ tooling:
    LMML_DAQ_DEVICE=<index> python scripts/record_daq_wav.py --all
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import DIGIT_LABELS, RAW_DATA_DIR, SAMPLE_RATE  # noqa: E402
from preprocess import load_waveform_from_array, save_wav  # noqa: E402
from signal_backend import _DAQSource  # noqa: E402

# A channel quieter than this while speaking is effectively silence (full scale = 1.0).
SILENT_RMS = 0.002


def _next_index(out_dir: Path, digit: str, speaker_tag: str) -> int:
    """Find the next free index so we never overwrite existing recordings."""
    existing = list(out_dir.glob(f"{digit}_{speaker_tag}_*.wav"))
    return len(existing)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def _save(raw: np.ndarray, rate: int, path: Path) -> None:
    """Resample a captured channel to the project SAMPLE_RATE and write the WAV."""
    save_wav(load_waveform_from_array(raw, rate), path, SAMPLE_RATE)


def record_one(daq: _DAQSource, digit: str, speaker: str, channel: str, indices: dict,
               seconds: float, out_dir: Path):
    """Record one take from the already-open DAQ stream and save it."""
    if digit == "unk":
        print(f"  >> Make the sound NOW ({seconds:.1f}s)")
    else:
        print(f"  >> Say '{digit}' NOW ({seconds:.1f}s)")
    time.sleep(seconds)
    std, laser = daq.latest_channels(seconds)   # sample-aligned pair

    if std.size < int(0.5 * daq.rate):
        print(f"     WARNING: only {std.size} samples captured, skipping this take")
        return

    wanted = {"both": ("std", "laser"), "std": ("std",), "laser": ("laser",), "mix": ("mix",)}[channel]
    signals = {"std": std, "laser": laser, "mix": (std + laser) / 2.0}
    levels = []
    for tag in wanted:
        # 'mix' keeps the old unsuffixed naming: <digit>_<speaker>_<n>.wav
        speaker_tag = speaker if tag == "mix" else f"{speaker}_{tag}"
        idx = indices.setdefault(tag, _next_index(out_dir, digit, speaker_tag))
        path = out_dir / f"{digit}_{speaker_tag}_{idx}.wav"
        _save(signals[tag], daq.rate, path)
        indices[tag] = idx + 1
        levels.append(f"{tag} rms={_rms(signals[tag]):.4f} -> {path.name}")
    print("     saved: " + " | ".join(levels))

    if "laser" in wanted and _rms(laser) < SILENT_RMS:
        print(f"     WARNING: laser channel is near-silent (rms {_rms(laser):.5f}); "
              f"standard mic rms {_rms(std):.4f}. The laser path may not be picking up the speech.")


def main():
    parser = argparse.ArgumentParser(description="Record spoken-digit WAV files via the PCM1808/laser DAQ.")
    parser.add_argument("--digit", type=str, choices=DIGIT_LABELS,
                        help="single digit to record (0-9)")
    parser.add_argument("--all", action="store_true",
                        help="record every digit 0-9")
    parser.add_argument("--speaker", type=str, default="me",
                        help="speaker label used in the filename")
    parser.add_argument("--takes", type=int, default=3,
                        help="recordings per digit")
    parser.add_argument("--seconds", type=float, default=1.5,
                        help="length of each recording (seconds)")
    parser.add_argument("--unknown", type=int, default=0, metavar="N",
                        help="record N NON-digit takes for the 'unknown' class (silence, room noise, "
                             "taps/bumps on the laser surface, coughs, other words...) into data/raw/unknown/")
    parser.add_argument("--channel", choices=("both", "laser", "std", "mix"), default="both",
                        help="which channel(s) to save (default: both, as a pair)")
    args = parser.parse_args()

    if not args.all and args.digit is None and args.unknown <= 0:
        parser.error("specify --digit N, --all, or --unknown N")

    out_dir = Path(RAW_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    daq = _DAQSource(buffer_seconds=max(args.seconds + 1.0, 2.0))
    if not daq.available:
        print(f"ERROR: DAQ not available: {daq.error}")
        print("Check: pyaudio installed, PCM1808 connected, LMML_DAQ_DEVICE if auto-detect picked wrong.")
        sys.exit(1)

    print(f"Device: [{daq.device}] {daq.device_name}")
    print(f"Recording at {daq.rate} Hz (resampled to {SAMPLE_RATE} Hz), speaker='{args.speaker}', "
          f"channel={args.channel}, {args.takes} take(s) per digit. Files go to {out_dir}")

    digits = DIGIT_LABELS if args.all else ([args.digit] if args.digit is not None else [])

    daq.start()
    try:
        if args.unknown > 0:
            # Non-digit sounds the model must learn to REJECT. Saved as
            # data/raw/unknown/unk_<speaker>_<std|laser>_<n>.wav -> label 'unknown'.
            unk_dir = out_dir / "unknown"
            unk_dir.mkdir(parents=True, exist_ok=True)
            prompts = ["stay SILENT", "just room noise (don't speak)", "TAP the laser surface",
                       "BUMP / knock the table", "cough or clear your throat",
                       "say a NON-digit word (hello, yes, stop, left...)", "clap once",
                       "rub or brush the surface", "move / walk near the setup",
                       "speak a short phrase with no digits"]
            indices_u: dict = {}
            print(f"\nUnknown class: {args.unknown} takes -> {unk_dir}")
            for t in range(args.unknown):
                what = prompts[t % len(prompts)]
                input(f"  [{t + 1}/{args.unknown}] Next: {what}. Press ENTER, then do it...")
                record_one(daq, "unk", args.speaker, args.channel, indices_u, args.seconds, unk_dir)
        for digit in digits:
            indices: dict = {}   # next free file index per channel tag, for this digit
            print(f"\nDigit '{digit}':")
            for t in range(args.takes):
                input(f"  Press ENTER to record take {t + 1}/{args.takes} of '{digit}'...")
                record_one(daq, digit, args.speaker, args.channel, indices, args.seconds, out_dir)
    finally:
        daq.stop()

    total = len(list(out_dir.glob("*.wav")))
    print(f"\nDone. data/raw/ now has {total} WAV files. Next: python src/train.py --augment")


if __name__ == "__main__":
    main()
