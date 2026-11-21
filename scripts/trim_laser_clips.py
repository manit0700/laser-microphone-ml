"""
trim_laser_clips.py
===================
Make a TRIMMED COPY of the dataset for a fine-tuning experiment.

Why: the pipeline already cuts every long clip down to 1 s around its loudest
128 ms (preprocess.center_on_peak). On the laser channel the loudest moment is
sometimes a table thump or a click, not the word. This script finds the word
with a thump-resistant energy envelope and cuts the clip to a short window
around it, so the pipeline's own centering then lands on the word.

Safety
------
- The source folder is only READ. No source file is changed, moved or deleted.
- Output goes to a NEW folder (default: data/raw_trimmed next to this script).
  Trimmed laser clips are written there under the SAME file names; every other
  WAV is a symlink to the original, so finetune.py sees a complete dataset.
- Same file names + same seed => finetune.py holds out the SAME 160 clips, so
  the before/after accuracy numbers compare the same recordings.
- No model is trained, read or written here.

USAGE (on the Jetson)
---------------------
    python3 scripts/trim_laser_clips.py
    bash scripts/finetune_trimmed.sh            # dry run, saves nothing
"""

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DEFAULT_SPEAKERS = "ali,brooke,brooke2,john,michael,laserali,laserjohn,lasermichaeltrim"


def _is_target(path: Path, speakers, only_tag: str) -> bool:
    parts = path.stem.split("_")
    if len(parts) < 2 or parts[1] not in speakers:
        return False
    return not only_tag or f"_{only_tag}_" in path.name


def _envelope(x, sr, frame_s, hop_s):
    import numpy as np
    frame, hop = max(1, int(frame_s * sr)), max(1, int(hop_s * sr))
    if x.size < frame:
        return np.array([float(np.sum(x * x))]), hop
    n = 1 + (x.size - frame) // hop
    csum = np.concatenate([[0.0], np.cumsum(x * x)])
    starts = np.arange(n) * hop
    return csum[starts + frame] - csum[starts], hop


def find_word_window(x, sr, seconds):
    """Return (start, end, info) of the `seconds`-long window that holds the word.

    1. Band-pass 200-3800 Hz (same band the laser channel uses).
    2. Frame energy every 10 ms, then a 90 ms median filter: bursts shorter than
       about 45 ms (clicks, single thumps) are flattened, speech survives.
    3. Subtract the noise floor (20th percentile) and cap at the 95th percentile
       so one loud burst cannot outweigh a whole word.
    4. Take the window with the most remaining energy, then re-centre it on the
       energy centroid inside it.
    """
    import numpy as np
    from scipy.ndimage import median_filter
    from scipy.signal import butter, sosfiltfilt

    want = int(round(seconds * sr))
    high = min(3800.0, 0.45 * sr)
    xf = x
    if x.size > 64 and high > 250.0:
        xf = sosfiltfilt(butter(4, [200.0, high], btype="band", fs=sr, output="sos"), x)

    # Where the pipeline's own centering would land: loudest 128 ms.
    env_p, hop_p = _envelope(xf, sr, 0.128, 0.032)
    naive_peak = int(np.argmax(env_p)) * hop_p + int(0.064 * sr)

    if x.size <= want:
        return 0, x.size, {"trimmed": False, "naive_peak_s": naive_peak / sr}

    env, hop = _envelope(xf, sr, 0.020, 0.010)
    smooth = median_filter(env, size=9, mode="nearest")
    act = np.clip(smooth - np.percentile(smooth, 20), 0.0, None)
    cap = np.percentile(act, 95)
    if cap > 0:
        act = np.minimum(act, cap)
    w = max(1, min(act.size, want // hop))
    csum = np.concatenate([[0.0], np.cumsum(act)])
    sums = csum[w:] - csum[:-w]
    f0 = int(np.argmax(sums))
    seg = act[f0:f0 + w]
    centre_f = f0 + (float(np.sum(seg * np.arange(seg.size)) / seg.sum()) if seg.sum() > 0 else w / 2.0)
    centre = int(centre_f * hop + 0.010 * sr)
    start = max(0, min(centre - want // 2, x.size - want))
    end = start + want
    return start, end, {
        "trimmed": True, "naive_peak_s": naive_peak / sr, "word_centre_s": centre / sr,
        "naive_peak_inside": bool(start <= naive_peak < end),
        "shift_s": abs(naive_peak - centre) / sr,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Trimmed copy of the laser clips (source is never changed).")
    p.add_argument("--repo", default=str(Path.home() / "laser_microphone_ml"),
                   help="project copy whose data/raw holds the recordings (read only)")
    p.add_argument("--src", default=None, help="source data folder (default: <repo>/data/raw)")
    p.add_argument("--out-dir", default=str(HERE / "data" / "raw_trimmed"))
    p.add_argument("--speakers", default=DEFAULT_SPEAKERS)
    p.add_argument("--only-tag", default="laser", help="only files containing _<tag>_ ('' = all)")
    p.add_argument("--seconds", type=float, default=1.25,
                   help="length kept around the word (default 1.25; the pipeline then takes its 1.0 s)")
    p.add_argument("--report", default=str(HERE / "results" / "jetson_trim" / "trim_report.json"))
    args = p.parse_args()

    import numpy as np
    import soundfile as sf

    src = Path(args.src).expanduser().resolve() if args.src else Path(args.repo).expanduser().resolve() / "data" / "raw"
    out = Path(args.out_dir).expanduser().resolve()
    if not src.is_dir():
        print(f"ERROR: source folder not found: {src}")
        return 2
    if out == src or src in out.parents or out in src.parents:
        print(f"ERROR: --out-dir must be outside the source folder.\n  source {src}\n  out    {out}")
        return 2
    speakers = {s.strip() for s in args.speakers.split(",") if s.strip()}
    out.mkdir(parents=True, exist_ok=True)
    print(f"Host: {socket.gethostname()} | source (read only): {src}\nOutput: {out} | keep {args.seconds:.2f} s")

    wavs = sorted(src.rglob("*.wav"))
    stats, linked, written, skipped = {}, 0, 0, 0
    for path in wavs:
        dest = out / path.relative_to(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not _is_target(path, speakers, args.only_tag):
            if not (dest.exists() or dest.is_symlink()):
                os.symlink(path, dest)
            linked += 1
            continue
        info_in = sf.info(str(path))
        data, sr = sf.read(str(path), dtype="float64", always_2d=True)
        mono = data.mean(axis=1)
        start, end, info = find_word_window(mono, sr, args.seconds)
        if dest.is_symlink():
            dest.unlink()                       # a link in OUR output folder, never a source file
        sf.write(str(dest), data[start:end], sr, subtype=info_in.subtype)
        written += 1
        s = stats.setdefault(path.stem.split("_")[1], {"n": 0, "dur": [], "rms": [], "shift": [], "outside": 0, "trimmed": 0})
        s["n"] += 1
        s["dur"].append(mono.size / sr)
        s["rms"].append(float(np.sqrt(np.mean(mono * mono))) if mono.size else 0.0)
        if info["trimmed"]:
            s["trimmed"] += 1
            s["shift"].append(info["shift_s"])
            s["outside"] += 0 if info["naive_peak_inside"] else 1
        else:
            skipped += 1

    if not written:
        print(f"ERROR: no clips matched speakers {sorted(speakers)} with tag '{args.only_tag}' in {src}")
        return 3

    print(f"\n{written} clips written ({written - skipped} trimmed, {skipped} already short), {linked} other files linked")
    print(f"\n{'session':20s} {'clips':>5s} {'length':>7s} {'rms':>7s} {'peak moved':>10s} {'peak outside':>12s}")
    report = {"host": socket.gethostname(), "date": time.strftime("%Y-%m-%d %H:%M:%S"), "source": str(src),
              "out_dir": str(out), "seconds": args.seconds, "speakers": sorted(speakers),
              "only_tag": args.only_tag, "written": written, "linked": linked, "sessions": {}}
    tot_out = tot_trim = 0
    for name in sorted(stats):
        s = stats[name]
        shift = float(np.median(s["shift"])) if s["shift"] else 0.0
        row = {"clips": s["n"], "median_length_s": float(np.median(s["dur"])), "median_rms": float(np.median(s["rms"])),
               "trimmed": s["trimmed"], "median_peak_shift_s": shift, "loudest_point_outside_word_window": s["outside"]}
        report["sessions"][name] = row
        tot_out += s["outside"]
        tot_trim += s["trimmed"]
        print(f"{name:20s} {s['n']:5d} {row['median_length_s']:6.2f}s {row['median_rms']:7.4f} {shift:9.2f}s {s['outside']:5d}/{s['trimmed']:<5d}")
    print(f"\n'peak outside' = clips whose loudest 128 ms is NOT in the word window (a thump or click won).")
    print(f"Total: {tot_out} of {tot_trim} trimmed clips. Near 0 means trimming changes little.")
    rep = Path(args.report).expanduser()
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(report, indent=2))
    print(f"report -> {rep}\n\nNext (dry run, saves no model):\n  bash scripts/finetune_trimmed.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
