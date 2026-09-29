"""
physics_laser_sim.py
====================
Physics-based laser-microphone simulator: turns clean spoken-digit recordings
into realistic "heard through the laser" recordings, so the model can be
trained for laser audio even before the hardware produces good data.

THE SIGNAL CHAIN THAT IS SIMULATED
----------------------------------
    voice -> air pressure p(t)
          -> (1) surface vibration x(t)      mass-spring-damper modes
          -> (2) laser spot moves on the photodiode edge   erf() knife-edge optics
          -> (3) photocurrent + noise        shot noise, electronic noise
          -> (4) interference                60 Hz mains hum + harmonics, 120 Hz light flicker
          -> (5) receiver electronics        AC coupling (high-pass), amplifier bandwidth (low-pass)
          -> (6) ADC                         gain + clipping at full scale

(1) SURFACE. A thin surface driven by sound pressure behaves like a set of
    damped mass-spring resonators. Each mode i has a natural frequency f_i and
    quality factor Q_i:
        X_i(s) / P(s) = w_i^2 / (s^2 + (w_i/Q_i) s + w_i^2),   w_i = 2*pi*f_i
    Below f_i the surface follows the pressure; above it the response falls
    at -40 dB/decade (why laser audio sounds muffled); near f_i it rings.
    1-3 modes are summed with random weights.

(2) OPTICS. A laser spot (Gaussian, width w) sits over the photodiode edge.
    The fraction of light on the diode is
        I(x) = 0.5 * (1 + erf((x0 + x) / (sqrt(2) * w)))
    x0 = alignment offset (0 = spot centred on the edge = maximum sensitivity).
    For small x this is linear; for large x or a badly placed spot it
    saturates -- the reason alignment matters so much.

(3) DETECTOR. Shot noise has variance proportional to the light level
    (sigma ~ sqrt(I)); electronic noise is white.

(4) INTERFERENCE. Mains hum at ~60 Hz with decaying harmonics, and ambient
    light flicker at 120 Hz, both with random level and phase.

(5)-(6) ELECTRONICS. 1st-order AC coupling (a few-30 Hz), amplifier
    low-pass (3-8 kHz), random gain, hard clipping at +-1 like the PCM1808.

Every parameter is randomised per clip over physically plausible ranges
("domain randomisation"), so the model learns what stays constant -- the
digit -- across many possible laser setups, including ours.

CALIBRATION (optional, once the laser hears speech)
---------------------------------------------------
Pass --calibrate-from <speaker> to use REAL paired recordings (standard mic +
laser captured at the same time by record_daq_wav.py or
replay_record_laser.py --save-std). The script measures your hardware's actual
frequency response |H(f)| = |Laser(f)| / |Mic(f)| and noise floor, and uses
those instead of the random surface model (with small random variation).

USAGE (project root)
--------------------
    # 300 simulated laser clips per digit + 500 'unknown', plus a preview figure
    python3 scripts/physics_laser_sim.py --per-digit 300 --unknown 500 --preview

    # fine-tune the models on them
    python3 scripts/finetune.py --speakers physlaser

Output files: data/raw/<digit>_physlaser_<n>.wav and data/raw/unknown/unk_physlaser_<n>.wav
Preview figure (for reports): results/plots/physics_laser_sim.png
Parameters of every generated clip: results/physics_laser_sim_params.csv
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import bilinear, butter, lfilter, resample_poly, sosfilt
from scipy.special import erf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import DIGIT_LABELS, PLOTS_DIR, RAW_DATA_DIR, RESULTS_DIR, SAMPLE_RATE, UNKNOWN_LABEL  # noqa: E402
from preprocess import load_waveform_from_array, save_capture, save_wav  # noqa: E402,F401

FS = 16000  # internal simulation rate (Hz); output is resampled to SAMPLE_RATE
SKIP_TAGS = ("_laser_", "laserplay", "lasersynth", "physlaser")  # never re-simulate laser/derived clips


# ---------------------------------------------------------------------------
# Physics blocks
# ---------------------------------------------------------------------------
def surface_response(p: np.ndarray, modes, fs: int = FS) -> np.ndarray:
    """Sum of damped mass-spring modes driven by pressure p (displacement, arb. units)."""
    x = np.zeros_like(p)
    for f0, q, weight in modes:
        w0 = 2 * np.pi * f0
        b, a = bilinear([w0 ** 2], [1.0, w0 / q, w0 ** 2], fs=fs)
        x += weight * lfilter(b, a, p)
    return x


def knife_edge_optics(x: np.ndarray, depth: float, offset: float) -> np.ndarray:
    """Fraction of a Gaussian spot on the photodiode as the spot moves by x.

    x is scaled so its peak equals `depth` beam widths; `offset` is the
    alignment (in beam widths) of the spot centre relative to the edge.
    """
    peak = np.max(np.abs(x)) or 1.0
    xn = x / peak * depth
    return 0.5 * (1.0 + erf((offset + xn) / np.sqrt(2.0)))


def detector_noise(intensity: np.ndarray, shot: float, electronic: float, rng) -> np.ndarray:
    return (intensity
            + shot * np.sqrt(np.clip(intensity, 0, None)) * rng.standard_normal(intensity.size)
            + electronic * rng.standard_normal(intensity.size))


def interference(n: int, hum: float, flicker: float, rng, fs: int = FS) -> np.ndarray:
    t = np.arange(n) / fs
    f_mains = rng.uniform(59.9, 60.1)
    out = np.zeros(n)
    for k in range(1, 8):  # 60, 120, ... 420 Hz, decaying harmonics
        out += hum * rng.uniform(0.2, 1.0) / k ** rng.uniform(0.7, 1.5) * np.sin(
            2 * np.pi * k * f_mains * t + rng.uniform(0, 2 * np.pi))
    for k in (1, 2, 3):  # light flicker at 120 Hz and harmonics
        out += flicker / k * np.sin(2 * np.pi * 120 * k * t + rng.uniform(0, 2 * np.pi))
    return out


def electronics(v: np.ndarray, hp_hz: float, lp_hz: float, fs: int = FS) -> np.ndarray:
    sos_hp = butter(1, hp_hz / (fs / 2), btype="highpass", output="sos")
    sos_lp = butter(2, min(lp_hz, fs / 2 * 0.95) / (fs / 2), btype="lowpass", output="sos")
    return sosfilt(sos_lp, sosfilt(sos_hp, v))


def adc(v: np.ndarray, headroom: float) -> np.ndarray:
    """Scale so the speech peak sits at `headroom` of full scale, then clip at +-1."""
    peak = np.max(np.abs(v)) or 1.0
    return np.clip(v / peak * headroom, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Calibration from real paired recordings (optional)
# ---------------------------------------------------------------------------
def _load(path: Path, fs: int = FS) -> np.ndarray:
    a, sr = sf.read(str(path), dtype="float32", always_2d=True)
    a = a.mean(axis=1).astype(np.float64)
    if sr != fs:
        g = np.gcd(int(sr), fs)
        a = resample_poly(a, fs // g, int(sr) // g)
    return a


def calibrate(raw_dir: Path, speaker: str, nfft: int = 1024):
    """Measure |H(f)| = |laser|/|mic| from paired clips and the laser noise floor."""
    pairs = []
    for lp in raw_dir.rglob(f"*_{speaker}_laser_*.wav"):
        for sp in (lp.with_name(lp.name.replace(f"_{speaker}_laser_", f"_{speaker}_std_")),
                   lp.with_name(lp.name.replace(f"_{speaker}_laser_", f"_{speaker}ref_std_"))):
            if sp.exists():
                pairs.append((sp, lp))
                break
    if len(pairs) < 5:
        raise SystemExit(f"--calibrate-from {speaker}: found only {len(pairs)} std/laser pairs "
                         f"(need >= 5). Record with record_daq_wav.py --channel both, or "
                         f"replay_record_laser.py --save-std.")
    num = np.zeros(nfft // 2 + 1)
    den = np.zeros(nfft // 2 + 1)
    noise = []
    for sp, lp in pairs:
        s, l = _load(sp), _load(lp)
        n = min(len(s), len(l))
        s, l = s[:n], l[:n]
        for i in range(0, n - nfft, nfft // 2):
            S = np.fft.rfft(s[i:i + nfft] * np.hanning(nfft))
            L = np.fft.rfft(l[i:i + nfft] * np.hanning(nfft))
            num += np.abs(L * np.conj(S))
            den += np.abs(S) ** 2
        # quietest 20% of 32 ms frames = noise floor
        frames = l[: n - n % 512].reshape(-1, 512)
        e = np.sqrt((frames ** 2).mean(axis=1))
        noise.append(np.percentile(e, 20) / (np.max(np.abs(l)) or 1.0))
    H = num / (den + 1e-12)
    H = np.convolve(H, np.ones(9) / 9, mode="same")  # smooth across frequency
    H /= np.max(H) or 1.0
    print(f"Calibrated from {len(pairs)} real std/laser pairs; relative noise floor "
          f"{np.median(noise):.3f}")
    return H, float(np.median(noise))


def apply_measured_response(p: np.ndarray, H: np.ndarray, rng) -> np.ndarray:
    """Filter by the measured |H(f)| (zero-phase), +-3 dB random ripple per clip."""
    nfft = 2 * (len(H) - 1)
    ripple = 10 ** (rng.uniform(-3, 3, size=8) / 20)
    ripple = np.interp(np.linspace(0, 7, len(H)), np.arange(8), ripple)
    n = int(2 ** np.ceil(np.log2(len(p) + nfft)))
    Hf = np.interp(np.linspace(0, 1, n // 2 + 1), np.linspace(0, 1, len(H)), H * ripple)
    return np.fft.irfft(np.fft.rfft(p, n) * Hf, n)[: len(p)]


# ---------------------------------------------------------------------------
# One clip
# ---------------------------------------------------------------------------
def simulate(clean: np.ndarray, rng, calib=None):
    """Return (laser_signal, params) for one clean waveform at FS."""
    p = clean / (np.max(np.abs(clean)) or 1.0)
    pad = np.zeros(int(0.15 * FS))
    p = np.concatenate([pad, p, pad])  # silence around the word, like a real capture

    prm = {}
    if calib is not None:
        H, noise_floor = calib
        x = apply_measured_response(p, H, rng)
        prm["surface"] = "measured"
    else:
        n_modes = int(rng.integers(1, 4))
        modes = []
        for _ in range(n_modes):
            modes.append((rng.uniform(250, 3000), rng.uniform(0.7, 10), rng.uniform(0.3, 1.0)))
        x = surface_response(p, modes)
        prm["surface"] = ";".join(f"{f:.0f}Hz/Q{q:.1f}" for f, q, _ in modes)
        noise_floor = None

    depth = rng.uniform(0.05, 1.2)     # vibration amplitude in beam widths
    offset = rng.uniform(-1.2, 1.2)    # alignment of spot vs. edge (0 = best)
    intensity = knife_edge_optics(x, depth, offset)
    prm.update(depth=round(depth, 3), offset=round(offset, 3))

    # Signal swing actually produced by the optics -> noise levels relative to it.
    swing = np.std(intensity - intensity.mean()) or 1e-6
    snr_db = rng.uniform(0, 30)  # speech-to-noise at the detector
    if noise_floor is not None:
        snr_db = float(np.clip(-20 * np.log10(max(noise_floor, 1e-4)) + rng.uniform(-6, 6), -3, 40))
    noise_rms = swing / 10 ** (snr_db / 20)
    shot_frac = rng.uniform(0.2, 0.8)
    mean_i = max(float(intensity.mean()), 1e-3)
    v = detector_noise(intensity,
                       shot=noise_rms * shot_frac / np.sqrt(mean_i),
                       electronic=noise_rms * (1 - shot_frac), rng=rng)

    hum_db = rng.uniform(-20, 12)      # hum relative to the speech swing
    v += interference(len(v), hum=swing * 10 ** (hum_db / 20),
                      flicker=swing * 10 ** (rng.uniform(-40, -5) / 20), rng=rng)

    hp, lp = rng.uniform(3, 30), rng.uniform(3000, 7500)
    v = electronics(v, hp, lp)
    headroom = rng.uniform(0.2, 1.3)   # >1 means the ADC clips
    v = adc(v, headroom)
    prm.update(snr_db=round(snr_db, 1), hum_db=round(hum_db, 1), hp_hz=round(hp, 1),
               lp_hz=round(lp), headroom=round(headroom, 2))
    return v, prm


# ---------------------------------------------------------------------------
def _pick(raw_dir: Path, per_digit: int, n_unknown: int, rng):
    digits = {d: [] for d in DIGIT_LABELS}
    unknown = []
    for pth in raw_dir.rglob("*.wav"):
        if any(t in pth.name for t in SKIP_TAGS):
            continue
        if pth.parent.name == UNKNOWN_LABEL:
            unknown.append(pth)
            continue
        lbl = pth.stem.split("_")[0]
        if lbl in digits:
            digits[lbl].append(pth)
    jobs = []
    for d, items in digits.items():
        if not items:
            continue
        rng.shuffle(items)
        # sample with replacement if asked for more than exist (each copy gets new physics)
        jobs += [(d, items[i % len(items)]) for i in range(per_digit)]
    rng.shuffle(unknown)
    if unknown:
        jobs += [(UNKNOWN_LABEL, unknown[i % len(unknown)]) for i in range(n_unknown)]
    return jobs, {d: len(v) for d, v in digits.items()}, len(unknown)


def _preview(examples, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(examples)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.4 * n))
    axes = np.atleast_2d(axes)
    for r, (name, clean, sim, prm) in enumerate(examples):
        for c, (sig, title) in enumerate(((clean, f"clean mic: {name}"),
                                          (sim, "simulated laser: " + ", ".join(
                                              f"{k}={v}" for k, v in prm.items())))):
            ax = axes[r, c]
            ax.specgram(sig + 1e-6 * np.random.default_rng(0).standard_normal(sig.size), NFFT=256, Fs=FS, noverlap=192, cmap="magma")
            ax.set_ylim(0, 4000)
            import textwrap
            ax.set_title("\n".join(textwrap.wrap(title, 75)), fontsize=7)
            ax.set_xlabel("time (s)", fontsize=7)
            ax.set_ylabel("Hz", fontsize=7)
            ax.tick_params(labelsize=6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"Preview figure -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Physics-based laser-microphone simulator for training data.")
    ap.add_argument("--per-digit", type=int, default=300, help="simulated clips per digit (default 300)")
    ap.add_argument("--unknown", type=int, default=500, help="simulated 'unknown' clips (default 500)")
    ap.add_argument("--speaker", default="physlaser", help="speaker tag in filenames (default physlaser)")
    ap.add_argument("--calibrate-from", default=None,
                    help="speaker tag of REAL paired std/laser recordings to calibrate the surface "
                         "response and noise floor from (optional)")
    ap.add_argument("--preview", action="store_true", help="also save a spectrogram figure of 4 examples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw_dir = Path(RAW_DATA_DIR)
    rng_py = random.Random(args.seed)
    rng = np.random.default_rng(args.seed)
    calib = calibrate(raw_dir, args.calibrate_from) if args.calibrate_from else None

    jobs, counts, n_unk = _pick(raw_dir, args.per_digit, args.unknown, rng_py)
    if not jobs:
        print(f"No clean clips found in {raw_dir}. Run scripts/download_fsdd.py first.")
        return 1
    print(f"Clean source clips per digit: {counts} | unknown: {n_unk}")
    print(f"Generating {len(jobs)} simulated laser clips -> {raw_dir} (*_{args.speaker}_*.wav)")

    unk_dir = raw_dir / UNKNOWN_LABEL
    next_idx = {}
    log_path = Path(RESULTS_DIR) / "physics_laser_sim_params.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_log = not log_path.exists()
    examples = []
    with open(log_path, "a", newline="") as fh:
        writer = None
        for n, (label, src) in enumerate(jobs, 1):
            clean = _load(src)
            sim, prm = simulate(clean, rng, calib)
            folder, prefix = (unk_dir, "unk") if label == UNKNOWN_LABEL else (raw_dir, label)
            key = (folder, prefix)
            if key not in next_idx:
                next_idx[key] = len(list(folder.glob(f"{prefix}_{args.speaker}_*.wav")))
            out = folder / f"{prefix}_{args.speaker}_{next_idx[key]}.wav"
            next_idx[key] += 1
            save_capture(sim.astype(np.float32), FS, out)   # kept at 16 kHz

            row = {"file": out.name, "label": label, "source": src.name, **prm}
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row.keys()), extrasaction="ignore")
                if new_log:
                    writer.writeheader()
            writer.writerow(row)
            if (args.preview and len(examples) < 4 and label != UNKNOWN_LABEL
                    and label not in {e[0].split("_")[0] for e in examples}):
                pad = np.zeros(int(0.15 * FS))
                examples.append((src.name, np.concatenate([pad, clean, pad]), sim, prm))
            if n % 250 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)}")

    if examples:
        _preview(examples, Path(PLOTS_DIR) / "physics_laser_sim.png")
    print(f"Parameters log -> {log_path}")
    print(f"Next:  python3 scripts/finetune.py --speakers {args.speaker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
