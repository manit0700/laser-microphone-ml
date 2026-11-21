"""
check_split.py  (Sprint 8, agent A)
===================================
Independent integrity check of results/sprint8/data/manifest.csv.

Writes (results/sprint8/data/):
    split_integrity.json   every check with its count / offending items
    SPLIT_VERIFIED         ONLY if every leak count is 0: manifest sha256 + timestamp
    sanity_listen.json     RMS / peak / clipped fraction / finite for 20 random sim clips
    sim_examples.png       clean vs simulated spectrograms, 3 clips at different SNR bins
    snr_histogram.png      distribution of sim snr_db with the condition-bin boundaries

Exit code 1 if any check fails or any sampled sim clip is degenerate.

Usage (project root, venv active):
    python scripts/sprint8/check_split.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from config import DIGIT_LABELS, UNKNOWN_LABEL  # noqa: E402

DATA = ROOT / "results" / "sprint8" / "data"
MANIFEST = DATA / "manifest.csv"
SPLITS = ("train", "val", "test")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def shared_across_splits(df: pd.DataFrame, col: str) -> list:
    by = df.groupby(col)["split"].agg(lambda s: sorted(set(s)))
    return sorted(k for k, v in by.items() if len(v) > 1)


def expected_label(file: str, kind: str) -> str | None:
    p = Path(file)
    tok = p.stem.split("_")[0]
    if kind == "clean":
        if tok in DIGIT_LABELS:
            return tok
        return UNKNOWN_LABEL if p.parent.name == UNKNOWN_LABEL else None
    # sim: data/sprint8_sim/<split>/<label>/<source_stem>__simK.wav
    folder = p.parent.name
    name_lab = tok if tok in DIGIT_LABELS else UNKNOWN_LABEL
    return folder if folder == name_lab else None


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-suffix", default="", help="e.g. '_v2' -> manifest_v2.csv, split_integrity_v2.json, "
                    "SPLIT_VERIFIED_V2, sanity_listen_v2.json, sim_examples_v2.png, snr_histogram_v2.png")
    ap.add_argument("--compare-with", default=None,
                    help="path of an older manifest (e.g. results/sprint8/data/manifest.csv) to compare sim audio against "
                         "on 200 shared sim clips -> sim_v1_vs_v2.json + sim_fix_before_after.png")
    args = ap.parse_args()
    global MANIFEST
    sfx = args.manifest_suffix
    MANIFEST = DATA / f"manifest{sfx}.csv"
    df = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
    clean = df[df.kind == "clean"]
    sim = df[df.kind == "sim"]
    res = {"manifest": str(MANIFEST.relative_to(ROOT)), "rows": len(df),
           "clean_rows": len(clean), "sim_rows": len(sim), "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # 1. speakers shared between splits (all rows; sim inherits speaker)
    spk = shared_across_splits(df, "speaker")
    res["speakers_shared_between_splits"] = {"count": len(spk), "items": spk[:50]}

    # 2. source_sha256 shared between splits (a recording's content must live in one split)
    src = shared_across_splits(df, "source_sha256")
    res["source_sha256_shared_between_splits"] = {"count": len(src), "items": src[:50]}

    # 3. sim clips whose source clean clip is in another split (or missing)
    clean_split = dict(zip(clean.file, clean.split))
    bad = [r.file for r in sim.itertuples() if clean_split.get(r.source_file) != r.split]
    res["sim_source_in_other_split"] = {"count": len(bad), "items": bad[:50]}

    # 4. duplicate sha256 in the manifest
    dup_clean = clean[clean.duplicated("sha256", keep=False)]["sha256"].unique().tolist()
    dup_all = df[df.duplicated("sha256", keep=False)]["sha256"].unique().tolist()
    res["duplicate_sha256_among_clean"] = {"count": len(dup_clean), "items": dup_clean[:50]}
    res["duplicate_sha256_in_manifest"] = {"count": len(dup_all), "items": dup_all[:50]}

    # 5. label consistency with file name / folder, and sim folder == split
    lab_bad = [r.file for r in df.itertuples() if expected_label(r.file, r.kind) != r.label]
    res["label_inconsistent"] = {"count": len(lab_bad), "items": lab_bad[:50]}
    fold_bad = [r.file for r in sim.itertuples() if Path(r.file).parts[2] != r.split]
    res["sim_folder_split_mismatch"] = {"count": len(fold_bad), "items": fold_bad[:50]}

    # 6. team speaker never trained; schema; files exist; condition bins
    team = df[df.dataset == "team"]
    res["team_clips_outside_test"] = {"count": int((team.split != "test").sum())}
    want_cols = ["file", "label", "speaker", "split", "kind", "dataset", "synthetic_noise", "sha256",
                 "source_file", "source_sha256", "sim_seed", "sim_params", "snr_db", "condition"]
    res["schema_ok"] = list(df.columns) == want_cols
    missing = [f for f in df.file if not (ROOT / f).exists()]
    res["missing_files"] = {"count": len(missing), "items": missing[:50]}
    snr = pd.to_numeric(sim.snr_db)
    want = np.where(snr >= 20, "sim_good", np.where(snr >= 10, "sim_medium", "sim_poor"))
    res["condition_bin_errors"] = {"count": int((want != sim.condition.values).sum())}
    res["clean_condition_errors"] = {"count": int((clean.condition != "clean").sum())}
    # sc hashes in digits + unknown words share a split (covered by check 1; reported for the record)
    both = set(clean[clean.dataset == "speech_commands"].speaker) & set(clean[clean.dataset == "sc_unknown"].speaker)
    res["sc_hashes_in_digits_and_unknown"] = len(both)
    res["speakers_per_split"] = {s: int(df[df.split == s].speaker.nunique()) for s in SPLITS}
    res["clean_clips_per_split"] = {s: int((clean.split == s).sum()) for s in SPLITS}
    res["sim_clips_per_split"] = {s: int((sim.split == s).sum()) for s in SPLITS}

    # 7. sanity-listen in numbers: 20 random sim clips
    rng = np.random.default_rng(42)
    pick = sim.iloc[rng.choice(len(sim), size=min(20, len(sim)), replace=False)]
    listen, degenerate = [], []
    for r in pick.itertuples():
        a, sr = sf.read(str(ROOT / r.file), dtype="float32")
        a = np.asarray(a).reshape(len(a), -1).mean(axis=1)
        finite = bool(np.all(np.isfinite(a)))
        rms = float(np.sqrt(np.mean(a ** 2))) if finite else float("nan")
        peak = float(np.max(np.abs(a))) if finite else float("nan")
        clipped = float(np.mean(np.abs(a) >= 0.999)) if finite else float("nan")
        row = {"file": r.file, "sr": int(sr), "seconds": round(len(a) / sr, 3), "rms": round(rms, 4),
               "peak": round(peak, 4), "clipped_frac": round(clipped, 4), "finite": finite,
               "all_zero": bool(finite and peak == 0.0), "snr_db": float(r.snr_db), "condition": r.condition,
               "headroom": json.loads(r.sim_params)["headroom"]}
        # degenerate: NaN, all-zero, essentially silent, or almost entirely clipped
        row["degenerate"] = (not finite) or row["all_zero"] or rms < 1e-3 or clipped > 0.5 or sr != 16000
        listen.append(row)
        if row["degenerate"]:
            degenerate.append(r.file)
    (DATA / f"sanity_listen{sfx}.json").write_text(json.dumps(listen, indent=1))
    res["sanity_listen"] = {"n": len(listen), "degenerate_count": len(degenerate), "degenerate": degenerate,
                            "rms_min": min(x["rms"] for x in listen), "rms_max": max(x["rms"] for x in listen),
                            "clipped_frac_max": max(x["clipped_frac"] for x in listen),
                            "n_with_any_clipping": sum(x["clipped_frac"] > 0 for x in listen)}

    leak_keys = ["speakers_shared_between_splits", "source_sha256_shared_between_splits",
                 "sim_source_in_other_split", "duplicate_sha256_among_clean", "duplicate_sha256_in_manifest",
                 "label_inconsistent", "sim_folder_split_mismatch", "team_clips_outside_test", "missing_files",
                 "condition_bin_errors", "clean_condition_errors"]
    failures = [k for k in leak_keys if res[k]["count"] != 0]
    if not res["schema_ok"]:
        failures.append("schema_ok")
    if degenerate:
        failures.append("sanity_listen")
    res["failures"] = failures
    res["verified"] = not failures
    (DATA / f"split_integrity{sfx}.json").write_text(json.dumps(res, indent=1))

    for k in leak_keys:
        print(f"{k:40s} {res[k]['count']}")
    print(f"{'schema_ok':40s} {res['schema_ok']}")
    print(f"{'sanity_listen degenerate':40s} {len(degenerate)} / {len(listen)}")

    make_figures(df, sim, sfx)
    if args.compare_with:
        compare_versions(Path(args.compare_with), df)

    marker = DATA / f"SPLIT_VERIFIED{sfx.upper()}"
    if failures:
        if marker.exists():
            marker.unlink()
        print(f"FAILED checks: {failures}  (no SPLIT_VERIFIED written)")
        return 1
    msha = sha256_file(MANIFEST)
    marker.write_text(f"manifest_sha256 {msha}\nverified_at {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
                      f"rows {len(df)} clean {len(clean)} sim {len(sim)}\n")
    print(f"all checks passed -> {marker.relative_to(ROOT)} (manifest sha256 {msha[:16]}...)")
    return 0


def make_figures(df: pd.DataFrame, sim: pd.DataFrame, sfx: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mt

    INK, MUTED, GRID, ACCENT = "#1f2937", "#6b7280", "#e5e7eb", "#3b6fb6"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "text.color": INK})

    # --- snr histogram: one series, one hue, condition boundaries drawn + labelled
    snr = pd.to_numeric(sim.snr_db).values
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(snr, bins=np.arange(0, 31, 1), color=ACCENT, edgecolor="white", linewidth=0.8)
    for x, lab in ((10, "sim_poor | sim_medium"), (20, "sim_medium | sim_good")):
        ax.axvline(x, color=MUTED, lw=1, ls="--")
        ax.text(x, ax.get_ylim()[1] * 0.97, lab, ha="center", va="top", fontsize=8, color=MUTED,
                bbox=dict(fc="white", ec="none", pad=1))
    counts = sim.condition.value_counts()
    ax.set_title(f"Simulated-laser SNR, all splits (n={len(sim)}): "
                 + ", ".join(f"{c} {counts.get(c, 0)}" for c in ("sim_poor", "sim_medium", "sim_good")),
                 fontsize=9, loc="left")
    ax.set_xlabel("snr_db (speech swing / detector noise, from simulate())")
    ax.set_ylabel("clips")
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(DATA / f"snr_histogram{sfx}.png", dpi=130)
    plt.close(fig)

    # --- clean vs sim spectrograms for 3 clips, one per SNR bin (digits, deterministic choice)
    rng = np.random.default_rng(7)
    examples = []
    for cond in ("sim_good", "sim_medium", "sim_poor"):
        pool = sim[(sim.condition == cond) & (sim.label != UNKNOWN_LABEL)]
        if len(pool):
            examples.append(pool.iloc[int(rng.integers(len(pool)))])
    if not examples:
        return
    import scipy.signal as ss
    fig, axes = plt.subplots(len(examples), 2, figsize=(10, 2.7 * len(examples)), squeeze=False)
    for r, ex in enumerate(examples):
        a_clean, sr_c = sf.read(str(ROOT / ex.source_file), dtype="float32")
        a_sim, sr_s = sf.read(str(ROOT / ex.file), dtype="float32")
        prm = json.loads(ex.sim_params)
        for c, (a, sr, title) in enumerate((
                (a_clean, sr_c, f"clean  {Path(ex.source_file).name}  ({sr_c} Hz, label {ex.label}, {ex.split})"),
                (a_sim, sr_s, f"sim  {ex.condition}  snr {prm['snr_db']} dB  hum {prm['hum_db']} dB  lp {prm['lp_hz']} Hz\n"
                              f"headroom {prm['headroom']}  depth {prm['depth']}  offset {prm['offset']}  surface {prm['surface'][:34]}"))):
            a = np.asarray(a).reshape(len(a), -1).mean(axis=1)
            f, t, S = ss.spectrogram(a, fs=sr, nperseg=256, noverlap=192)
            ax = axes[r, c]
            ax.pcolormesh(t, f, 10 * np.log10(S + 1e-10), cmap="magma", shading="auto", vmin=-110, vmax=-30)
            ax.set_ylim(0, 4000)
            ax.set_title(title, fontsize=8, loc="left")
            ax.set_ylabel("Hz")
            if r == len(examples) - 1:
                ax.set_xlabel("time (s)")
            ax.yaxis.set_major_locator(mt.MultipleLocator(1000))
    fig.suptitle("Clean source vs. physics_laser_sim variant (same clip, three SNR bins); "
                 "colour = power dB (magma, -110..-30)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(DATA / f"sim_examples{sfx}.png", dpi=130)
    plt.close(fig)
    print(f"figures -> {DATA.relative_to(ROOT)}/sim_examples{sfx}.png, snr_histogram{sfx}.png")


def _clip_stats(path: Path, snr_db: float):
    """Peak position, speech-window RMS and peak for one sim clip.

    The simulator pads 0.15 s of silence before and after the word, so the 'speech window'
    is the clip minus 0.15 s at each end (the AC-coupling click, when present, is in the
    first ~20 ms of the leading pad).
    """
    a, sr = sf.read(str(path), dtype="float32")
    a = np.asarray(a).reshape(len(a), -1).mean(axis=1).astype(np.float64)
    n_pad = int(0.15 * sr)
    speech = a[n_pad:len(a) - n_pad] if len(a) > 2 * n_pad + sr // 10 else a
    i_peak = int(np.argmax(np.abs(a)))
    return {"peak_time_s": i_peak / sr, "peak_in_first_30ms": bool(i_peak < 0.03 * sr),
            "peak": float(np.abs(a).max()), "speech_rms": float(np.sqrt(np.mean(speech ** 2))),
            "first30ms_rms": float(np.sqrt(np.mean(a[: int(0.03 * sr)] ** 2))),
            "speech_peak": float(np.abs(speech).max()), "sr": int(sr)}


def compare_versions(old_manifest: Path, new_df: pd.DataFrame, n: int = 200) -> None:
    """v1 vs v2 of the simulator on the same 200 sim clips (same sim_seed/params, new audio)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scipy.signal as ss

    old = pd.read_csv(old_manifest, dtype=str, keep_default_na=False)
    old_sim = old[old.kind == "sim"].set_index("file")
    new_sim = new_df[new_df.kind == "sim"].copy()
    # same clip <=> same (source_file, variant index). file names differ only by sim dir.
    def key(f):
        p = Path(f); return (p.parts[-3], p.parts[-2], p.name)
    old_by_key = {key(f): f for f in old_sim.index}
    new_sim["key"] = new_sim.file.map(key)
    shared = new_sim[new_sim.key.isin(old_by_key)]
    rng = np.random.default_rng(2026)
    pick = shared.iloc[np.sort(rng.choice(len(shared), size=min(n, len(shared)), replace=False))]
    rows = []
    params_identical = 0
    for r in pick.itertuples():
        of = old_by_key[r.key]
        o = old_sim.loc[of]
        params_identical += int(o.sim_seed == r.sim_seed and o.sim_params == r.sim_params)
        so = _clip_stats(ROOT / of, float(r.snr_db))
        sn = _clip_stats(ROOT / r.file, float(r.snr_db))
        rows.append({"file_v1": of, "file_v2": r.file, "condition": r.condition, "snr_db": float(r.snr_db),
                     "v1": so, "v2": sn})
    def agg(v):
        return {"n": len(rows),
                "frac_peak_in_first_30ms": float(np.mean([x[v]["peak_in_first_30ms"] for x in rows])),
                "median_speech_rms": float(np.median([x[v]["speech_rms"] for x in rows])),
                "median_peak": float(np.median([x[v]["peak"] for x in rows])),
                "median_speech_peak": float(np.median([x[v]["speech_peak"] for x in rows])),
                "median_first30ms_rms": float(np.median([x[v]["first30ms_rms"] for x in rows])),
                "p10_speech_rms": float(np.percentile([x[v]["speech_rms"] for x in rows], 10)),
                "p90_speech_rms": float(np.percentile([x[v]["speech_rms"] for x in rows], 90))}
    out = {"old_manifest": str(old_manifest), "new_manifest": str(MANIFEST.relative_to(ROOT)),
           "n_compared": len(rows), "same_sim_seed_and_params": params_identical,
           "speech_window": "clip minus 0.15 s pad at each end (simulate() pads 0.15 s)",
           "v1": agg("v1"), "v2": agg("v2"), "clips": rows}
    (DATA / "sim_v1_vs_v2.json").write_text(json.dumps(out, indent=1))
    print(f"v1 vs v2 on {len(rows)} sim clips (seed/params identical: {params_identical}/{len(rows)}):")
    for v in ("v1", "v2"):
        a = out[v]
        print(f"  {v}: peak in first 30 ms {a['frac_peak_in_first_30ms']:.3f}, median speech RMS "
              f"{a['median_speech_rms']:.4f}, median peak {a['median_peak']:.3f}, median speech peak {a['median_speech_peak']:.3f}")

    # before/after figure for one clip: the one with the largest v1 click (peak in first 30 ms, lowest speech rms)
    cands = [x for x in rows if x["v1"]["peak_in_first_30ms"] and x["condition"] == "sim_good"] or rows
    ex = min(cands, key=lambda x: x["v1"]["speech_rms"])
    INK, MUTED = "#1f2937", "#6b7280"
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    for c, (v, f) in enumerate((("v1 (before fix)", ex["file_v1"]), ("v2 (after fix)", ex["file_v2"]))):
        a, sr = sf.read(str(ROOT / f), dtype="float32")
        a = np.asarray(a).reshape(len(a), -1).mean(axis=1)
        t = np.arange(len(a)) / sr
        ax = axes[0, c]
        ax.plot(t, a, color="#3b6fb6", lw=0.6)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"{v}: {Path(f).name}\nsnr {ex['snr_db']} dB; peak at {_clip_stats(ROOT / f, 0)['peak_time_s']*1000:.0f} ms, "
                     f"speech RMS {_clip_stats(ROOT / f, 0)['speech_rms']:.3f}", fontsize=8, loc="left", color=INK)
        ax.set_ylabel("amplitude (full scale = 1)")
        ax.axvspan(0, 0.03, color="#f3f4f6", zorder=0)
        ax.text(0.031, 0.95, "first 30 ms", fontsize=7, color=MUTED, va="top")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        fq, tt, S = ss.spectrogram(a, fs=sr, nperseg=256, noverlap=192)
        ax2 = axes[1, c]
        ax2.pcolormesh(tt, fq, 10 * np.log10(S + 1e-10), cmap="magma", shading="auto", vmin=-110, vmax=-30)
        ax2.set_ylim(0, 4000)
        ax2.set_xlabel("time (s)"); ax2.set_ylabel("Hz")
    fig.suptitle("physics_laser_sim electronics() start-up click: same clip, same seed and params, before vs after "
                 "the steady-state filter fix\n(top: waveform, bottom: spectrogram dB, magma -110..-30)", fontsize=9,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(DATA / "sim_fix_before_after.png", dpi=130)
    plt.close(fig)
    print(f"figure -> {(DATA / 'sim_fix_before_after.png').relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
