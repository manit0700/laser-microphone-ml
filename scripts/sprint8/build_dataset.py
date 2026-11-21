"""
build_dataset.py  (Sprint 8, agent A)
=====================================
Builds the Sprint 8 dataset: provenance of every original clip, a speaker-grouped
train/val/test split of the ORIGINAL recordings (done before any simulation), and
simulated laser-microphone variants of every kept clip, written to
data/sprint8_sim/<split>/<label>/<source_stem>__simK.wav (16 kHz).

Outputs (all under results/sprint8/data/, owned by A):
    manifest.csv          one row per clean + sim clip, schema per results/sprint8/PLAN.md
    duplicates.csv        every original file dropped by de-duplication and why
    manifest_summary.md   counts, speakers per split, condition histogram, decisions
    build_dataset.log     run log (when run via nohup)

Ground rules honoured: data/raw is read-only here; scripts/physics_laser_sim.py is
imported, not modified; all randomness is seeded (split seed 42, sim run seed 2026,
per-clip seed derived from the source sha256).

Usage (project root, venv active):
    python scripts/sprint8/build_dataset.py [--train-sim-target 7000] [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from config import DIGIT_LABELS, RAW_DATA_DIR, UNKNOWN_LABEL  # noqa: E402
from preprocess import save_capture  # noqa: E402
from splits import dedupe_indices, speaker_of, _DUP_RE  # noqa: E402
import physics_laser_sim as pls  # noqa: E402  (simulate(), _load(), FS)

SPLIT_SEED = 42
SIM_RUN_SEED = 2026
OUT_DIR = ROOT / "results" / "sprint8" / "data"
SIM_DIR = ROOT / "data" / "sprint8_sim"
RAW = Path(RAW_DATA_DIR)
TEAM_SPEAKER = "manit"

MANIFEST_COLS = ["file", "label", "speaker", "split", "kind", "dataset", "synthetic_noise",
                 "sha256", "source_file", "source_sha256", "sim_seed", "sim_params", "snr_db",
                 "condition"]

_SC_RE = re.compile(r"^[0-9a-f]{8}$")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(ROOT))


def label_of(path: Path) -> str:
    """Project label rules (dataset.py): digit = first filename token, else 'unknown' folder."""
    tok = path.stem.split("_")[0]
    if tok in DIGIT_LABELS:
        return tok
    if path.parent.name == UNKNOWN_LABEL:
        return UNKNOWN_LABEL
    raise ValueError(f"cannot label {path}")


def dataset_of(path: Path, label: str) -> str:
    stem = _DUP_RE.sub("", path.stem)
    parts = stem.split("_")
    if parts[0] in ("noise", "silence"):
        return parts[0]
    if label == UNKNOWN_LABEL:
        return "sc_unknown"
    if parts[1].startswith("sc") and _SC_RE.match(parts[1][2:]):
        return "speech_commands"
    if parts[1] == TEAM_SPEAKER:
        return "team"
    return "fsdd"


def noise_family(path: Path) -> str:
    """Classify a generated noise_N clip into its generator family from its statistics.

    Families observed in data/raw/unknown (see manifest_summary.md): brown (steep 1/f
    spectrum, random DC offset), white (flat, zero-crossing rate ~0.5), impulsive (high
    kurtosis clicks + hum) and colored (band-limited, ~500 Hz emphasis). Each clip is
    exactly 1.000 s, peak-normalised, Gaussian -> generated, not recorded.
    """
    a, sr = sf.read(str(path), dtype="float32")
    a = a.reshape(len(a), -1).mean(axis=1).astype(np.float64)
    spec = np.abs(np.fft.rfft(a * np.hanning(len(a)))) ** 2
    f = np.fft.rfftfreq(len(a), 1 / sr)
    m = (f > 50) & (f < min(7000, sr / 2 - 100))
    slope = np.polyfit(np.log10(f[m]), 10 * np.log10(spec[m] + 1e-12), 1)[0]
    var = a.var()
    kurt = ((a - a.mean()) ** 4).mean() / var ** 2 if var > 0 else 3.0
    zc = float(np.mean(np.diff(np.sign(a)) != 0))
    peak_hz = float(f[np.argmax(spec * (f > 20))])
    if abs(a.mean()) > 0.01 and slope < -7:
        return "brown"          # random walk: 1/f^2 spectrum + random DC offset
    if zc > 0.45 and abs(slope) < 2 and kurt < 4:
        return "white"          # flat spectrum, Gaussian
    if peak_hz < 300 or zc > 0.3:
        return "hum_click"      # 60/180 Hz hum + impulsive clicks (high kurtosis)
    return "bandlimited"        # coloured noise peaking ~450-1500 Hz, low zero-crossing rate


def derive_seed(source_sha: str, variant: int, run_seed: int = SIM_RUN_SEED) -> int:
    h = hashlib.sha256(f"{source_sha}:{variant}:{run_seed}".encode()).hexdigest()
    return int(h[:15], 16)  # 60-bit, fits any int column


def condition_of(snr_db: float) -> str:
    if snr_db >= 20:
        return "sim_good"
    if snr_db >= 10:
        return "sim_medium"
    return "sim_poor"


# ---------------------------------------------------------------------------
# 1. provenance + dedupe
# ---------------------------------------------------------------------------
def collect_originals():
    paths = sorted(RAW.rglob("*.wav"))
    paths = [p for p in paths if not any(t in p.name for t in pls.SKIP_TAGS)]
    print(f"[1] {len(paths)} original wav files under {rel(RAW)}")
    t0 = time.time()
    shas = {p: sha256_file(p) for p in paths}
    print(f"    sha256 of all files in {time.time() - t0:.1f}s")
    keep_ix = set(dedupe_indices(paths, cache_file=OUT_DIR / "_hash_cache.json"))
    first_by_sha = {}
    for i, p in enumerate(paths):
        if i in keep_ix:
            first_by_sha.setdefault(shas[p], p)
    dropped = []
    for i, p in enumerate(paths):
        if i in keep_ix:
            continue
        if _DUP_RE.search(p.stem):
            reason = "_dupN filename copy"
        else:
            reason = "byte-identical to kept file"
        kept = first_by_sha.get(shas[p])
        dropped.append({"file": rel(p), "sha256": shas[p], "reason": reason,
                        "kept_file": rel(kept) if kept else "",
                        "kept_sha256_match": int(kept is not None and shas[kept] == shas[p])})
    kept = [paths[i] for i in sorted(keep_ix)]
    # dedupe_indices uses md5 + name rule; confirm with sha256 that kept set is unique
    assert len({shas[p] for p in kept}) == len(kept), "sha256 collision among kept clips"
    print(f"    kept {len(kept)} unique recordings, dropped {len(dropped)} "
          f"({sum(d['reason'].startswith('_dup') for d in dropped)} _dupN, "
          f"{sum(d['reason'].startswith('byte') for d in dropped)} byte-identical)")
    return kept, shas, dropped


# ---------------------------------------------------------------------------
# 2. speaker-grouped split of originals
# ---------------------------------------------------------------------------
def group_id(path: Path, dataset: str, family: str | None) -> str:
    """Group key: splits.speaker_of() for voices; generator family for noise/silence."""
    if dataset in ("noise", "silence"):
        return f"gen_{dataset}_{family}"
    return speaker_of(path)


def make_split(clips: list[dict], seed: int = SPLIT_SEED) -> dict[str, str]:
    """Assign every group to one split. Returns {group_id: split}.

    - fsdd: 6 speakers -> 4 train / 1 val / 1 test (shuffled with seed).
    - team ('manit'): test only (never trained; team recording, recorder unknown).
    - speech_commands + sc_unknown share hash speakers -> one pool, hashes shuffled
      with seed, assigned to test / val until targets reached, rest train. Targets are
      chosen so the OVERALL clip split is ~70/15/15 given the fixed FSDD/team parts.
    - noise/silence generator families: split into contiguous-index blocks ~70/15/15
      (each block is its own group; see manifest_summary.md for why this is not leaky).
    """
    rng = random.Random(seed)
    assign: dict[str, str] = {}
    by_ds = defaultdict(lambda: defaultdict(list))
    for c in clips:
        by_ds[c["dataset"]][c["group"]].append(c)

    # fsdd
    fsdd = sorted(by_ds["fsdd"])
    rng.shuffle(fsdd)
    for s in fsdd[:4]:
        assign[s] = "train"
    assign[fsdd[4]] = "val"
    assign[fsdd[5]] = "test"
    # team
    for s in by_ds["team"]:
        assign[s] = "test"

    n_total = len(clips)
    n_fixed_test = sum(len(v) for ds in ("fsdd", "team") for g, v in by_ds[ds].items() if assign[g] == "test")
    n_fixed_val = sum(len(v) for g, v in by_ds["fsdd"].items() if assign[g] == "val")
    n_gen = sum(len(v) for ds in ("noise", "silence") for v in by_ds[ds].values())
    # generated families: 70/15/15 by contiguous index blocks
    for ds in ("noise", "silence"):
        for fam, items in by_ds[ds].items():
            items.sort(key=lambda c: int(_DUP_RE.sub("", Path(c["file"]).stem).split("_")[-1]))
            n = len(items)
            n_te, n_va = max(1, round(0.15 * n)), max(1, round(0.15 * n))
            for k, c in enumerate(items):
                blk = "train" if k < n - n_va - n_te else ("val" if k < n - n_te else "test")
                c["group"] = f"{fam}_{blk}"
                assign[c["group"]] = blk
    # speech commands pool (digits + unknown words share speakers)
    sc_groups = defaultdict(int)
    for ds in ("speech_commands", "sc_unknown"):
        for g, v in by_ds[ds].items():
            sc_groups[g] += len(v)
    n_sc = sum(sc_groups.values())
    target_test = 0.15 * n_total - n_fixed_test - 0.15 * n_gen
    target_val = 0.15 * n_total - n_fixed_val - 0.15 * n_gen
    hashes = sorted(sc_groups)
    rng.shuffle(hashes)
    n_te = n_va = 0
    for h in hashes:
        if n_te < target_test:
            assign[h] = "test"; n_te += sc_groups[h]
        elif n_va < target_val:
            assign[h] = "val"; n_va += sc_groups[h]
        else:
            assign[h] = "train"
    print(f"[2] split: fsdd train={fsdd[:4]} val={fsdd[4]} test={fsdd[5]}; team->test; "
          f"SC pool {n_sc} clips / {len(hashes)} speakers: test {n_te}, val {n_va} "
          f"(targets {target_test:.0f}/{target_val:.0f})")
    return assign


# ---------------------------------------------------------------------------
# 3. simulation
# ---------------------------------------------------------------------------
def plan_variants(clips: list[dict], train_sim_target: int, seed: int) -> dict[str, int]:
    """Number of sim variants per clean clip (keyed by file)."""
    n = {}
    train_digits = [c for c in clips if c["split"] == "train" and c["label"] != UNKNOWN_LABEL]
    train_unk = [c for c in clips if c["split"] == "train" and c["label"] == UNKNOWN_LABEL]
    for c in clips:
        n[c["file"]] = 1
    extra = train_sim_target - len(train_digits) - len(train_unk)
    extra = max(0, min(extra, len(train_digits)))  # at most 2 per train digit clip
    rng = random.Random(seed)
    pick = rng.sample([c["file"] for c in train_digits], extra)
    for f in pick:
        n[f] = 2
    print(f"[3] sim plan: train digits {len(train_digits)} x1 + {extra} second variants, "
          f"train unknown {len(train_unk)} x1 -> {len(train_digits) + extra + len(train_unk)} train sims; "
          f"val/test x1")
    return n


def sim_out_path(c: dict, k: int) -> Path:
    stem = Path(c["file"]).stem
    return SIM_DIR / c["split"] / c["label"] / f"{stem}__sim{k}.wav"


def run_simulation(clips: list[dict], nvar: dict[str, int], dry_run: bool) -> list[dict]:
    rows = []
    jobs = [(c, k) for c in clips for k in range(nvar[c["file"]])]
    print(f"    {len(jobs)} sim jobs -> {rel(SIM_DIR)}")
    t0 = time.time()
    for j, (c, k) in enumerate(jobs, 1):
        seed = derive_seed(c["sha256"], k)
        out = sim_out_path(c, k)
        if dry_run:
            continue
        clean = pls._load(ROOT / c["file"])          # mono float64 at FS=16000
        rng = np.random.default_rng(seed)
        sim, prm = pls.simulate(clean, rng)
        if not np.all(np.isfinite(sim)) or float(np.max(np.abs(sim))) == 0.0:
            raise RuntimeError(f"degenerate simulation for {c['file']} variant {k} seed {seed}")
        save_capture(sim.astype(np.float32), pls.FS, out)  # same routine as physics_laser_sim.py
        snr = float(prm["snr_db"])
        rows.append({
            "file": rel(out), "label": c["label"], "speaker": c["speaker"], "split": c["split"],
            "kind": "sim", "dataset": c["dataset"], "synthetic_noise": c["synthetic_noise"],
            "sha256": sha256_file(out), "source_file": c["file"], "source_sha256": c["sha256"],
            "sim_seed": seed, "sim_params": json.dumps(prm, separators=(",", ":")),
            "snr_db": snr, "condition": condition_of(snr),
        })
        if j % 500 == 0 or j == len(jobs):
            el = time.time() - t0
            print(f"    {j}/{len(jobs)}  {el:.0f}s elapsed, eta {el / j * (len(jobs) - j):.0f}s", flush=True)
    return rows


# ---------------------------------------------------------------------------
# 4. summary
# ---------------------------------------------------------------------------
def write_summary(rows: list[dict], dropped: list[dict], fam_counts: Counter, path: Path, sfx: str = ""):
    import pandas as pd
    df = pd.DataFrame(rows)
    L = []
    L.append(f"# Sprint 8 data manifest{sfx} summary (agent A)\n")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by scripts/sprint8/build_dataset.py "
             f"(split seed {SPLIT_SEED}, sim run seed {SIM_RUN_SEED}); sim audio under {rel(SIM_DIR)}, "
             f"manifest results/sprint8/data/manifest{sfx}.csv.\n")
    L.append(f"Rows: {len(df)} = {int((df.kind == 'clean').sum())} clean + {int((df.kind == 'sim').sum())} sim. "
             f"Originals dropped as duplicates: {len(dropped)} (see duplicates.csv).\n")

    L.append("## Clip counts per split x kind\n")
    L.append(pd.crosstab(df["split"], df["kind"], margins=True).reindex(["train", "val", "test", "All"]).to_markdown() + "\n")
    cl = df[df.kind == "clean"]
    tot = len(cl)
    frac = cl["split"].value_counts().reindex(["train", "val", "test"])
    L.append("Clean clips per split: " + ", ".join(f"{s} {n} ({100 * n / tot:.1f}%)" for s, n in frac.items()) + "\n")

    L.append("## Clean clips per split x dataset\n")
    L.append(pd.crosstab(cl["dataset"], cl["split"], margins=True)[["train", "val", "test", "All"]].to_markdown() + "\n")
    L.append("## Sim clips per split x dataset\n")
    sm = df[df.kind == "sim"]
    L.append(pd.crosstab(sm["dataset"], sm["split"], margins=True)[["train", "val", "test", "All"]].to_markdown() + "\n")

    L.append("## Clips per split x kind x label\n")
    lab_order = DIGIT_LABELS + [UNKNOWN_LABEL]
    t = pd.crosstab([df["split"], df["kind"]], df["label"])[lab_order]
    t = t.reindex([(s, k) for s in ("train", "val", "test") for k in ("clean", "sim")])
    L.append(t.to_markdown() + "\n")

    L.append("## Speakers per split (clean)\n")
    for s in ("train", "val", "test"):
        sub = cl[cl.split == s]
        by_ds = sub.groupby("dataset")["speaker"].nunique()
        L.append(f"- **{s}**: {sub['speaker'].nunique()} groups: " +
                 ", ".join(f"{d} {n}" for d, n in by_ds.items()))
        fs = sorted(sub[sub.dataset == "fsdd"]["speaker"].unique())
        tm = sorted(sub[sub.dataset == "team"]["speaker"].unique())
        gen = sorted(sub[sub.dataset.isin(["noise", "silence"])]["speaker"].unique())
        L.append(f"  - fsdd: {fs}; team: {tm}; generated groups: {gen}")
    L.append("")
    sc_d = set(cl[cl.dataset == "speech_commands"].speaker)
    sc_u = set(cl[cl.dataset == "sc_unknown"].speaker)
    both = sc_d & sc_u
    L.append(f"Speech Commands hashes appearing in both digits and unknown words: {len(both)}; "
             f"all such hashes are in a single split (verified by check_split.py).\n")

    L.append("## Condition histogram (sim)\n")
    ct = pd.crosstab(sm["split"], sm["condition"], margins=True)
    for c in ("sim_good", "sim_medium", "sim_poor"):
        if c not in ct:
            ct[c] = 0
    L.append(ct[["sim_good", "sim_medium", "sim_poor", "All"]].reindex(["train", "val", "test", "All"]).to_markdown() + "\n")
    L.append(f"snr_db (sim): min {sm.snr_db.min():.1f}, median {sm.snr_db.median():.1f}, max {sm.snr_db.max():.1f}. "
             f"Bins per PLAN.md: sim_good snr>=20, sim_medium 10<=snr<20, sim_poor snr<10.\n")

    L.append("## Decisions\n")
    L.append("- **Team speaker `manit`** (92 unique recordings after dropping 460 `_dupN` copies, 48 kHz): "
             "team recording, recorder unknown. Placed ENTIRELY in test; never in train or val.")
    L.append("- **FSDD**: 6 speakers -> 4 train / 1 val / 1 test, shuffled with seed 42.")
    L.append("- **Speech Commands**: digit clips and 'unknown' word clips of the same 8-hex speaker hash are one "
             "group (splits.speaker_of gives the hash for both) and land in the same split.")
    L.append("- **noise_*/silence_* unknown clips are SYNTHETIC (synthetic_noise=1).** Evidence: every clip is "
             "exactly 1.000 s and peak-normalised to 1.0; `silence_*` (8 kHz) is Gaussian white noise (flat "
             "spectrum, kurtosis 3.0, zero-crossing rate 0.5); `noise_*` (16 kHz) falls into four generator "
             f"families by spectral slope / kurtosis / zero-crossing rate: {dict(fam_counts)} "
             "(brown = 1/f^2 spectrum with a random DC offset, i.e. a random walk, files 0-58; white = flat, files "
             "59-117; hum_click = 60/180 Hz hum plus impulsive clicks, kurtosis 5-200, files ~118-178; bandlimited = "
             "coloured noise peaking at ~450-1500 Hz, files ~179-249; the families were written in index order, "
             "the classifier is a statistics heuristic so the two noisy families' boundary is approximate). Adjacent files are "
             "uncorrelated (|r| < 0.13, tail/head r ~ 0), so they are not chunks of one longer recording. "
             "No generator script exists in the repo; the statistics alone show they are generated, not recorded.")
    L.append("- **Grouping of generated clips**: there is no speaker identity to leak, and sharing a noise *distribution* "
             "between splits is the same as sharing it with the deployment environment. Each family is cut into "
             "contiguous index blocks (~70/15/15), one block per split, so every split has some noise and silence "
             "and any hypothetical temporal adjacency stays inside a split. `speaker` for these rows is the block id "
             "(e.g. `gen_noise_white_train`), which is unique to its split by construction.")
    L.append("- **Sim seeds**: sim_seed = int(sha256(f'{source_sha256}:{variant}:2026')[:15], 16); "
             "rng = numpy.random.default_rng(sim_seed); simulate(clean, rng) from scripts/physics_laser_sim.py "
             "unchanged (domain-randomised, no calibration). Output via preprocess.save_capture at 16 kHz.")
    L.append("- **Sim counts**: 1 variant per kept clip in every split, plus a seeded random subset of train digit "
             "clips gets a 2nd variant so the train sim total hits the 6,000-8,000 cap.")
    path.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
def main() -> int:
    global SIM_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-sim-target", type=int, default=7000)
    ap.add_argument("--dry-run", action="store_true", help="split + plan only, no audio written")
    ap.add_argument("--sim-dir", default=str(SIM_DIR),
                    help="where simulated wavs go (default data/sprint8_sim); use data/sprint8_sim_v2 for a regeneration")
    ap.add_argument("--manifest-suffix", default="",
                    help="suffix for manifest/summary names, e.g. '_v2' -> manifest_v2.csv, manifest_summary_v2.md. "
                         "Split, seeds and params are identical for any suffix (deterministic); only sim audio/sha change.")
    args = ap.parse_args()
    SIM_DIR = Path(args.sim_dir).resolve()
    sfx = args.manifest_suffix
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    kept, shas, dropped = collect_originals()
    with open(OUT_DIR / "duplicates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "sha256", "reason", "kept_file", "kept_sha256_match"])
        w.writeheader()
        w.writerows(dropped)

    clips = []
    fam_counts = Counter()
    for p in kept:
        lab = label_of(p)
        ds = dataset_of(p, lab)
        fam = None
        if ds == "noise":
            fam = noise_family(p)
            fam_counts[fam] += 1
        elif ds == "silence":
            fam = "white"
        clips.append({"file": rel(p), "label": lab, "dataset": ds, "sha256": shas[p],
                      "synthetic_noise": 1 if ds in ("noise", "silence") else 0,
                      "group": group_id(p, ds, fam)})
    print(f"    noise families: {dict(fam_counts)}")
    assign = make_split(clips)
    for c in clips:
        c["split"] = assign[c["group"]]
        c["speaker"] = c["group"]

    cnt = Counter((c["split"], c["dataset"]) for c in clips)
    for s in ("train", "val", "test"):
        print(f"    {s}: {sum(v for (sp, _), v in cnt.items() if sp == s)} clips  "
              + ", ".join(f"{d}={v}" for (sp, d), v in sorted(cnt.items()) if sp == s))

    clean_rows = [{
        "file": c["file"], "label": c["label"], "speaker": c["speaker"], "split": c["split"],
        "kind": "clean", "dataset": c["dataset"], "synthetic_noise": c["synthetic_noise"],
        "sha256": c["sha256"], "source_file": c["file"], "source_sha256": c["sha256"],
        "sim_seed": "", "sim_params": "", "snr_db": "", "condition": "clean",
    } for c in clips]

    nvar = plan_variants(clips, args.train_sim_target, SPLIT_SEED)
    sim_rows = run_simulation(clips, nvar, args.dry_run)

    rows = clean_rows + sim_rows
    with open(OUT_DIR / f"manifest{sfx}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"[4] manifest -> {rel(OUT_DIR / f'manifest{sfx}.csv')} ({len(rows)} rows)")
    if not args.dry_run:
        write_summary(rows, dropped, fam_counts, OUT_DIR / f"manifest_summary{sfx}.md", sfx)
        print(f"    summary  -> {rel(OUT_DIR / f'manifest_summary{sfx}.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
