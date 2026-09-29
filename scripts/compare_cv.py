"""
compare_cv.py
=============
Compare training configurations (e.g. baseline vs improved) with speaker-grouped
K-fold cross-validation -- choosing the winner on VALIDATION data only and
looking at the TEST folds exactly once, at the very end.

WHY THIS ORDER MATTERS
----------------------
If you compare configurations by their test accuracy, pick the better one, and
then report that same test accuracy, the test set has been used for tuning and
the reported number is optimistic. So:

  Phase 1  TRAIN   every config on every fold. Inside each run the fold's
                   validation speakers drive early stopping, temperature
                   calibration and the 'unknown' threshold. No test data is read.
  Phase 2  SELECT  the winner by mean VALIDATION balanced accuracy across folds.
                   The decision is written to selection.json BEFORE any test score
                   exists.
  Phase 3  TEST    score every config's fold models on their held-out test folds,
                   once (with each model's own calibration + threshold). Refuses to
                   run again unless --force, so the test set can't be re-tuned on.

USAGE (project root; this is 2 x K full trainings -- run it on Kaggle)
-----
  python3 scripts/compare_cv.py --model cnn --folds 5 \\
      --config baseline="--scheduler none --noise-min-snr 8" \\
      --config improved="--specaugment --scheduler plateau --noise-min-snr 0 --label-smoothing 0.1" \\
      -- --augment --unknown-class --dropout 0.4 --epochs 60 --patience 8

  Everything after "--" goes to src/train.py for every run.
  Inside a --config, LMML_*=value tokens are environment settings for that config, e.g.
      --config sr8k="LMML_SAMPLE_RATE=8000"  --config sr16k="LMML_SAMPLE_RATE=16000"
  Re-running resumes: finished fold trainings are skipped.

OUTPUT (--out, default <results>/cvcompare_<model>/)
  <config>/fold_<k>/...        each run's model, curves, split
  selection.json               the val-based decision (written before testing)
  test_results.csv             per config x fold test metrics
  summary.txt / summary.json   mean +/- std, and the paired difference per fold
"""

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" if (ROOT / "src" / "train.py").exists() else Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

METRICS = (("balanced_acc", "Balanced accuracy (11 classes)"), ("overall_acc", "Overall accuracy"),
           ("digit_acc", "Digits correct"), ("digit_rejected", "Digits wrongly rejected"),
           ("unknown_rejection", "Unknown rejected"))


def _ms(vals):
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan"), float("nan")
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def _ckpt_name(model):
    return "best_model.pt" if model == "lstm" else f"best_model_{model}.pt"


def evaluate_fold(run_dir: Path, model: str) -> dict:
    """Score one trained fold model on ITS test fold (calibrated + tuned threshold)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    from calibration import apply_threshold, decision_metrics
    from config import BATCH_SIZE, CONFIDENCE_THRESHOLD, DEVICE
    from dataset import SpokenDigitDataset
    from model import model_from_checkpoint

    ckpt = torch.load(run_dir / "models" / _ckpt_name(model), map_location=DEVICE)
    labels = ckpt["labels"]
    test_ix = json.load(open(run_dir / "models" / "test_indices.json"))["test_indices"]
    ds = SpokenDigitDataset(feature=ckpt["feature"], include_unknown="unknown" in labels,
                            cache_in_memory=False)
    loader = DataLoader(Subset(ds, test_ix), batch_size=BATCH_SIZE, shuffle=False)
    net = model_from_checkpoint(ckpt).to(DEVICE).eval()
    t = float(ckpt.get("temperature", 1.0))
    thr = float(ckpt.get("threshold") if ckpt.get("threshold") is not None else CONFIDENCE_THRESHOLD)
    probs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(F.softmax(net(x.to(DEVICE)) / t, dim=1).cpu())
            ys.append(y)
    probs, ys = torch.cat(probs), torch.cat(ys)
    m = decision_metrics(apply_threshold(probs, thr, labels.index("unknown")), ys, labels.index("unknown"))
    m.update(n_test=len(ys), threshold=thr, temperature=t)
    return m


def evaluate_fold_subprocess(run_dir: Path, model: str, extra_env: dict) -> dict:
    """Run evaluate_fold in a fresh process so each config's sample rate (read by
    config.py at import) is applied -- 8 kHz and 16 kHz models can't share one process."""
    code = ("import json,sys; sys.path.insert(0, %r); sys.argv=['x']; import compare_cv as c; "
            "print('__RESULT__' + json.dumps(c.evaluate_fold(__import__('pathlib').Path(%r), %r)))"
            % (str(Path(__file__).resolve().parent), str(run_dir), model))
    env = dict(os.environ, LMML_OUTPUT_DIR=str(run_dir), PYTHONPATH=str(SRC), **extra_env)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    line = next((l for l in out.stdout.splitlines() if l.startswith("__RESULT__")), None)
    if out.returncode != 0 or line is None:
        raise RuntimeError(f"evaluation failed for {run_dir}:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
    return json.loads(line[len("__RESULT__"):])


def main() -> int:
    argv = sys.argv[1:]
    common = []
    if "--" in argv:
        i = argv.index("--")
        argv, common = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description="Compare configs with speaker-grouped K-fold CV.")
    ap.add_argument("--model", choices=["lstm", "cnn"], required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--config", action="append", required=True, metavar='NAME="train.py flags"',
                    help='a configuration to compare, e.g. improved="--specaugment --scheduler plateau"')
    ap.add_argument("--out", default=None)
    ap.add_argument("--test-winner-only", action="store_true",
                    help="in phase 3, score only the selected config (default: all, after the decision is locked)")
    ap.add_argument("--force", action="store_true", help="allow phase 3 to run again (normally refused)")
    args = ap.parse_args(argv)

    configs, config_env = {}, {}
    for c in args.config:
        name, _, flags = c.partition("=")
        toks = shlex.split(flags)
        # Tokens like LMML_SAMPLE_RATE=16000 are environment settings for that config
        # (they must be set before train.py imports config), everything else is a flag.
        config_env[name.strip()] = dict(t.split("=", 1) for t in toks if t.startswith("LMML_") and "=" in t)
        configs[name.strip()] = [t for t in toks if not (t.startswith("LMML_") and "=" in t)]
    from config import RESULTS_DIR  # noqa: E402
    out = Path(args.out) if args.out else Path(RESULTS_DIR) / f"cvcompare_{args.model}"
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- Phase 1: train (validation only) ----------------
    runs = {name: [] for name in configs}
    for k in range(args.folds):
        for name, flags in configs.items():
            run_dir = out / name / f"fold_{k}"
            run_json = sorted((run_dir / "results" / "reports").glob(f"training_run_{args.model}_*.json"))
            if not (run_json and (run_dir / "models" / _ckpt_name(args.model)).exists()):
                print("\n" + "#" * 72 + f"\n#  PHASE 1  {name}  fold {k + 1}/{args.folds}\n" + "#" * 72, flush=True)
                env = dict(os.environ, LMML_OUTPUT_DIR=str(run_dir), **config_env[name])
                subprocess.run([sys.executable, str(SRC / "train.py"), "--model", args.model,
                                "--split", f"kfold:{k}/{args.folds}", *common, *flags], check=True, env=env)
                run_json = sorted((run_dir / "results" / "reports").glob(f"training_run_{args.model}_*.json"))
            else:
                print(f"PHASE 1  {name} fold {k + 1}: already trained, skipping")
            r = json.load(open(run_json[-1]))
            runs[name].append({"fold": k, "val_balanced_acc": r.get("val_balanced_acc", float("nan")),
                               "val_loss": r.get("best_val_loss"), "threshold": r.get("threshold"),
                               "epochs": r.get("epochs_run"), "run_dir": str(run_dir)})

    # ---------------- Phase 2: select on validation ----------------
    sel_path = out / "selection.json"
    val_summary = {}
    for name, rs in runs.items():
        m, sd = _ms([r["val_balanced_acc"] for r in rs])
        lm, _ = _ms([r["val_loss"] for r in rs if r["val_loss"] is not None])
        val_summary[name] = {"val_balanced_acc_mean": m, "val_balanced_acc_std": sd, "val_loss_mean": lm}
    winner = max(val_summary, key=lambda n: (val_summary[n]["val_balanced_acc_mean"],
                                             -val_summary[n]["val_loss_mean"]))
    if sel_path.exists() and json.load(open(sel_path)).get("winner") != winner and not args.force:
        print(f"ERROR: selection.json already chose a different winner. Refusing to overwrite (--force).")
        return 1
    if not sel_path.exists() or args.force:
        sel_path.write_text(json.dumps({"winner": winner, "criterion": "mean validation balanced accuracy",
                                        "validation": val_summary, "configs": {n: " ".join(f) for n, f in configs.items()},
                                        "decided_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "test_set_used": False}, indent=2))
    print("\n" + "=" * 72 + "\nPHASE 2  selection on VALIDATION data (test not yet touched)")
    for n, v in val_summary.items():
        print(f"  {n:<12} val balanced acc {v['val_balanced_acc_mean']:.2%} +/- {v['val_balanced_acc_std']:.2%}"
              f"   val loss {v['val_loss_mean']:.4f}" + ("   <- SELECTED" if n == winner else ""))

    # ---------------- Phase 3: test, once ----------------
    res_path = out / "test_results.csv"
    if res_path.exists() and not args.force:
        print(f"\nPhase 3 already done ({res_path}). The test set is scored once; pass --force to redo.")
        print((out / "summary.txt").read_text() if (out / "summary.txt").exists() else "")
        return 0
    to_test = [winner] if args.test_winner_only else list(configs)
    rows = []
    print("\n" + "=" * 72 + f"\nPHASE 3  final TEST evaluation (once): {', '.join(to_test)}")
    for name in to_test:
        for r in runs[name]:
            m = evaluate_fold_subprocess(Path(r["run_dir"]), args.model, config_env[name])
            rows.append({"config": name, "fold": r["fold"], **m})
            print(f"  {name:<12} fold {r['fold'] + 1}: balanced {m['balanced_acc']:.2%} | digits "
                  f"{m['digit_acc']:.2%} | unknown rejected {m['unknown_rejection']:.2%} | thr {m['threshold']:.2f}",
                  flush=True)
    import csv
    with open(res_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    sel = json.load(open(sel_path))
    sel["test_set_used"] = True
    sel_path.write_text(json.dumps(sel, indent=2))

    lines = [f"{args.model.upper()}  speaker-grouped {args.folds}-fold CV  |  selected on validation: {winner}",
             f"(test folds scored once, after the selection was saved; every speaker is tested exactly once)", ""]
    summary = {"winner": winner, "folds": args.folds, "configs": sel["configs"], "test": {}}
    header = f"  {'metric':<32}" + "".join(f"{n:>22}" for n in to_test)
    lines.append(header)
    for key, label in METRICS:
        cells = []
        for n in to_test:
            mm, sd = _ms([r[key] for r in rows if r["config"] == n])
            summary["test"].setdefault(n, {})[key] = {"mean": mm, "std": sd}
            cells.append(f"{mm:>13.2%} +/- {sd:.2%}")
        lines.append(f"  {label:<32}" + "".join(f"{c:>22}" for c in cells))
    if len(to_test) == 2:
        a, b = to_test
        diffs = []
        for k in range(args.folds):
            ra = next(r for r in rows if r["config"] == a and r["fold"] == k)
            rb = next(r for r in rows if r["config"] == b and r["fold"] == k)
            diffs.append(rb["balanced_acc"] - ra["balanced_acc"])
        dm, dsd = _ms(diffs)
        better = sum(d > 0 for d in diffs)
        summary["paired_diff_balanced_acc"] = {"mean": dm, "std": dsd, "folds_better": better}
        lines += ["", f"  {b} - {a} (balanced acc, same folds): {dm:+.2%} +/- {dsd:.2%}; "
                      f"{b} better on {better}/{args.folds} folds"]
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 72 + "\n" + text + "\n" + "=" * 72)
    print(f"Saved -> {out}/summary.txt, test_results.csv, selection.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
