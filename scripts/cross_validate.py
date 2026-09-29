"""
cross_validate.py
=================
Speaker-grouped K-fold cross-validation: a mean +/- std accuracy for the report.

WHY
---
One train/test split gives one number, and with a few hundred test clips per
digit that number moves by a point or two depending on WHICH speakers happen to
land in the test set. K-fold repeats training K times; each time a different
1/K of the speakers is held out as the test set, so every clip is tested
exactly once by a model that never heard that speaker. Reporting the mean and
standard deviation over the K folds shows both the expected accuracy and how
stable it is.

USAGE (project root; each fold is a full training run -- use Kaggle for the full data)
-----
    python3 scripts/cross_validate.py --model cnn --folds 5 -- --augment --unknown-class --dropout 0.4
    python3 scripts/cross_validate.py --model lstm --folds 5 -- --augment --unknown-class --deltas

Everything after "--" is passed to src/train.py unchanged.

OUTPUT (in --out, default results/cv_<model>/)
    fold_<k>/...            each fold's model, curves and evaluation report
    cv_results.csv          one row per fold
    cv_summary.json / .txt  mean +/- std of overall, digit and unknown accuracy
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" if (ROOT / "src" / "train.py").exists() else Path(__file__).resolve().parent  # Kaggle: flat folder


def _metrics(report: dict) -> dict:
    cm, labels = report["confusion_matrix"], report["labels"]
    digits = [i for i, l in enumerate(labels) if l != "unknown"]
    d_total = sum(sum(cm[i]) for i in digits)
    out = {"overall_acc": report["overall_test_accuracy"],
           "digit_acc": sum(cm[i][i] for i in digits) / d_total if d_total else float("nan"),
           "n_test": report["num_test_samples"]}
    if "unknown" in labels:
        u = labels.index("unknown")
        out["unknown_rejection"] = cm[u][u] / sum(cm[u]) if sum(cm[u]) else float("nan")
    return out


def _mean_std(vals):
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan"), float("nan")
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def main() -> int:
    argv = sys.argv[1:]
    passthrough = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description="Speaker-grouped K-fold cross-validation.")
    ap.add_argument("--model", choices=["lstm", "cnn"], required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only-fold", type=int, default=None, help="run just this fold (0-based)")
    ap.add_argument("--out", default=None, help="output folder (default <results>/cv_<model>)")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(SRC))
    from config import RESULTS_DIR  # noqa: E402
    out_root = Path(args.out) if args.out else Path(RESULTS_DIR) / f"cv_{args.model}"
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    folds = [args.only_fold] if args.only_fold is not None else range(args.folds)
    for k in folds:
        fold_dir = out_root / f"fold_{k}"
        env = dict(os.environ, LMML_OUTPUT_DIR=str(fold_dir), LMML_SPLIT=f"kfold:{k}/{args.folds}")
        print("\n" + "#" * 70 + f"\n#  {args.model.upper()}  fold {k + 1}/{args.folds}\n" + "#" * 70, flush=True)
        subprocess.run([sys.executable, str(SRC / "train.py"), "--model", args.model,
                        "--split", f"kfold:{k}/{args.folds}", *passthrough], check=True, env=env)
        subprocess.run([sys.executable, str(SRC / "evaluate.py"), "--model", args.model],
                       check=True, env=env)
        rep = json.load(open(fold_dir / "results" / "reports" / f"evaluation_report_{args.model}.json"))
        runs = sorted((fold_dir / "results" / "reports").glob(f"training_run_{args.model}_*.json"))
        run = json.load(open(runs[-1])) if runs else {}
        row = {"fold": k, **_metrics(rep), "best_epoch": run.get("best_epoch"),
               "epochs_run": run.get("epochs_run"), "val_ece_after": run.get("val_ece_after"),
               "temperature": run.get("temperature")}
        rows.append(row)
        print(f"FOLD {k}: overall {row['overall_acc']:.4f} | digits {row['digit_acc']:.4f} | "
              f"unknown {row.get('unknown_rejection', float('nan')):.4f}", flush=True)

    import csv
    with open(out_root / "cv_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), restval="")
        w.writeheader()
        w.writerows(rows)

    summary = {"model": args.model, "folds": len(rows), "train_args": passthrough}
    lines = [f"{args.model.upper()} - speaker-grouped {args.folds}-fold cross-validation "
             f"({len(rows)} fold(s) run)", ""]
    for key, name in (("overall_acc", "Overall accuracy"), ("digit_acc", "Digit accuracy"),
                      ("unknown_rejection", "Unknown rejection"), ("val_ece_after", "Calibration error (ECE)")):
        m, sd = _mean_std([r.get(key, float("nan")) for r in rows if r.get(key) is not None])
        summary[key] = {"mean": m, "std": sd}
        fmt = (lambda v: f"{v * 100:.2f}%") if key != "val_ece_after" else (lambda v: f"{v:.4f}")
        lines.append(f"  {name:<24} {fmt(m)} +/- {fmt(sd)}")
    lines.append(f"\n  Test clips per fold: {[r['n_test'] for r in rows]} (every speaker tested once)")
    text = "\n".join(lines)
    (out_root / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    (out_root / "cv_summary.txt").write_text(text + "\n")
    print("\n" + "=" * 70 + "\n" + text + "\n" + "=" * 70)
    print(f"Saved -> {out_root}/cv_summary.txt, cv_results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
