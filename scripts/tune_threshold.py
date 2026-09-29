"""
tune_threshold.py
=================
Tune the 'unknown' threshold of the model(s) the dashboard actually uses -- on the
VALIDATION split, never the test set -- and store it in the checkpoint(s).

train.py already tunes each single model's threshold after training. The live
dashboard, though, runs the LSTM + CNN *ensemble*, whose averaged probabilities
behave differently, so the ensemble needs its own threshold. Run this after both
models are trained (on the same split) or fine-tuned.

    python3 scripts/tune_threshold.py                  # ensemble (default) -> saves 'ensemble_threshold'
    python3 scripts/tune_threshold.py --model cnn      # one model -> saves 'threshold'
    python3 scripts/tune_threshold.py --dry-run        # show the result, save nothing

Criterion: BALANCED accuracy over the 11 classes (average per-class recall), so
rejecting non-digit sounds counts as much as recognising each digit.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from calibration import tune_threshold  # noqa: E402
from config import DIGIT_LABELS, model_checkpoint  # noqa: E402
from dataset import SpokenDigitDataset  # noqa: E402
from preprocess import load_audio  # noqa: E402
from train import TEST_INDICES_PATH  # noqa: E402
from utils import load_json  # noqa: E402
import predict as predict_mod  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Tune the 'unknown' threshold on validation data.")
    ap.add_argument("--model", choices=["ensemble", "lstm", "cnn"], default="ensemble")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N validation clips")
    args = ap.parse_args()

    split = load_json(TEST_INDICES_PATH)
    if "val_indices" not in split:
        print("No validation indices saved with these models (trained before this feature). Retrain first.")
        return 1
    val_ix = split["val_indices"][: args.limit] if args.limit else split["val_indices"]

    names = ["lstm", "cnn"] if args.model == "ensemble" else [args.model]
    ckpts = {m: torch.load(model_checkpoint(m), map_location="cpu") for m in names}
    if not all("unknown" in c.get("labels", []) for c in ckpts.values()):
        print("These models have no 'unknown' class, so there is nothing to tune against.")
        return 1

    ds = SpokenDigitDataset(include_unknown=True, cache_in_memory=False)
    labels_out = list(DIGIT_LABELS) + ["unknown"]
    probs, ys = [], []
    print(f"Scoring {len(val_ix)} validation clips with the {args.model} ...")
    for n, i in enumerate(val_ix, 1):
        path, y = ds.samples[i]
        wave = load_audio(path)
        if args.model == "ensemble":
            _, p = predict_mod.infer_ensemble(wave, threshold=0.0)
        else:
            _, p = predict_mod.infer(wave, threshold=0.0, model_type=args.model)
        probs.append([p.get(l, 0.0) for l in labels_out])
        ys.append(y)
        if n % 500 == 0:
            print(f"  {n}/{len(val_ix)}")
    probs, ys = torch.tensor(probs), torch.tensor(ys)

    unk = labels_out.index("unknown")
    t, m, m60 = tune_threshold(probs, ys, unk)
    print("\n" + "=" * 70)
    print(f"{args.model.upper()} 'unknown' threshold, tuned on {len(ys)} VALIDATION clips")
    print(f"{'':22}{'at 0.60':>10}{f'at {t:.2f}':>12}")
    for k, name in (("balanced_acc", "balanced accuracy"), ("digit_acc", "digits correct"),
                    ("digit_rejected", "digits rejected"), ("unknown_rejection", "unknown rejected")):
        print(f"  {name:<20}{m60[k]:>10.2%}{m[k]:>12.2%}")
    print("=" * 70)

    if args.dry_run:
        print("--dry-run: nothing saved.")
        return 0
    key = "ensemble_threshold" if args.model == "ensemble" else "threshold"
    for name, c in ckpts.items():
        c[key] = t
        torch.save(c, model_checkpoint(name))
        print(f"Saved {key}={t:.2f} into {Path(model_checkpoint(name)).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
