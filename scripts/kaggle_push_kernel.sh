#!/bin/bash
# kaggle_push_kernel.sh
# =====================
# Push notebooks/kaggle_train.ipynb to Kaggle as a runnable Kernel (Notebook),
# pre-wired with GPU + Internet + your code dataset attached. After this you can
# just open it on Kaggle and click "Run All" - no manual upload/settings needed.
#
# PREREQUISITES
#   pip install kaggle ; ~/.kaggle/kaggle.json present
#   The code dataset must already exist (run scripts/kaggle_push.sh first).
#
# USAGE
#   bash scripts/kaggle_push_kernel.sh     # first run creates it; reruns update it

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KERNEL_SLUG="laser-microphone-train"
DATASET_SLUG="laser-microphone-ml"

USERNAME="$(python -c "import json,os; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")"
echo "Kaggle user: $USERNAME"

STAGE="$(mktemp -d)"
cp "$PROJECT_ROOT/notebooks/kaggle_train.ipynb" "$STAGE/kaggle_train.ipynb"
# Optional: KAGGLE_MODEL=lstm bash scripts/kaggle_push_kernel.sh  (notebook default is cnn)
if [ -n "$KAGGLE_MODEL" ]; then
  sed "s/MODEL = 'cnn'/MODEL = '$KAGGLE_MODEL'/" "$STAGE/kaggle_train.ipynb" > "$STAGE/kt.tmp" \
    && mv "$STAGE/kt.tmp" "$STAGE/kaggle_train.ipynb"
  grep -q "MODEL = '$KAGGLE_MODEL'" "$STAGE/kaggle_train.ipynb" || { echo "ERROR: could not set MODEL"; exit 1; }
fi

# Optional run settings (set any of these in front of the command):
#   KAGGLE_EPOCHS=80          max epochs per run (early stopping may end sooner)
#   KAGGLE_DROPOUTS=0.4       dropout value(s) tried in MODE=sweep, comma-separated (e.g. 0.3,0.4)
#   KAGGLE_DROPOUT=0.4        dropout for MODE=train
#   KAGGLE_PATIENCE=10        early-stopping patience in epochs (0 = off)
#   KAGGLE_FOLDS=5            number of speaker-grouped folds for MODE=cv
#   KAGGLE_EXTRA="--deltas --label-smoothing 0.1"   extra train.py flags for MODE=train / sweep / cv
# e.g.  KAGGLE_MODE=sweep KAGGLE_MODEL=cnn KAGGLE_DROPOUTS=0.4 KAGGLE_EPOCHS=80 bash scripts/kaggle_push_kernel.sh
if [ -n "$KAGGLE_EPOCHS$KAGGLE_DROPOUTS$KAGGLE_DROPOUT$KAGGLE_PATIENCE$KAGGLE_FOLDS$KAGGLE_EXTRA" ]; then
  NB="$STAGE/kaggle_train.ipynb" python - <<'PYEDIT' || { echo "ERROR: could not apply run settings"; exit 1; }
import json, os, re
path = os.environ["NB"]
nb = json.load(open(path))
subs = []
if os.environ.get("KAGGLE_EPOCHS"):
    subs.append((r"^MAX_EPOCHS = \S+", "MAX_EPOCHS = %d" % int(os.environ["KAGGLE_EPOCHS"])))
if os.environ.get("KAGGLE_DROPOUTS"):
    vals = [float(v) for v in os.environ["KAGGLE_DROPOUTS"].split(",") if v.strip()]
    assert vals and all(0.0 <= v < 1.0 for v in vals), "dropouts must be between 0 and 1"
    subs.append((r"^SWEEP_DROPOUTS = \[[^\]]*\]", "SWEEP_DROPOUTS = [%s]" % ", ".join("%g" % v for v in vals)))
if os.environ.get("KAGGLE_DROPOUT"):
    v = float(os.environ["KAGGLE_DROPOUT"]); assert 0.0 <= v < 1.0
    subs.append((r"^DROPOUT = \S+", "DROPOUT = %g" % v))
if os.environ.get("KAGGLE_PATIENCE"):
    subs.append((r"^PATIENCE = \S+", "PATIENCE = %d" % int(os.environ["KAGGLE_PATIENCE"])))
if os.environ.get("KAGGLE_FOLDS"):
    subs.append((r"^CV_FOLDS = \S+", "CV_FOLDS = %d" % int(os.environ["KAGGLE_FOLDS"])))
if os.environ.get("KAGGLE_EXTRA"):
    import shlex
    extra = shlex.split(os.environ["KAGGLE_EXTRA"])
    subs.append((r"^EXTRA_TRAIN_ARGS = .*$", "EXTRA_TRAIN_ARGS = %r" % extra))
for pattern, repl in subs:
    hits = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        new = []
        for line in cell["source"]:
            line2, n = re.subn(pattern, repl, line)
            hits += n
            new.append(line2)
        cell["source"] = new
    assert hits == 1, "setting %r matched %d lines (expected 1)" % (pattern, hits)
    print("  set:", repl)
json.dump(nb, open(path, "w"), indent=1, ensure_ascii=False)
PYEDIT
fi

# Inputs: the code dataset + Speech Commands. In evaluate mode the saved models are attached too.
SOURCES="\"$USERNAME/$DATASET_SLUG\", \"yashdogra/speech-commands\""
# Optional: KAGGLE_MODE=evaluate bash scripts/kaggle_push_kernel.sh
#   scores the saved lstm + cnn + ensemble on the held-out test split instead of training
#   (needs the private dataset $USERNAME/laser-microphone-models: best_model.pt,
#   best_model_cnn.pt, test_indices.json).
if [ -n "$KAGGLE_MODE" ]; then
  sed "s/MODE = 'train'/MODE = '$KAGGLE_MODE'/" "$STAGE/kaggle_train.ipynb" > "$STAGE/kt.tmp" \
    && mv "$STAGE/kt.tmp" "$STAGE/kaggle_train.ipynb"
  grep -q "MODE = '$KAGGLE_MODE'" "$STAGE/kaggle_train.ipynb" || { echo "ERROR: could not set MODE"; exit 1; }
  if [ "$KAGGLE_MODE" = "evaluate" ]; then
    SOURCES="$SOURCES, \"$USERNAME/laser-microphone-models\""
  fi
fi

# Kernel metadata Kaggle requires. We attach the code dataset and request GPU +
# Internet so the notebook runs end-to-end unattended.
cat > "$STAGE/kernel-metadata.json" <<JSON
{
  "id": "$USERNAME/$KERNEL_SLUG",
  "title": "laser-microphone-train",
  "code_file": "kaggle_train.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [$SOURCES],
  "competition_sources": [],
  "kernel_sources": []
}
JSON

echo "Pushing kernel..."
# Optional: KAGGLE_ACCELERATOR=NvidiaTeslaT4 bash scripts/kaggle_push_kernel.sh
# The default "Gpu" can hand out a P100, which the preinstalled PyTorch has no
# kernels for ("no kernel image is available") -> training silently falls back to CPU.
if [ -n "$KAGGLE_ACCELERATOR" ]; then
  kaggle kernels push -p "$STAGE" --accelerator "$KAGGLE_ACCELERATOR"
else
  kaggle kernels push -p "$STAGE"
fi

echo ""
echo "Done. Open: https://www.kaggle.com/code/$USERNAME/$KERNEL_SLUG"
echo "On that page: click 'Edit' then 'Run All' (GPU + Internet + dataset are preset)."
rm -rf "$STAGE"
