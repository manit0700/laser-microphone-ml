#!/bin/bash
# kaggle_train_with_mod.sh
# ========================
# ONE COMMAND: train on Kaggle (GPU) on the usual data PLUS the modified Speech
# Commands digits in google_commands_0_9_mod (zero_mod/ ... nine_mod/).
#
# WHAT IT DOES
#   1. Uploads the modified clips as a PRIVATE Kaggle dataset
#      <you>/google-commands-0-9-mod (first run only, ~1.2 GB; later runs skip it).
#   2. Pushes the current pipeline code            (scripts/kaggle_push.sh).
#   3. Pushes the notebook, which starts the run   (scripts/kaggle_push_kernel.sh).
#
# THE RUN (defaults = the "improved" recipe of Exp 4, docs/EXPERIMENTS.md)
#   CNN, 8 kHz, dropout 0.4, up to 60 epochs, early stopping patience 8, SpecAugment,
#   ReduceLROnPlateau, noise down to 0 dB SNR, label smoothing 0.1, T4 GPU.
#   Speaker-grouped split: a clip and its modified copy always share a split, and
#   the test report scores original and modified clips separately.
#   Override any setting the usual way, e.g.  KAGGLE_MODEL=lstm KAGGLE_EPOCHS=40 bash scripts/kaggle_train_with_mod.sh
#
# PREREQUISITES
#   pip install kaggle ; ~/.kaggle/kaggle.json present
#
# USAGE
#   bash scripts/kaggle_train_with_mod.sh                          # clips in ~/Downloads/google_commands_0_9_mod
#   bash scripts/kaggle_train_with_mod.sh /path/to/google_commands_0_9_mod

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MOD_DIR="${1:-$HOME/Downloads/google_commands_0_9_mod}"
MOD_SLUG="google-commands-0-9-mod"
CODE_SLUG="laser-microphone-ml"
KERNEL_SLUG="laser-microphone-train"

command -v kaggle >/dev/null 2>&1 || { echo "ERROR: kaggle CLI not found. Run: pip install kaggle"; exit 1; }
[ -f "$HOME/.kaggle/kaggle.json" ] || { echo "ERROR: ~/.kaggle/kaggle.json not found (kaggle.com -> Account -> Create New Token)."; exit 1; }
USERNAME="$(python -c "import json,os; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")"
echo "Kaggle user: $USERNAME"

# Wait until Kaggle has finished processing a dataset:  wait_ready <id> <max minutes>
wait_ready() {
  local id="$1" tries=$(( ${2:-30} * 6 )) status=""
  for _ in $(seq 1 "$tries"); do
    status="$(kaggle datasets status "$id" 2>/dev/null | tr 'A-Z' 'a-z' | tr -d '[:space:]')" || true
    case "$status" in *ready*) echo "  $id: ready"; return 0 ;; esac
    sleep 10
  done
  echo "ERROR: $id is still not ready (last status: '${status:-none}'). Check https://www.kaggle.com/datasets/$id and rerun."
  return 1
}

# ---- 1) modified clips -> private Kaggle dataset (only if it is not there yet) ----
if kaggle datasets status "$USERNAME/$MOD_SLUG" >/dev/null 2>&1; then
  echo "[1/3] Modified clips are already on Kaggle ($USERNAME/$MOD_SLUG) - skipping the upload."
else
  [ -d "$MOD_DIR/zero_mod" ] && [ -d "$MOD_DIR/nine_mod" ] || {
    echo "ERROR: $MOD_DIR does not contain zero_mod/ ... nine_mod/. Pass the folder as the first argument."; exit 1; }
  echo "[1/3] Uploading $(find "$MOD_DIR" -name '*.wav' | wc -l | tr -d ' ') modified clips from $MOD_DIR (~1.2 GB, one zip per digit)..."
  # Kaggle reads the dataset id from this small file inside the folder.
  cat > "$MOD_DIR/dataset-metadata.json" <<JSON
{
  "title": "$MOD_SLUG",
  "id": "$USERNAME/$MOD_SLUG",
  "licenses": [{"name": "other"}]
}
JSON
  kaggle datasets create -p "$MOD_DIR" --dir-mode zip
fi
echo "  waiting for Kaggle to finish processing the clips..."
wait_ready "$USERNAME/$MOD_SLUG" 60

# ---- 2) current code ----
echo "[2/3] Pushing the pipeline code..."
bash "$PROJECT_ROOT/scripts/kaggle_push.sh"
sleep 20          # give Kaggle a moment to register the new version before asking for its status
wait_ready "$USERNAME/$CODE_SLUG" 15

# ---- 3) notebook: pushing it starts the run ----
echo "[3/3] Starting the training run..."
KAGGLE_ACCELERATOR="${KAGGLE_ACCELERATOR:-NvidiaTeslaT4}" \
KAGGLE_MODEL="${KAGGLE_MODEL:-cnn}" \
KAGGLE_DROPOUT="${KAGGLE_DROPOUT:-0.4}" \
KAGGLE_EPOCHS="${KAGGLE_EPOCHS:-60}" \
KAGGLE_PATIENCE="${KAGGLE_PATIENCE:-8}" \
KAGGLE_EXTRA="${KAGGLE_EXTRA:---specaugment --scheduler plateau --noise-min-snr 0 --label-smoothing 0.1}" \
KAGGLE_MOD_DATASET="$USERNAME/$MOD_SLUG" \
  bash "$PROJECT_ROOT/scripts/kaggle_push_kernel.sh"

OUT="kaggle_mod_${KAGGLE_MODEL:-cnn}"
echo ""
echo "Run started: https://www.kaggle.com/code/$USERNAME/$KERNEL_SLUG"
echo "Check progress:   kaggle kernels status $USERNAME/$KERNEL_SLUG"
echo "When 'complete':  kaggle kernels output $USERNAME/$KERNEL_SLUG -p $PROJECT_ROOT/$OUT"
echo "(The download goes to its own folder, so the shipped models in models/ are not overwritten.)"
