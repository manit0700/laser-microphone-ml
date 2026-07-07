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
  "dataset_sources": ["$USERNAME/$DATASET_SLUG"],
  "competition_sources": [],
  "kernel_sources": []
}
JSON

echo "Pushing kernel..."
kaggle kernels push -p "$STAGE"

echo ""
echo "Done. Open: https://www.kaggle.com/code/$USERNAME/$KERNEL_SLUG"
echo "On that page: click 'Edit' then 'Run All' (GPU + Internet + dataset are preset)."
rm -rf "$STAGE"
