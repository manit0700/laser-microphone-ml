#!/bin/bash
# kaggle_push.sh
# ==============
# Upload (or update) the pipeline CODE to Kaggle as a private Dataset, so the
# Kaggle GPU notebook (notebooks/kaggle_train.ipynb) can import src/ and train.
#
# WHAT IT UPLOADS
#   Only the src/*.py code (small). It does NOT upload data or the trained model
#   - the notebook downloads FSDD itself and writes outputs to /kaggle/working.
#
# PREREQUISITES
#   pip install kaggle
#   ~/.kaggle/kaggle.json  (your API token from kaggle.com -> Account -> Create New Token)
#
# USAGE
#   bash scripts/kaggle_push.sh            # first run creates the dataset; later runs update it
#
# After it runs, the dataset appears at:
#   https://www.kaggle.com/datasets/<your-username>/laser-microphone-ml

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SLUG="laser-microphone-ml"

# Read the Kaggle username from the token so the dataset id is correct.
USERNAME="$(python -c "import json,os; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")"
echo "Kaggle user: $USERNAME"

# Stage just the code in a temp folder (Kaggle uploads a folder's contents).
STAGE="$(mktemp -d)"
cp "$PROJECT_ROOT"/src/*.py "$STAGE"/

# Dataset metadata Kaggle requires.
cat > "$STAGE/dataset-metadata.json" <<JSON
{
  "title": "laser-microphone-ml",
  "id": "$USERNAME/$SLUG",
  "licenses": [{"name": "CC0-1.0"}]
}
JSON

echo "Staged $(ls "$STAGE"/*.py | wc -l | tr -d ' ') code files in $STAGE"

# Create on first run; version (update) on subsequent runs.
if kaggle datasets status "$USERNAME/$SLUG" >/dev/null 2>&1; then
  echo "Dataset exists -> pushing a new version..."
  kaggle datasets version -p "$STAGE" -m "update $(date '+%Y-%m-%d %H:%M')" --dir-mode zip
else
  echo "Creating new private dataset..."
  kaggle datasets create -p "$STAGE" --dir-mode zip
fi

echo ""
echo "Done. Open: https://www.kaggle.com/datasets/$USERNAME/$SLUG"
echo "In your Kaggle notebook: Add Input -> this dataset, set GPU + Internet, run kaggle_train.ipynb."
rm -rf "$STAGE"
