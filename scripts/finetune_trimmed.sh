#!/usr/bin/env bash
# Fine-tune the laser CNN on the TRIMMED copy made by trim_laser_clips.py.
#
# Same recipe as the Nov 11 team fine-tune (same speakers, base model, band-pass,
# enhance, 15 epochs, same seed and held-out clips); only the data folder differs.
#
#   bash scripts/finetune_trimmed.sh          dry run: prints accuracy, saves nothing
#   bash scripts/finetune_trimmed.sh save     saves models/best_model_cnn_laser_team_trim.pt
#                                             (a NEW file; the dashboard does not load it)
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
LIVE="${LMML_LIVE:-$HOME/laser_microphone_ml}"
DATA="$HERE/data/raw_trimmed"
BASE="models/backups/best_model_cnn_laser_oct13.pt"
OUT="models/best_model_cnn_laser_team_trim.pt"

[ -d "$DATA" ] || { echo "ERROR: $DATA not found. Run: python3 scripts/trim_laser_clips.py"; exit 2; }
cd "$LIVE"
[ -f "$BASE" ] || { echo "ERROR: base model $LIVE/$BASE not found"; exit 2; }

if [ "${1:-}" = "save" ]; then
  [ -e "$OUT" ] && { echo "ERROR: $LIVE/$OUT already exists. Not overwriting."; exit 3; }
  SAVE=(--out "$OUT")
else
  SAVE=(--no-save)
fi

python3 scripts/finetune.py \
  --speakers ali,brooke,brooke2,john,michael,laserali,laserjohn,lasermichaeltrim \
  --model cnn --only-tag laser --bandpass 200,3800 --enhance --epochs 15 \
  --base "$BASE" --data-dir "$DATA" "${SAVE[@]}"
