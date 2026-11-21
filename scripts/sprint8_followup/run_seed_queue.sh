#!/usr/bin/env bash
# Bounded, finite queue: update-matched clean-only controls and extra fine-tune seeds.
# Writes ONLY under results/sprint8_followup/. Never touches results/sprint8/models or models/.
# Update budget: fine-tune = 30 epochs x ceil(13391/32)=419 -> 12,570 updates.
# Control (clean replay only, 6,391 clips/epoch -> 200 updates) = 63 epochs -> ~12,600 updates; early stop disabled.
set -u
cd "$(dirname "$0")/../.."
PY=/home/claude/.venv-lmml/bin/python
M=results/sprint8/data/manifest_v2.csv
O=results/sprint8_followup
run_control() { s=$1; $PY scripts/sprint8/finetune_sim.py --manifest $M --no-sim --epochs 63 --patience 999 --seed $s \
   --out-ckpt $O/models/control_matched_s$s.pt --out-dir $O/train/control_matched_s$s; echo "DONE control s$s rc=$?"; }
run_ft() { s=$1; $PY scripts/sprint8/finetune_sim.py --manifest $M --epochs 30 --patience 4 --seed $s \
   --out-ckpt $O/models/ft_s$s.pt --out-dir $O/train/ft_s$s; echo "DONE finetune s$s rc=$?"; }
run_control 42
run_ft 43
run_control 43
run_ft 44
run_control 44
echo QUEUE_FINISHED
