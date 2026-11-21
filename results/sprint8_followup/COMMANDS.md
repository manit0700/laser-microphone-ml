# Exact commands run (Sprint 8 follow-up)

Environment: /home/claude/.venv-lmml (python 3.13, torch 2.15 CPU, 2 threads), project root /home/claude/lmml.
Checkpoints are read-only; outputs only under results/sprint8_followup/.

```bash
cd /home/claude/lmml
PY=/home/claude/.venv-lmml/bin/python

# U2 train/val overfitting diagnostics (eval mode, frozen T and threshold; caches logits under results/sprint8_followup/cache/)
$PY scripts/sprint8_followup/diagnose_overfit.py finetuned_v2 reference
#   rerun after the cleanup fix (same numbers, cached logits):
$PY scripts/sprint8_followup/diagnose_overfit.py finetuned_v2 reference

# U3 calibration 2-fold speaker-group cross-fit on validation only
$PY scripts/sprint8_followup/calibration_crossfit.py reference continued_clean finetuned_v2

# U4 unknown rejection, validation only (rule comparison, pre-declared requirement)
$PY scripts/sprint8_followup/unknown_rejection_dev.py finetuned_v2

# U5 duration effect on validation digit accuracy (inline; see PRINCIPLES_AUDIT.md §D)
```
Logs: results/sprint8_followup/logs/. Test split not read by any command above.
