# Agent R — commands run (all read-only against the project tree)

Environment: `. /home/claude/.venv-lmml/bin/activate` (torch 2.15.0, torchaudio 2.11.1, scipy 1.18.1), cloud dev container (2 vCPU, CPU-only).

```bash
# Numeric checks: torchaudio/scipy resampling alias rejection, Butterworth 200-3800 Hz response and
# group delay (causal vs zero-phase), simulated-hum leakage through the band-pass, rig SNR from Exp 6 RMS.
python -I results/sprint8/research/scripts/checks.py

# Digit duration / internal pause / trailing-quiet statistics on 150 clean clips per digit per corpus
# (FSDD + Speech Commands in data/raw), 8 kHz, 5 % energy gate (config defaults). Reads data/raw only.
python -I results/sprint8/research/scripts/durations.py
```

No seeds are involved (deterministic except `checks.py` hum Monte-Carlo, which uses `np.random.default_rng(0)`).
Outputs are quoted inline in assumptions_review.md (sections 4, 5, 7, 8 and the derived rig numbers).
