# Sprint 8 training/evaluation environment (Agent B)

Compute: cloud dev container (2 vCPU, CPU-only PyTorch 2.15).

- Python 3.13.16
- torch 2.15.0+cu132 (device cpu, threads 2)
- platform Linux-6.18.44-fc-v310-x86_64-with-glibc2.39
- CPU: Intel(R) Xeon(R) Processor @ 2.10GHz x 2 logical cores
- sample rate 8000 Hz
- LMML_OUTPUT_DIR=/home/claude/lmml/results/sprint8/train

## pip freeze (excerpt)

```
matplotlib==3.11.2
numpy==2.5.4
pandas==3.1.1
scikit-learn==1.9.1
scipy==1.18.1
soundfile==0.14.0
torch==2.15.0
torchaudio==2.11.1
```

## Measured wall time per stage

| stage | wall time |
|---|---|
| reference training | 23.6 min (40 epochs, mean 35 s/epoch, preload 7 s) |
| fine-tune (finetuned_cnn.pt, max 15 ep) | 12.3 min (15 epochs, mean 45 s/epoch, preload 52 s) |
| control continued_clean (no sim) | 5.4 min (10 epochs, mean 10 s/epoch, preload 155 s) |
| fine-tune long (max 30 ep) | 25.9 min (30 epochs, mean 50 s/epoch, preload 56 s) |
| calibration round 1 (val; last run = additions) | 1.6 min |
| test round 1 (last run = additions only) | 1.4 min |
| fine-tune v2 (manifest_v2, max 30 ep) | 12.3 min (30 epochs, mean 17 s/epoch, preload 234 s) |
| calibration round 2 (v2 val) | 0.2 min |
| test round 2 (v2 test) | 1.3 min |
