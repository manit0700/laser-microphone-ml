# Hardware to ML Input Contract

Sprint 4 goal: make the ML pipeline ready to test real PCM1808 / DAQ captures
when hardware becomes available.

## Preferred Input

Use short WAV files when possible.

| Field | Requirement |
| --- | --- |
| File type | `.wav` preferred, `.csv` supported |
| Channels | mono preferred; stereo is averaged to mono |
| Sample rate | known sample rate required; 8000 Hz or 16000 Hz preferred |
| Clip length | about 1 second per spoken digit |
| Naming | start filename with expected digit |

Example filenames:

```text
0_pcm1808_01.wav
1_pcm1808_01.wav
2_laser_01.csv
```

## CSV Input

CSV captures are supported for DAQ-style output.

Preferred columns:

```text
time,voltage
0.00000,0.0123
0.00002,0.0156
```

Rules:

- `voltage` is treated as the signal.
- `time` is used to estimate the sample rate.
- If no `voltage` column exists, the last numeric column is used.
- If no `time` column exists, pass the sample rate manually.

Example:

```bash
python scripts/test_hardware_captures.py --input-dir data/hardware_test --sample-rate 16000
```

## ML Output

The ML prediction output is:

```json
{
  "prediction": "5",
  "confidence": 0.87,
  "status": "recognized"
}
```

If confidence is too low:

```json
{
  "prediction": "unknown",
  "confidence": 0.42,
  "status": "low_confidence"
}
```

## Sprint 4 Test Command

Put hardware captures in:

```text
data/hardware_test/
```

Run:

```bash
python scripts/test_hardware_captures.py
```

The script writes:

```text
results/hardware_test_report.csv
```

The report includes:

- expected digit
- predicted digit
- confidence
- sample rate
- duration
- peak
- RMS
- signal issue notes

## What Hardware Team Should Send

Minimum useful dataset:

- 1 to 2 recordings for each digit `0-9`
- file format used: WAV or CSV
- sample rate
- whether the signal is mono or stereo
- any gain/filtering used during capture

