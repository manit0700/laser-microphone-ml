# Linux / Jetson Manual Setup

This project is written in Python and should run on Linux, including Jetson Orin
Nano Super. The safest first test is the command-line ML capture test, because
it does not need a GUI display or a live microphone.

Jetson note: the code is portable, but PyTorch installation depends on the
installed JetPack version. NVIDIA's Jetson PyTorch documentation says to install
JetPack first, then install Jetson-compatible PyTorch packages/wheels for that
JetPack version.

## 1. Clone And Enter Project

```bash
git clone https://github.com/manit0700/laser-microphone-ml.git
cd laser-microphone-ml
```

If the folder is already on the Jetson/Linux laptop:

```bash
cd laser-microphone-ml
git pull origin main
```

## 2. Create Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Check JetPack and system version:

```bash
cat /etc/nv_tegra_release
python3 --version
uname -m
```

Expected architecture is usually:

```text
aarch64
```

## 3. Install System Packages

For Ubuntu/Linux:

```bash
sudo apt update
sudo apt install -y python3-dev python3-pip python3-venv libsndfile1 portaudio19-dev
```

For the desktop dashboard:

```bash
sudo apt install -y python3-pyqt5
```

## 4. Install Python Packages

Normal Linux laptop:

```bash
pip install -r requirements.txt
pip install PyQt5 pyqtgraph sounddevice
```

Jetson note: PyTorch/torchaudio may need NVIDIA's Jetson-specific install
instead of the normal pip wheel. If `pip install torch torchaudio` fails or CUDA
does not work, install the PyTorch wheel that matches the JetPack version, then
install the rest:

```bash
pip install numpy scipy scikit-learn matplotlib pandas soundfile pyqtgraph sounddevice
```

After installing PyTorch, verify it:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

If `cuda available` is `False`, the project can still run on CPU for small demo
inference. For faster Jetson inference, install the NVIDIA PyTorch build that
matches your JetPack version.

## 4b. One-Command Runtime Check

Run this after installing packages:

```bash
python scripts/check_jetson_runtime.py
```

This checks:

- required Python packages
- optional dashboard/microphone packages
- PyTorch CUDA status
- model files
- one simulated prediction

## 5. Verify ML Without GUI

Run this first:

```bash
python scripts/test_hardware_captures.py
```

Expected result with the included simulated files:

```text
Accuracy on labeled captures: 9/10
```

This proves the model files, preprocessing, MFCC/mel feature extraction, and
prediction code are working.

## 6. Run With A Saved Capture

```bash
python src/predict.py data/hardware_test/5_simulated_01.wav --model ensemble
```

Expected output format:

```json
{
  "prediction": "5",
  "confidence": 0.98,
  "status": "recognized"
}
```

## 7. Run Dashboard On Linux

Only run this on a Linux desktop or Jetson connected to a display:

```bash
python signal_dashboard.py
```

If the microphone or PCM1808 input is not ready, replay a saved file:

```bash
LMML_SIGNAL_SOURCE="data/hardware_test/5_simulated_01.wav" python signal_dashboard.py
```

If running over SSH, use a display or X forwarding. Without a real display, the
PyQt5 dashboard will not open.

## 8. PCM1808 Notes

The ML code does not talk to PCM1808 directly. The hardware/software side must
provide one of these:

- WAV file
- CSV file with `time,voltage`
- live array/buffer passed into the Python backend

For the ML side, put files here:

```text
data/hardware_test/
```

Then run:

```bash
python scripts/test_hardware_captures.py
```
