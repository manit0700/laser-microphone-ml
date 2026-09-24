"""
Check whether the Laser Microphone ML project is ready to run on Linux/Jetson.

Run on the Jetson from the project root:
    python scripts/check_jetson_runtime.py

This does not need a GUI or microphone. It checks required imports, PyTorch/CUDA,
model files, and one simulated prediction.
"""

from __future__ import annotations

import importlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.append(str(SRC))


def check_import(name: str, required: bool = True) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "installed")
        return True, str(version)
    except Exception as exc:  # noqa: BLE001 - report any setup/import problem
        label = "required" if required else "optional"
        return False, f"{label} import failed: {type(exc).__name__}: {exc}"


def main() -> int:
    print("Laser Microphone ML - Jetson/Linux Runtime Check")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print()

    required = ["numpy", "scipy", "pandas", "soundfile", "torch", "torchaudio"]
    optional = ["PyQt5", "pyqtgraph", "sounddevice", "pyaudio"]

    ok = True
    print("Required packages:")
    for name in required:
        passed, message = check_import(name, required=True)
        ok = ok and passed
        print(f"  {'OK' if passed else 'FAIL'} {name}: {message}")

    print("\nOptional packages:")
    for name in optional:
        passed, message = check_import(name, required=False)
        print(f"  {'OK' if passed else 'WARN'} {name}: {message}")

    if not ok:
        print("\nRequired imports failed. Fix package installation before running ML.")
        return 1

    import torch

    print("\nPyTorch runtime:")
    print(f"  torch version: {torch.__version__}")
    print(f"  cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  cuda version: {torch.version.cuda}")
        print(f"  gpu count: {torch.cuda.device_count()}")
        print(f"  gpu name: {torch.cuda.get_device_name(0)}")
    else:
        print("  note: CPU inference can still work; GPU acceleration is optional for demo.")

    model_files = [
        ROOT / "models" / "best_model.pt",
        ROOT / "models" / "best_model_cnn.pt",
    ]
    print("\nModel files:")
    for path in model_files:
        exists = path.exists()
        ok = ok and exists
        size = f"{path.stat().st_size / 1024 / 1024:.2f} MB" if exists else "missing"
        print(f"  {'OK' if exists else 'FAIL'} {path.relative_to(ROOT)}: {size}")

    sample = ROOT / "data" / "hardware_test" / "5_simulated_01.wav"
    print("\nPrediction smoke test:")
    if not sample.exists():
        print(f"  FAIL missing sample: {sample.relative_to(ROOT)}")
        return 1

    try:
        from predict import predict_file

        result = predict_file(sample, model="ensemble")
        print(json.dumps(result, indent=2))
        status_ok = result.get("status") == "recognized"
        pred_ok = str(result.get("prediction")) == "5"
        if status_ok and pred_ok:
            print("  OK prediction pipeline works")
            return 0
        print("  WARN prediction ran, but result was not expected digit 5")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL prediction error: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
