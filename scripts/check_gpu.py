"""
check_gpu.py
============
One-command check that PyTorch can actually use the GPU, and how much faster
it is than the CPU. Run it inside whichever Python environment you train in:

    python scripts/check_gpu.py

Note: CPU timing is noisy if other training runs are using the cores.
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _bench(device: str, n: int = 2048, iters: int = 10) -> float:
    """Average seconds per n x n matmul on the given device."""
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    for _ in range(2):  # warm-up
        a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def main() -> int:
    print(f"torch {torch.__version__} | built for CUDA {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: PyTorch can't use the GPU in this environment.")
        return 1

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"device: {name} | compute capability {cap[0]}.{cap[1]}")

    cpu_t, gpu_t = _bench("cpu"), _bench("cuda")
    print(f"matmul 2048x2048: CPU {cpu_t * 1000:.1f} ms | GPU {gpu_t * 1000:.1f} ms "
          f"| {cpu_t / gpu_t:.1f}x faster")

    from config import DEVICE  # what train.py will actually pick
    print(f"train.py will use: {DEVICE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
