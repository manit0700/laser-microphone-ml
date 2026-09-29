"""
export_jetson.py
================
Export a trained model for fast inference on the Jetson Orin Nano and measure
how fast each option really is ON THAT BOARD.

Steps (each is skipped with a message if the tool isn't available):
  1. PyTorch FP32 latency          (baseline -- what the dashboard runs today)
  2. PyTorch FP16 latency          (CUDA only: model.half())
  3. ONNX export  -> models/<name>.onnx   (+ output check with onnxruntime if installed)
  4. TensorRT FP16 engine via trtexec -> models/<name>_fp16.engine  (+ its latency)
  5. Feature-extraction time per clip (preprocess + mel/MFCC), because the
     end-to-end delay is features + model.

The CNN is the natural choice for the Jetson: ~35K parameters vs ~545K for the
LSTM. Run with --model lstm too to compare, and --ensemble to see what running
both costs.

USAGE (on the Jetson, project root)
-----
    python3 scripts/export_jetson.py                 # CNN
    python3 scripts/export_jetson.py --model lstm
    python3 scripts/export_jetson.py --model cnn --runs 500

Results are printed and saved to results/reports/jetson_export_<model>.json.
TensorRT: JetPack ships `trtexec` at /usr/src/tensorrt/bin/trtexec.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from config import DEVICE, MAX_AUDIO_SAMPLES, REPORTS_DIR, model_checkpoint  # noqa: E402
from features import extract_features  # noqa: E402
from model import model_from_checkpoint  # noqa: E402
from preprocess import preprocess_waveform  # noqa: E402


class ExportableAdaptiveAvgPool2d(torch.nn.Module):
    """Exact, ONNX/TensorRT-friendly replacement for nn.AdaptiveAvgPool2d.

    The CNN pools a 10x15 feature map down to 4x4. ONNX cannot export adaptive
    pooling when the input size isn't a multiple of the output size (15 / 4), so
    we compute the SAME result as two matrix multiplications:  P_h @ X @ P_w^T,
    where row i of P_h averages input rows floor(i*H/out) .. ceil((i+1)*H/out)-1
    -- exactly PyTorch's adaptive-pooling windows.
    """

    def __init__(self, output_size):
        super().__init__()
        self.out_h, self.out_w = (output_size, output_size) if isinstance(output_size, int) else output_size

    @staticmethod
    def _pool_matrix(n_in: int, n_out: int, like: torch.Tensor) -> torch.Tensor:
        m = torch.zeros(n_out, n_in, dtype=like.dtype, device=like.device)
        for i in range(n_out):
            a = (i * n_in) // n_out
            b = -((-(i + 1) * n_in) // n_out)          # ceil((i+1)*n_in/n_out)
            m[i, a:b] = 1.0 / (b - a)
        return m

    def forward(self, x):
        h, w = int(x.shape[-2]), int(x.shape[-1])
        ph = self._pool_matrix(h, self.out_h, x)
        pw = self._pool_matrix(w, self.out_w, x)
        return torch.matmul(torch.matmul(ph, x), pw.t())


def make_exportable(model: torch.nn.Module) -> torch.nn.Module:
    """Copy of the model with every AdaptiveAvgPool2d swapped for the exact equivalent."""
    import copy
    m = copy.deepcopy(model)
    for name, mod in list(m.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, torch.nn.AdaptiveAvgPool2d):
                setattr(mod, child_name, ExportableAdaptiveAvgPool2d(child.output_size))
    return m.eval()


def _bench(fn, runs: int, sync: bool) -> dict:
    for _ in range(max(5, runs // 10)):          # warm-up
        fn()
    if sync:
        torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return {"mean_ms": float(np.mean(times)), "p95_ms": float(np.percentile(times, 95))}


def _find_trtexec():
    return shutil.which("trtexec") or next(
        (p for p in ("/usr/src/tensorrt/bin/trtexec", "/usr/local/tensorrt/bin/trtexec") if Path(p).exists()), None)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export + benchmark a model for the Jetson.")
    ap.add_argument("--model", choices=["cnn", "lstm"], default="cnn")
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--no-trt", action="store_true", help="skip the TensorRT engine build")
    args = ap.parse_args()

    ckpt_path = Path(model_checkpoint(args.model))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    feature = ckpt.get("feature", "mel" if args.model == "cnn" else "mfcc")
    model = model_from_checkpoint(ckpt).eval()
    n_params = sum(p.numel() for p in model.parameters())
    cuda = DEVICE.type == "cuda"
    print(f"Model: {args.model} ({ckpt_path.name}) | feature: {feature} | params: {n_params:,} | device: {DEVICE}")

    # A realistic input: 1 s of noise through the real preprocessing + feature path.
    wave = torch.randn(MAX_AUDIO_SAMPLES) * 0.05
    feat = extract_features(preprocess_waveform(wave), feature).unsqueeze(0)
    print(f"Input shape: {tuple(feat.shape)}")
    res = {"model": args.model, "feature": feature, "params": n_params,
           "input_shape": list(feat.shape), "device": str(DEVICE)}

    # 1. Features (CPU, per clip)
    res["features_cpu"] = _bench(lambda: extract_features(preprocess_waveform(wave), feature), 100, False)

    # 2. PyTorch FP32
    m32, x32 = model.to(DEVICE), feat.to(DEVICE)
    with torch.no_grad():
        ref = m32(x32).float().cpu()
        res["torch_fp32"] = _bench(lambda: m32(x32), args.runs, cuda)

    # 3. PyTorch FP16 (CUDA only)
    if cuda:
        import copy
        m16 = copy.deepcopy(m32).half()
        x16 = x32.half()
        with torch.no_grad():
            out16 = m16(x16).float().cpu()
            res["torch_fp16"] = _bench(lambda: m16(x16), args.runs, True)
        res["torch_fp16"]["max_abs_diff_vs_fp32"] = float((out16 - ref).abs().max())
        res["torch_fp16"]["same_prediction"] = bool(out16.argmax(1).eq(ref.argmax(1)).all())
    else:
        res["torch_fp16"] = "skipped (no CUDA -- FP16 is only faster on the GPU)"

    # 4. ONNX export
    onnx_path = ckpt_path.with_suffix(".onnx")
    model_cpu = make_exportable(model_from_checkpoint(ckpt).eval())
    with torch.no_grad():
        swap_diff = float((model_cpu(feat) - ref).abs().max())
    print(f"Export-safe model vs original: max logit difference {swap_diff:.2e} (should be ~0)")
    res["export_safe_max_diff"] = swap_diff
    try:
        kw = dict(input_names=["features"], output_names=["logits"], opset_version=17)
        try:
            torch.onnx.export(model_cpu, feat, str(onnx_path), dynamo=False, **kw)
        except TypeError:                         # older torch without the dynamo flag
            torch.onnx.export(model_cpu, feat, str(onnx_path), **kw)
        res["onnx"] = {"path": str(onnx_path), "size_kb": round(onnx_path.stat().st_size / 1024, 1)}
        print(f"ONNX exported -> {onnx_path}")
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(onnx_path), providers=ort.get_available_providers())
            feed = {"features": feat.numpy()}
            out = torch.from_numpy(sess.run(None, feed)[0])
            res["onnx"]["max_abs_diff_vs_torch"] = float((out - ref).abs().max())
            res["onnx"]["onnxruntime"] = _bench(lambda: sess.run(None, feed), args.runs, False)
            res["onnx"]["providers"] = sess.get_providers()
        except ImportError:
            res["onnx"]["onnxruntime"] = "not installed (pip install onnxruntime) -- export still OK"
    except Exception as e:  # noqa: BLE001
        res["onnx"] = f"export failed: {type(e).__name__}: {e}"
        onnx_path = None

    # 5. TensorRT FP16 via trtexec
    trtexec = None if args.no_trt else _find_trtexec()
    if onnx_path is not None and trtexec:
        engine = ckpt_path.with_name(ckpt_path.stem + "_fp16.engine")
        print(f"Building TensorRT FP16 engine with {trtexec} (can take a minute)...")
        p = subprocess.run([trtexec, f"--onnx={onnx_path}", "--fp16", f"--saveEngine={engine}",
                            f"--iterations={args.runs}", "--avgRuns=100"],
                           capture_output=True, text=True)
        m = re.search(r"GPU Compute Time: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms", p.stdout)
        if p.returncode == 0 and m:
            res["tensorrt_fp16"] = {"engine": str(engine), "mean_ms": float(m.group(3)),
                                    "min_ms": float(m.group(1)), "max_ms": float(m.group(2))}
        else:
            res["tensorrt_fp16"] = "trtexec failed: " + (p.stdout + p.stderr)[-600:]
    else:
        res["tensorrt_fp16"] = ("skipped (--no-trt)" if args.no_trt else
                                "skipped (trtexec not found -- on the Jetson it is /usr/src/tensorrt/bin/trtexec)")

    # Report
    def fmt(v):
        return f"{v['mean_ms']:.3f} ms (p95 {v['p95_ms']:.3f})" if isinstance(v, dict) and "p95_ms" in v else (
            f"{v['mean_ms']:.3f} ms" if isinstance(v, dict) and "mean_ms" in v else str(v))
    print("\n" + "=" * 70)
    print(f"{args.model.upper()} latency per clip on {DEVICE}")
    print(f"  features (preprocess + {feature}, CPU): {fmt(res['features_cpu'])}")
    print(f"  model, PyTorch FP32:                  {fmt(res['torch_fp32'])}")
    print(f"  model, PyTorch FP16:                  {fmt(res['torch_fp16'])}")
    onnx = res["onnx"]
    print(f"  model, ONNX Runtime:                  {fmt(onnx.get('onnxruntime')) if isinstance(onnx, dict) else onnx}")
    print(f"  model, TensorRT FP16:                 {fmt(res['tensorrt_fp16'])}")
    if isinstance(res["torch_fp16"], dict):
        print(f"  FP16 vs FP32: max logit diff {res['torch_fp16']['max_abs_diff_vs_fp32']:.4f}, "
              f"same prediction: {res['torch_fp16']['same_prediction']}")
    print("=" * 70)
    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    out = Path(REPORTS_DIR) / f"jetson_export_{args.model}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"Saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
