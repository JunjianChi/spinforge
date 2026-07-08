"""Timing benchmark for the native CUDA cuFFT demag (single GPU). Run on a CUDA host.

    CUDA_HOME=/usr/local/cuda-13.0 PATH=$CUDA_HOME/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9 \
        PYTHONPATH=src python benchmarks/bench_demag.py
"""

from __future__ import annotations

import json

import torch

from spinforge import native
from spinforge.core.mesh import Mesh
from spinforge.io import set_seed
from spinforge.native import native_demag


def _time(n: int, dtype: torch.dtype, iters: int) -> float:
    mesh = Mesh(n=(n, n, n), dx=(2e-9, 2e-9, 2e-9))
    ms = 8e5
    m = torch.randn(n, n, n, 3, dtype=dtype, device="cuda")
    m = m / m.norm(dim=-1, keepdim=True)
    native_demag(m, mesh, ms)  # warm-up (builds + caches the kernel, compiles the op)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        native_demag(m, mesh, ms)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench(n: int, iters: int = 50) -> None:
    """Compare FP64 (Z2Z) vs FP32 (C2C) demag at size n: time + throughput + speedup.

    Peak memory is not reported: the cuFFT working buffers are raw-``cudaMalloc``'d, invisible to
    the torch allocator, so any torch peak figure would miss the dominant working set. FP32 halves
    the complex-buffer bytes analytically (a 2x working-set reduction at the spatial wall).
    """
    try:
        t64, t32 = _time(n, torch.float64, iters), _time(n, torch.float32, iters)
    except RuntimeError as e:  # e.g. OOM at the largest size on a small card
        print(json.dumps({"n": n, "skipped": str(e).splitlines()[0]}))
        return
    mc64, mc32 = n**3 / t64 / 1e3, n**3 / t32 / 1e3
    print(
        json.dumps(
            {
                "n": n,
                "fp64_ms": round(t64, 2),
                "fp32_ms": round(t32, 2),
                "fp64_Mcells_s": round(mc64, 1),
                "fp32_Mcells_s": round(mc32, 1),
                "speedup": round(t64 / t32, 2),
            }
        )
    )


if __name__ == "__main__":
    set_seed(0)  # deterministic random m across runs (one seed drives every RNG)
    print(json.dumps({"device": torch.cuda.get_device_name()}))
    for n in (32, 64, 96, 128):
        bench(n)
        native._KERNEL_CACHE.clear()  # free per-size demag kernels so each size starts clean
        torch.cuda.empty_cache()
