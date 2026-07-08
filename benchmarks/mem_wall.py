"""Memory-wall sizing for the over-one-GPU demo: measure bytes/cell, predict the largest grid.

Two decoupled halves:

* **Fit logic (this module's tested core, no GPU):** take ``(grid, measured_bytes)`` points, fit
  the linear working-set model ``bytes = a * n_cells + b`` (a = bytes/cell, b = fixed overhead), and
  predict the largest cubic grid that fits a VRAM budget. Pure arithmetic -- unit-tested on CPU.
* **Measurement seam (injectable, GPU-only):** a ``Measurer`` callable ``grid -> MemPoint`` fills
  the points on the real card. The default single-GPU measurer lives in ``measure_demag_step`` and
  is imported lazily so this module (and its tests) load without CUDA.

**Caveat baked into the API (the spatial memory wall must be measured, not manufactured).**
``torch.cuda.max_memory_allocated`` misses cuFFT's raw
``cudaMalloc``'d working buffers (see ``bench_demag.py``), which for the FFT demag are the
*dominant* set. So ``MemPoint`` carries BOTH the torch peak AND an ``smi_delta_bytes`` slot (an
``nvidia-smi`` used-memory delta around the step); the fit prefers the smi delta when present. And
predicting a wall grid by a linear bytes/cell fit is a **measurement-based estimate to pick the demo
grid, NOT a scaling-curve extrapolation** -- the final over-one-GPU wall is measured on the card.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

Grid = tuple[int, int, int]


@dataclass(frozen=True)
class MemPoint:
    """One measured point: the grid and its working-set size by both accounting methods.

    ``torch_peak_bytes`` = ``torch.cuda.max_memory_allocated`` (misses raw cuFFT buffers).
    ``smi_delta_bytes`` = ``nvidia-smi`` used-memory delta around the step (the true wall), or None
    if unmeasured; the fit uses it in preference to the torch peak when available.
    """

    n: Grid
    torch_peak_bytes: float
    smi_delta_bytes: float | None

    @property
    def cells(self) -> int:
        return self.n[0] * self.n[1] * self.n[2]

    @property
    def bytes(self) -> float:
        """The working-set figure the fit uses: the smi delta if measured, else the torch peak."""
        return self.smi_delta_bytes if self.smi_delta_bytes is not None else self.torch_peak_bytes


@dataclass(frozen=True)
class MemFit:
    """Linear working-set model ``bytes = bytes_per_cell * n_cells + overhead_bytes``."""

    bytes_per_cell: float
    overhead_bytes: float
    max_abs_residual_bytes: float

    def predict_bytes(self, n: Grid) -> float:
        return self.bytes_per_cell * (n[0] * n[1] * n[2]) + self.overhead_bytes


def fit_bytes_per_cell(points: list[MemPoint]) -> MemFit:
    """Least-squares fit ``bytes = a * cells + b`` over >=2 distinct-``cells`` points (no GPU)."""
    xs = [float(p.cells) for p in points]
    ys = [p.bytes for p in points]
    if len({p.cells for p in points}) < 2:
        raise ValueError("need >=2 points with distinct cell counts to fit slope + intercept")
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    a = sxy / sxx
    b = my - a * mx
    residual = max(abs(y - (a * x + b)) for x, y in zip(xs, ys, strict=True))
    return MemFit(bytes_per_cell=a, overhead_bytes=b, max_abs_residual_bytes=residual)


def largest_grid_within(fit: MemFit, vram_bytes: float, step: int = 1) -> int:
    """Largest cubic side ``n`` (a multiple of ``step``) whose predicted working set fits ``vram``.

    An estimate for choosing the demo grid, not a promise -- re-measure the wall on the real card.
    """
    if step < 1:
        raise ValueError("step must be >= 1")
    n = 0
    while fit.predict_bytes((n + step, n + step, n + step)) <= vram_bytes:
        n += step
    return n


# --- GPU measurement seam (injectable; imported lazily so CPU/CI never needs CUDA) ---------------

Measurer = Callable[[Grid], MemPoint]


def sweep(grids: list[Grid], measurer: Measurer) -> list[MemPoint]:
    """Run ``measurer`` over ``grids``, collecting points; the measurer is the sole GPU dep."""
    return [measurer(g) for g in grids]


def measure_demag_step(
    dtype_name: str = "float32", mode: str = "fwd_adj", ms: float = 8e5
) -> Measurer:
    """Default single-GPU measurer: peak memory of one demag step (torch peak + nvidia-smi delta).

    GPU-only (imports torch/CUDA on call, not at module load). ``mode`` in
    {"fwd", "fwd_adj", "fwd_adj_ckpt"} mirrors the timing harness. Fill the smi delta from an
    ``nvidia-smi --query-gpu=memory.used`` read before/after so the raw cuFFT buffers are counted.
    """
    import subprocess

    import torch

    from spinforge.core.demag import DemagField
    from spinforge.core.mesh import Mesh

    dtype = getattr(torch, dtype_name)

    def _smi_used_bytes() -> float | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True,
            )
            return float(out.stdout.splitlines()[0]) * 2**20  # memory.used is MiB -> bytes
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError, IndexError):
            return None  # no smi -> torch peak only (undercounts cuFFT; flagged in the report)

    def _measure(n: Grid) -> MemPoint:
        mesh = Mesh(n=n, dx=(2e-9, 2e-9, 2e-9))
        field = DemagField(mesh)
        m = torch.randn(*n, 3, dtype=dtype, device="cuda")
        m = m / m.norm(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        smi0 = _smi_used_bytes()
        if mode == "fwd":
            with torch.no_grad():
                field(m, ms)
        else:
            mg = m.detach().clone().requires_grad_(True)
            field(mg, ms).sum().backward()  # fwd + adjoint working set held live
        torch.cuda.synchronize()
        smi1 = _smi_used_bytes()
        peak = float(torch.cuda.max_memory_allocated())
        delta = (smi1 - smi0) if (smi0 is not None and smi1 is not None) else None
        return MemPoint(n=n, torch_peak_bytes=peak, smi_delta_bytes=delta)

    return _measure


def main() -> None:
    """CLI: sweep grids on the single GPU, fit bytes/cell, predict + probe the capacity wall.

    --sweep 192,256,320   measure fwd+adjoint working sets, fit, predict the wall for --vram-gb
    --probe 512           attempt one single-GPU fwd+adjoint at 512^3: prints FITS or OOM (the
                          wall's boundary evidence -- a measurement, not a staged demo)
    """
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default=None, help="comma-separated cubic sides to measure")
    ap.add_argument("--probe", type=int, default=None, help="single cubic side to attempt")
    ap.add_argument(
        "--vram-gb",
        type=float,
        default=None,
        help="VRAM budget; default = read the actual card (never a hardcoded model's size)",
    )
    ap.add_argument("--reserve-gb", type=float, default=6.0, help="headroom for allocator/context")
    ap.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    ap.add_argument(
        "--step",
        type=int,
        default=32,
        help="wall prediction granularity (cubic side, cells per edge)",
    )
    args = ap.parse_args()

    if args.vram_gb is None:
        import torch

        args.vram_gb = torch.cuda.get_device_properties(0).total_memory / 2**30

    if args.sweep:
        sides = [int(x) for x in args.sweep.split(",")]
        measurer = measure_demag_step(dtype_name=args.dtype)
        points = sweep([(s, s, s) for s in sides], measurer)
        fit = fit_bytes_per_cell(points)
        budget = (args.vram_gb - args.reserve_gb) * 2**30
        wall = largest_grid_within(fit, budget, step=args.step)
        print(
            json.dumps(
                {
                    "points": [
                        {"side": p.n[0], "bytes": p.bytes, "torch_peak": p.torch_peak_bytes}
                        for p in points
                    ],
                    "bytes_per_cell": fit.bytes_per_cell,
                    "overhead_gb": fit.overhead_bytes / 2**30,
                    "max_abs_residual_gb": fit.max_abs_residual_bytes / 2**30,
                    "predicted_wall_side": wall,
                    "budget_gb": budget / 2**30,
                },
                indent=2,
            )
        )
    if args.probe is not None:
        import torch

        n = args.probe
        try:
            point = measure_demag_step(dtype_name=args.dtype)((n, n, n))
            print(json.dumps({"probe_side": n, "result": "FITS", "bytes": point.bytes}))
        except torch.OutOfMemoryError as err:
            print(json.dumps({"probe_side": n, "result": "OOM", "error": str(err)[:200]}))


if __name__ == "__main__":
    main()
