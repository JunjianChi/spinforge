"""Single-GPU baseline meter: T(best single-device implementation), the true-speedup denominator.

Speedup is measured against the best SINGLE-device implementation (the core
reference solver, no collectives, no pack) -- never against the distributed code on one rank,
whose transpose/pack overhead flatters the ratio. Same discipline as the distributed harness:
warmup, repetitions, device sync, raw samples, environment block.

    PYTHONPATH=src:. python benchmarks/bench_single.py --nx 256 --ny 256 --nz 256 \
        --dtype float32 --reps 20 --warmup 4 --run-name c2single_256
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch

from benchmarks.stats import timing_stats
from spinforge.core.demag import DemagField
from spinforge.core.mesh import Mesh
from spinforge.io import StorageLayout, capture_provenance, set_seed


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_single(
    n: tuple[int, int, int],
    dtype: torch.dtype,
    reps: int,
    warmup: int,
    device: str,
    ms: float,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Time the reference demag's fwd and fwd+adjoint steps; stats + raw samples per mode."""
    set_seed(seed)
    mesh = Mesh(n=n, dx=(2e-9, 2e-9, 2e-9))
    field = DemagField(mesh)
    m = torch.randn(*n, 3, dtype=dtype, device=device)
    m = m / m.norm(dim=-1, keepdim=True)

    def step(mode: str) -> None:
        if mode == "fwd":
            with torch.no_grad():
                field(m, ms)
        else:
            mg = m.detach().clone().requires_grad_(True)
            field(mg, ms).sum().backward()

    out: dict[str, dict[str, Any]] = {}
    for mode in ("fwd", "fwd_adj"):
        wall: list[float] = []
        for i in range(warmup + reps):
            _sync()
            t0 = time.perf_counter()
            step(mode)
            _sync()
            if i >= warmup:
                wall.append(time.perf_counter() - t0)
        out[mode] = timing_stats(wall)
    return out


def main() -> None:
    from benchmarks.bench_dist_demag import _environment  # the same self-describing env block

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--nz", type=int, required=True)
    ap.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ms", type=float, default=8e5)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = measure_single(
        n=(args.nx, args.ny, args.nz),
        dtype=getattr(torch, args.dtype),
        reps=args.reps,
        warmup=args.warmup,
        device=device,
        ms=args.ms,
        seed=args.seed,
    )
    summary = {
        "config": {
            "single": True,  # the true-speedup denominator marker (plot_c2 keys on it)
            "world_size": 1,
            "n": [args.nx, args.ny, args.nz],
            "dtype": args.dtype,
            "device": device,
            "reps": args.reps,
            "warmup": args.warmup,
            "seed": args.seed,
            "torch": torch.__version__,
            "env": _environment(),
        },
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    if args.run_name:
        run_dir = StorageLayout.at(args.repo_root).run_dir(args.run_name)
        capture_provenance(run_dir, summary["config"], args.seed)  # type: ignore[arg-type]
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
