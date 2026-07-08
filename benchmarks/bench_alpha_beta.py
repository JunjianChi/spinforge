"""Measure the transpose collective's (alpha, beta): a payload sweep fitted to t = a + b/beta.

The alpha-beta comm model (roofline.py) needs two MEASURED machine parameters: the per-message
latency alpha and the per-rank ejection bandwidth beta. This sweep times ``dist.all_to_all_single``
over a log-spread of payload sizes with the same discipline as the main harness (device sync, gloo
control-plane barrier and MAX-reduce, warmup, repetitions) and least-squares fits the model. The
fitted intercept is one COLLECTIVE's latency = (world-1) per-peer messages, so the reported
``alpha_per_message_s`` = intercept / (world-1) -- that is the number ``bench_dist_demag.py
--alpha-us`` expects.

Run (harness validation, Mac/CPU/gloo -- CPU numbers are NOT machine parameters):
    PYTHONPATH=src:. torchrun --nproc_per_node=2 benchmarks/bench_alpha_beta.py
Run (AutoDL, NCCL): same via torchrun with the real world size; add --run-name for provenance
capture under runs/ (seed + config + git commit), then feed the outputs to bench_dist_demag
--alpha-us/--link-bw-gbs.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch
import torch.distributed as dist

from benchmarks.bench_dist_demag import _environment
from benchmarks.roofline import fit_alpha_beta
from spinforge.io import set_seed


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_all_to_all(x: torch.Tensor, reps: int, warmup: int, ctrl: dist.ProcessGroup) -> float:
    """Mean wall seconds of one all_to_all_single at this payload, MAX-reduced over ranks."""
    out = torch.empty_like(x)
    walls: list[float] = []
    for i in range(warmup + reps):
        _sync()
        dist.barrier(group=ctrl)
        t0 = time.perf_counter()
        dist.all_to_all_single(out, x)
        _sync()
        dt = torch.tensor([time.perf_counter() - t0], dtype=torch.float64)
        dist.all_reduce(dt, op=dist.ReduceOp.MAX, group=ctrl)  # host tensor stays off NCCL
        if i >= warmup:
            walls.append(float(dt.item()))
    return sum(walls) / len(walls)


def measure_alpha_beta(
    sizes_bytes: list[int],
    reps: int,
    warmup: int,
    ctrl: dist.ProcessGroup,
    device: str,
    seed: int = 0,
) -> dict[str, Any]:
    """Sweep payload sizes, fit t = intercept + wire_bytes/beta, convert to per-message alpha.

    The fitted x-axis is the OFF-RANK wire bytes ((world-1)/world of the local payload), matching
    roofline.wire_bytes_per_rank, so the fitted beta plugs directly into --link-bw-gbs.
    """
    set_seed(seed)
    world = dist.get_world_size()
    if world < 2:
        # (world-1)/world of every payload is zero: a single rank puts nothing on the wire, so
        # there is nothing to fit (and NCCL cannot host 2 ranks on one GPU -- measured 4060 refusal)
        raise ValueError("the alpha-beta sweep needs >= 2 ranks; world=1 has no wire traffic")
    samples: list[dict[str, float | int]] = []
    for size in sizes_bytes:
        per_dest = max(1, size // (8 * world))  # f64 elements sent to each peer
        x = torch.randn(world, per_dest, dtype=torch.float64, device=device)
        wire = x.numel() * x.element_size() * (world - 1) // world
        samples.append({"wire_bytes": wire, "wall_s": _time_all_to_all(x, reps, warmup, ctrl)})
    intercept, beta = fit_alpha_beta(
        [int(s["wire_bytes"]) for s in samples], [float(s["wall_s"]) for s in samples]
    )
    return {
        "intercept_s": intercept,
        "alpha_per_message_s": intercept / (world - 1),
        "beta_bytes_per_s": beta,
        "samples": samples,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sizes-kb",
        default="4,32,256,2048,16384",
        help="comma-separated payload sizes (per-rank local bytes / 1024), log-spread",
    )
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default=None, help="capture provenance under runs/ (rank 0)")
    p.add_argument("--repo-root", default=".")
    args = p.parse_args()

    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(rank % torch.cuda.device_count())
        ctrl = dist.new_group(backend="gloo")
        assert ctrl is not None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sizes = [int(s) * 1024 for s in args.sizes_kb.split(",")]
        result = measure_alpha_beta(sizes, args.reps, args.warmup, ctrl, device, seed=args.seed)
        if rank == 0:
            summary = {
                "config": {
                    "world_size": world,
                    "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                    "backend": dist.get_backend(),
                    "reps": args.reps,
                    "warmup": args.warmup,
                    "seed": args.seed,
                    "torch": torch.__version__,
                    "env": _environment(),
                    "cpu_run_disclaimer": None
                    if torch.cuda.is_available()
                    else "CPU/gloo run: harness validation only, NOT machine parameters",
                },
                "alpha_us_per_message": 1e6 * result["alpha_per_message_s"],
                "beta_gbs": result["beta_bytes_per_s"] / 1e9,
                "result": result,
            }
            print(json.dumps(summary, indent=2))
            if args.run_name:
                from spinforge.io import StorageLayout, capture_provenance

                run_dir = StorageLayout.at(args.repo_root).run_dir(args.run_name)
                capture_provenance(run_dir, summary["config"], args.seed)  # type: ignore[arg-type]
                (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
