"""The alpha-beta payload sweep measures the transpose collective's latency/bandwidth split.

Gloo/CPU smoke of the measurement path only -- the fitted numbers on CPU loopback are harness
validation, never a performance claim; the real (alpha, beta) come from the GPU node. The pure fit
math is unit-tested in test_roofline.py; this pins the distributed sweep machinery: per-size
timings come back max-reduced over ranks, and the per-MESSAGE alpha is the fitted intercept
divided by (world-1) (one all_to_all = world-1 peer messages per rank).
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from benchmarks.bench_alpha_beta import measure_alpha_beta


def _worker(rank: int, world_size: int, init: str) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=init)
    try:
        ctrl = dist.new_group(backend="gloo")
        assert ctrl is not None
        # wide size spread so the slope stays positive above CPU-loopback noise
        result = measure_alpha_beta(
            sizes_bytes=[4_096, 4_194_304], reps=3, warmup=1, ctrl=ctrl, device="cpu"
        )
        assert result["beta_bytes_per_s"] > 0
        assert torch.isfinite(torch.tensor(result["alpha_per_message_s"]))
        # intercept -> per-message conversion: alpha_msg = intercept / (world - 1)
        assert result["alpha_per_message_s"] == pytest.approx(
            result["intercept_s"] / (world_size - 1), rel=1e-12
        )
        samples = result["samples"]
        assert len(samples) == 2
        # samples carry the fitted x-axis: off-rank wire bytes, ascending with requested size
        assert samples[0]["wire_bytes"] < samples[1]["wire_bytes"]
        assert all(s["wall_s"] > 0 for s in samples)
    finally:
        dist.destroy_process_group()


@pytest.mark.dist
def test_alpha_beta_sweep_runs_and_fits_on_gloo() -> None:
    world_size = 2
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(
            _worker, args=(world_size, init), nprocs=world_size, join=True, start_method="spawn"
        )


def _single_rank_worker(rank: int, world_size: int, init: str) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=init)
    try:
        ctrl = dist.new_group(backend="gloo")
        assert ctrl is not None
        # world=1 has zero off-rank wire bytes for EVERY payload -- the sweep is meaningless and
        # must say so, not die inside the fit with a confusing size-spread error. NCCL cannot run
        # 2 ranks on one GPU, so world=1 is what a single-GPU machine actually executes.
        with pytest.raises(ValueError, match="ranks"):
            measure_alpha_beta(
                sizes_bytes=[4_096, 4_194_304], reps=1, warmup=0, ctrl=ctrl, device="cpu"
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.dist
def test_alpha_beta_sweep_rejects_single_rank() -> None:
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(
            _single_rank_worker, args=(1, init), nprocs=1, join=True, start_method="spawn"
        )
