"""NCCL/scale safety of the timing harness (gloo-green, GPU-node-crash).

Two pinned contracts against the real-node failure modes:

1. **Control-plane reductions**: timing scalars are host floats; reducing them through the compute
   backend crashes under NCCL (no CPU tensors). The harness must carry them over an explicit gloo
   control-plane subgroup, whatever the compute backend is.
2. **Slab-only timing inputs**: the timing grid may exceed one device in aggregate (that is the
   point of the project), so no rank may materialize the full global m at the timing grid. The
   full-reference reference-match gate runs on a separate small gate grid instead.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from benchmarks.bench_dist_demag import (
    Config,
    make_ctrl_group,
    make_timing_inputs,
    reduce_max_seconds,
)


def _ctrl_worker(rank: int, world_size: int, init: str) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=init)
    try:
        ctrl = make_ctrl_group()
        assert dist.get_backend(ctrl) == "gloo"  # explicit gloo, NOT the compute backend
        # rank 0 contributes 0.1 s, rank 1 contributes 0.2 s -> everyone sees the slowest rank
        got = reduce_max_seconds(0.1 * (rank + 1), ctrl)
        assert got == pytest.approx(0.1 * world_size, rel=1e-12)
    finally:
        dist.destroy_process_group()


@pytest.mark.dist
def test_timing_reduction_runs_on_a_gloo_control_plane() -> None:
    world_size = 2
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(
            _ctrl_worker,
            args=(world_size, init),
            nprocs=world_size,
            join=True,
            start_method="spawn",
        )


def _cfg(n: tuple[int, int, int]) -> Config:
    return Config(n=n, dtype=torch.float64, reps=1, warmup=0, seed=7, link_bw=None, device="cpu")


def test_timing_inputs_are_slab_only() -> None:
    # only this rank's [nx, ny, nz/world, 3] slab is materialized, unit-normalized
    mesh, m_local = make_timing_inputs(_cfg((4, 4, 8)), rank=1, world=2)
    assert mesh.n == (4, 4, 8)
    assert m_local.shape == (4, 4, 4, 3)
    torch.testing.assert_close(
        m_local.norm(dim=-1), torch.ones(4, 4, 4, dtype=torch.float64), rtol=1e-12, atol=1e-14
    )


def test_timing_inputs_deterministic_and_rank_distinct() -> None:
    a0 = make_timing_inputs(_cfg((4, 4, 8)), rank=0, world=2)[1]
    b0 = make_timing_inputs(_cfg((4, 4, 8)), rank=0, world=2)[1]
    a1 = make_timing_inputs(_cfg((4, 4, 8)), rank=1, world=2)[1]
    torch.testing.assert_close(a0, b0, rtol=0.0, atol=0.0)  # same seed+rank -> same slab
    assert not torch.equal(a0, a1)  # different ranks -> different content
