"""Differentiable z-halo exchange: ghost layers match the neighbour slabs (forward) and the adjoint
folds ghost cotangents back into the owner's boundary layer, matching a single-process reference."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from spinforge.dist.collectives import all_reduce_sum
from spinforge.dist.halo import z_halo_exchange


def _slab(full: torch.Tensor, rank: int, nzl: int) -> torch.Tensor:
    return full[:, :, rank * nzl : (rank + 1) * nzl]


def _ref_haloed(full: torch.Tensor, rank: int, world: int, nzl: int) -> torch.Tensor:
    """Single-process equivalent of the distributed halo: slab padded, replicate at global ends."""
    s0, s1 = rank * nzl, (rank + 1) * nzl
    lo = full[:, :, s0 - 1 : s0] if rank > 0 else full[:, :, s0 : s0 + 1]
    hi = full[:, :, s1 : s1 + 1] if rank < world - 1 else full[:, :, s1 - 1 : s1]
    return torch.cat([lo, full[:, :, s0:s1], hi], dim=2)


def _worker(rank: int, world: int, init: str) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=init)
    try:
        nx, ny, nz = 3, 3, 8
        nzl = nz // world
        torch.manual_seed(0)
        full = torch.randn(nx, ny, nz, 3, dtype=torch.float64)

        # forward: ghost layers equal the neighbour slabs (replicate at the global ends)
        local = _slab(full, rank, nzl).clone()
        haloed = z_halo_exchange(local, world, rank)
        # the halo forward is a pure copy -> bit-identical (matches test_odd_world's explicit gate)
        torch.testing.assert_close(haloed, _ref_haloed(full, rank, world, nzl), rtol=0.0, atol=0.0)

        # adjoint: sum-of-all-haloed gradient (boundary cells appear in two slabs) matches the
        # single-process reference built from the same loop
        local_g = _slab(full, rank, nzl).clone().requires_grad_(True)
        loss = all_reduce_sum(z_halo_exchange(local_g, world, rank).sum())
        loss.backward()

        full_ref = full.clone().requires_grad_(True)
        total = sum(_ref_haloed(full_ref, r, world, nzl).sum() for r in range(world))
        total.backward()
        assert full_ref.grad is not None
        g_ref = _slab(full_ref.grad, rank, nzl)
        torch.testing.assert_close(local_g.grad, g_ref, rtol=1e-9, atol=1e-9)
        # fail-loud: wrong layout raises before any send/recv
        with pytest.raises(ValueError, match="nx, ny, nz_local, 3"):
            z_halo_exchange(torch.randn(3, 3, 2, dtype=torch.float64), world, rank)
    finally:
        dist.destroy_process_group()


@pytest.mark.dist
def test_z_halo_forward_and_adjoint() -> None:
    world_size = 4
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(
            _worker, args=(world_size, init), nprocs=world_size, join=True, start_method="spawn"
        )
