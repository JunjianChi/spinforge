"""Distributed exchange (halo-based) matches single-process exchange per slab, forward and adjoint:
the halo composes into a real field term whose gradient still crosses ranks correctly."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from spinforge.core.exchange import exchange_field
from spinforge.core.mesh import Mesh
from spinforge.dist.collectives import all_reduce_sum
from spinforge.dist.exchange import distributed_exchange_field


def _worker(rank: int, world: int, init: str) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=init)
    try:
        nx, ny, nz, a, ms = 4, 4, 8, 1.3e-11, 8e5
        nzl = nz // world
        mesh = Mesh(n=(nx, ny, nz), dx=(2e-9, 2e-9, 2e-9))
        sl = slice(rank * nzl, (rank + 1) * nzl)
        torch.manual_seed(0)
        full = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        full = full / full.norm(dim=-1, keepdim=True)
        w = torch.randn(nx, ny, nz, 3, dtype=torch.float64)

        # forward: distributed slab == single-process exchange restricted to the slab
        got = distributed_exchange_field(full[:, :, sl].clone(), mesh, a, ms, world, rank)
        ref = exchange_field(full, mesh, a, ms)[:, :, sl]
        torch.testing.assert_close(got, ref, rtol=1e-9, atol=1e-6)

        # adjoint: gradient crosses ranks through the halo and matches single-process autograd
        m_local = full[:, :, sl].clone().requires_grad_(True)
        h_local = distributed_exchange_field(m_local, mesh, a, ms, world, rank)
        all_reduce_sum((h_local * w[:, :, sl]).sum()).backward()

        m_ref = full.clone().requires_grad_(True)
        (exchange_field(m_ref, mesh, a, ms) * w).sum().backward()
        assert m_ref.grad is not None
        torch.testing.assert_close(m_local.grad, m_ref.grad[:, :, sl], rtol=1e-7, atol=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.dist
def test_distributed_exchange_matches_single_process() -> None:
    world_size = 4
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(
            _worker, args=(world_size, init), nprocs=world_size, join=True, start_method="spawn"
        )
