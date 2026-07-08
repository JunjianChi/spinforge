"""Capstone: the DISTRIBUTED adjoint gradient matches the single-process autograd gradient.

Gradients flow across rank boundaries through the differentiable all_to_all in the distributed
demag.
The single-process autograd gradient is itself float64-gradchecked, so matching it proves the
distributed adjoint is correct -- the piece nobody has built, and the "not a clone" proof.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from spinforge.core.demag import DemagField
from spinforge.core.mesh import Mesh
from spinforge.dist.collectives import all_reduce_sum
from spinforge.dist.demag import DistributedDemagField


def _worker(rank: int, world_size: int, init: str) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=init)
    try:
        nx, ny, nz, ms = 4, 4, 4, 8e5
        mesh = Mesh(n=(nx, ny, nz), dx=(2e-9, 2e-9, 2e-9))
        torch.manual_seed(0)  # same m and loss weights on every rank
        m_full = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        m_full = m_full / m_full.norm(dim=-1, keepdim=True)
        w = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        sl = slice(rank * (nz // world_size), (rank + 1) * (nz // world_size))

        # distributed adjoint: gradients cross ranks via the differentiable all_to_all
        m_local = m_full[:, :, sl].clone().requires_grad_(True)
        h_local = DistributedDemagField(mesh, world_size, rank)(m_local, ms)
        loss = all_reduce_sum((h_local * w[:, :, sl]).sum())
        loss.backward()
        g_dist = m_local.grad

        # single-process reference autograd gradient (itself gradchecked)
        m_ref = m_full.clone().requires_grad_(True)
        (DemagField(mesh)(m_ref, ms) * w).sum().backward()
        assert m_ref.grad is not None
        g_ref = m_ref.grad[:, :, sl]

        torch.testing.assert_close(g_dist, g_ref, rtol=1e-7, atol=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.dist
def test_distributed_adjoint_matches_single_process_gradient() -> None:
    world_size = 2
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(
            _worker, args=(world_size, init), nprocs=world_size, join=True, start_method="spawn"
        )
