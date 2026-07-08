"""S3: the cross-rank gradient-checkpointed distributed adjoint matches single-process autograd.

The recompute re-runs the transpose all_to_all collectives inside the backward pass; this proves
they stay coordinated across ranks (no deadlock) and that checkpointing changes only memory, not
the gradient -- the correctness prerequisite for the over-one-GPU regime (scale needs multi-GPU).
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
from spinforge.dist.checkpoint import checkpointed_demag
from spinforge.dist.collectives import all_reduce_sum
from spinforge.dist.demag import DistributedDemagField


def _worker(rank: int, world_size: int, init: str) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=init)
    try:
        nx, ny, nz, ms = 4, 4, 4, 8e5
        mesh = Mesh(n=(nx, ny, nz), dx=(2e-9, 2e-9, 2e-9))
        torch.manual_seed(0)
        m_full = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        m_full = m_full / m_full.norm(dim=-1, keepdim=True)
        w = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        sl = slice(rank * (nz // world_size), (rank + 1) * (nz // world_size))
        field = DistributedDemagField(mesh, world_size, rank)

        # checkpointed distributed adjoint: forward recomputed in backward (collectives re-run)
        m_local = m_full[:, :, sl].clone().requires_grad_(True)
        h_local = checkpointed_demag(field, m_local, ms)
        loss = all_reduce_sum((h_local * w[:, :, sl]).sum())
        loss.backward()
        g_ckpt = m_local.grad

        # forward must be unchanged by checkpointing
        with torch.no_grad():
            h_plain = field(m_full[:, :, sl].clone(), ms)
        torch.testing.assert_close(h_local, h_plain, rtol=1e-9, atol=1e-9)

        # single-process reference autograd gradient (gradchecked)
        m_ref = m_full.clone().requires_grad_(True)
        (DemagField(mesh)(m_ref, ms) * w).sum().backward()
        assert m_ref.grad is not None
        torch.testing.assert_close(g_ckpt, m_ref.grad[:, :, sl], rtol=1e-7, atol=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.dist
def test_checkpointed_distributed_adjoint_matches_single_process() -> None:
    world_size = 2
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(
            _worker, args=(world_size, init), nprocs=world_size, join=True, start_method="spawn"
        )
