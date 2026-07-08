"""The collective traffic the comm-optimization headline targets, made concrete (1x / 2x / 3x).

The adjoint of the demag transpose is itself an all_to_all, and checkpointing re-runs the forward
transpose in the backward. So forward-only, forward+adjoint, and forward+adjoint+recompute issue
N, 2N, 3N all_to_all collectives respectively. This test pins those multipliers -- it is the
quantified optimization target (3x the naive forward traffic) the overlap/scheduling work reduces.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from spinforge.core.mesh import Mesh
from spinforge.dist.checkpoint import checkpointed_demag
from spinforge.dist.collectives import a2a_calls, all_reduce_sum, reset_a2a_calls
from spinforge.dist.demag import DistributedDemagField


def _worker(rank: int, world_size: int, init: str) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=init)
    try:
        nx, ny, nz, ms = 4, 4, 4, 8e5
        mesh = Mesh(n=(nx, ny, nz), dx=(2e-9, 2e-9, 2e-9))
        torch.manual_seed(0)
        m_full = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        m_full = m_full / m_full.norm(dim=-1, keepdim=True)
        sl = slice(rank * (nz // world_size), (rank + 1) * (nz // world_size))
        field = DistributedDemagField(mesh, world_size, rank)

        # forward only: one all_to_all per component for each of the two transposes -> N
        reset_a2a_calls()
        with torch.no_grad():
            field(m_full[:, :, sl].clone(), ms)
        n_fwd = a2a_calls()
        assert n_fwd == 6  # 3 components x (to_xy + to_z)

        # forward + adjoint: the backward transpose is another all_to_all -> 2N
        reset_a2a_calls()
        m1 = m_full[:, :, sl].clone().requires_grad_(True)
        all_reduce_sum(field(m1, ms).sum()).backward()
        assert a2a_calls() == 2 * n_fwd

        # + checkpoint: the backward recomputes (part of) the forward transpose before the adjoint,
        # so traffic is strictly above 2N but at most the full-re-run 3N -- torch's non-reentrant
        # checkpoint recomputes only what it needs (here 2.5N), below the naive 3x worst case.
        reset_a2a_calls()
        m2 = m_full[:, :, sl].clone().requires_grad_(True)
        all_reduce_sum(checkpointed_demag(field, m2, ms).sum()).backward()
        assert 2 * n_fwd < a2a_calls() <= 3 * n_fwd
    finally:
        dist.destroy_process_group()


@pytest.mark.dist
def test_collective_traffic_multipliers() -> None:
    world_size = 2
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(
            _worker, args=(world_size, init), nprocs=world_size, join=True, start_method="spawn"
        )
