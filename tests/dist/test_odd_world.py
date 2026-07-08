"""Odd world sizes and uneven slabs: the divisibility seams heFFTe stresses with np=7 (gloo, CPU).

Every other multi-rank test runs at world 2 or 4 -- powers of two with even splits, which rank
arithmetic in the transpose chunking, the owner-computes kernel slice, and the halo scatter-add can
accidentally satisfy. world=3 exercises the demag end-to-end (forward + adjoint vs the
single-process reference; the op requires nz divisible by world and this mesh satisfies it), a
genuinely uneven z-split exercises the halo (which has no divisibility requirement -- including a
one-layer slab, where first == last), and the indivisible-nz guard is pinned so the constraint
stays loud at construction instead of silently corrupting a transpose.
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
from spinforge.dist.demag import DistributedDemagField
from spinforge.dist.halo import z_halo_exchange


def _demag_worker3(rank: int, world_size: int, init: str) -> None:
    """world=3 demag: forward matches the reference slab, adjoint matches the reference gradient.

    Anisotropic mesh (nx != ny != nz, dx != dy != dz) so a chunk-index typo cannot hide behind
    symmetry. The per-rank loss (h_local * w_local).sum() is symmetric across ranks, so the summed
    implicit loss equals the single-process (ref * w).sum() and each rank's m_local.grad must equal
    the reference gradient's slab -- the cross terms ride the adjoint all_to_all at world 3.
    """
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=init)
    try:
        mesh = Mesh(n=(3, 4, 6), dx=(1e-9, 2e-9, 3e-9))
        nx, ny, nz, ms = *mesh.n, 8e5
        torch.manual_seed(0)
        m_full = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        m_full = m_full / m_full.norm(dim=-1, keepdim=True)
        w = torch.randn(nx, ny, nz, 3, dtype=torch.float64)

        m_ref = m_full.clone().requires_grad_(True)
        ref = DemagField(mesh)(m_ref, ms)
        (ref * w).sum().backward()
        assert m_ref.grad is not None

        nzl = nz // world_size
        sl = slice(rank * nzl, (rank + 1) * nzl)
        field = DistributedDemagField(mesh, world_size, rank)
        m_local = m_full[:, :, sl].clone().requires_grad_(True)
        h_local = field(m_local, ms)
        torch.testing.assert_close(h_local, ref.detach()[:, :, sl], rtol=1e-9, atol=1e-6)

        (h_local * w[:, :, sl]).sum().backward()
        assert m_local.grad is not None
        torch.testing.assert_close(m_local.grad, m_ref.grad[:, :, sl], rtol=1e-7, atol=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.dist
def test_distributed_demag_world3_matches_single_process() -> None:
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(
            _demag_worker3, args=(3, init), nprocs=3, join=True, start_method="spawn"
        )


_SPLITS = (3, 1, 2)  # deliberately uneven; the middle rank owns a single layer (first == last)


def _rank_weights(nx: int, ny: int, splits: tuple[int, ...]) -> list[torch.Tensor]:
    # every process regenerates the same per-rank weights from fixed seeds (no broadcast needed)
    ws = []
    for r, nzl in enumerate(splits):
        g = torch.Generator().manual_seed(100 + r)
        ws.append(torch.randn(nx, ny, nzl + 2, 3, dtype=torch.float64, generator=g))
    return ws


def _halo_worker_uneven(rank: int, world_size: int, init: str) -> None:
    """Uneven-slab halo: ghosts come from the true neighbour layers and the scatter-add adjoint
    returns every ghost cotangent to its owning boundary layer, matching a single-process replica
    of the same per-rank padded losses."""
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=init)
    try:
        nx, ny = 2, 3
        nz = sum(_SPLITS)
        offs = [sum(_SPLITS[:r]) for r in range(world_size + 1)]
        torch.manual_seed(0)
        m_full = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        ws = _rank_weights(nx, ny, _SPLITS)

        # single-process replica: build each rank's padded slab from m_full directly
        m_ref = m_full.clone().requires_grad_(True)
        padded_ref = []
        loss_ref = torch.zeros((), dtype=torch.float64)
        for r in range(world_size):
            lo = m_ref[:, :, offs[r] - 1 : offs[r]] if r > 0 else m_ref[:, :, :1]
            hi = (
                m_ref[:, :, offs[r + 1] : offs[r + 1] + 1]
                if r < world_size - 1
                else m_ref[:, :, -1:]
            )
            p = torch.cat([lo, m_ref[:, :, offs[r] : offs[r + 1]], hi], dim=2)
            padded_ref.append(p)
            loss_ref = loss_ref + (p * ws[r]).sum()
        loss_ref.backward()
        assert m_ref.grad is not None

        m_local = m_full[:, :, offs[rank] : offs[rank + 1]].clone().requires_grad_(True)
        padded = z_halo_exchange(m_local, world_size, rank)
        torch.testing.assert_close(padded, padded_ref[rank].detach(), rtol=0.0, atol=0.0)

        (padded * ws[rank]).sum().backward()
        assert m_local.grad is not None
        torch.testing.assert_close(
            m_local.grad,
            m_ref.grad[:, :, offs[rank] : offs[rank + 1]],
            rtol=1e-12,
            atol=1e-14,
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.dist
def test_halo_uneven_slabs_forward_and_adjoint() -> None:
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(
            _halo_worker_uneven, args=(3, init), nprocs=3, join=True, start_method="spawn"
        )


def test_uneven_nz_rejected_at_construction() -> None:
    """nz not divisible by world must fail loudly at init, never mis-chunk a transpose."""
    with pytest.raises(ValueError):
        DistributedDemagField(Mesh(n=(4, 4, 5), dx=(2e-9, 2e-9, 2e-9)), 2, 0)
