"""The distributed effective field (exchange + demag + zeeman) matches single-process per slab,
forward and adjoint -- a multi-term distributed field whose gradient crosses ranks correctly."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from spinforge.core.mesh import Mesh
from spinforge.core.system import Material, System
from spinforge.dist.collectives import all_reduce_sum
from spinforge.dist.field import DistributedEffectiveField


def _worker(rank: int, world: int, init: str) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=init)
    try:
        nx, ny, nz = 4, 4, 8
        nzl = nz // world
        mesh = Mesh(n=(nx, ny, nz), dx=(2e-9, 2e-9, 2e-9))
        # exchange + demag + zeeman (no DMI -> single-process uses the decoupled path; matches here)
        mat = Material(ms=8e5, a_ex=1.3e-11, alpha=1.0)
        h_ext = (0.0, 0.0, 1e5)
        sl = slice(rank * nzl, (rank + 1) * nzl)
        torch.manual_seed(0)
        full = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        full = full / full.norm(dim=-1, keepdim=True)
        w = torch.randn(nx, ny, nz, 3, dtype=torch.float64)

        field = DistributedEffectiveField(mesh, mat, world, rank, h_ext=h_ext)
        ref_sys = System(mesh, mat, demag=True, h_ext=h_ext)

        # forward: distributed slab == single-process effective field on the slab
        got = field(full[:, :, sl].clone())
        ref = ref_sys.effective_field(full)[:, :, sl]
        torch.testing.assert_close(got, ref, rtol=1e-7, atol=1e-6)

        # adjoint: gradient crosses ranks (halo + all_to_all) and matches single-process autograd
        m_local = full[:, :, sl].clone().requires_grad_(True)
        all_reduce_sum((field(m_local) * w[:, :, sl]).sum()).backward()

        m_ref = full.clone().requires_grad_(True)
        (ref_sys.effective_field(m_ref) * w).sum().backward()
        assert m_ref.grad is not None
        torch.testing.assert_close(m_local.grad, m_ref.grad[:, :, sl], rtol=1e-6, atol=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.dist
def test_dmi_without_exchange_is_refused() -> None:
    """Fail-loud: the distributed field has no decoupled-DMI path -- System.effective_field
    would add bulk_dmi_field for d != 0, a_ex == 0, so silently computing exchange-only here would
    be wrong physics with no error. The constructor must refuse the combination."""
    mesh = Mesh(n=(4, 4, 4), dx=(2e-9, 2e-9, 2e-9))
    mat = Material(ms=1e5, a_ex=0.0, d=1e-4)
    with pytest.raises(ValueError, match="DMI without exchange"):
        DistributedEffectiveField(mesh, mat, world=2, rank=0)


def test_distributed_effective_field_matches_single_process() -> None:
    world_size = 4
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(
            _worker, args=(world_size, init), nprocs=world_size, join=True, start_method="spawn"
        )


# FeGe-like exchange + bulk DMI + anisotropy: the skyrmion-capable field. world=4 exercises BOTH
# rank classes: surface ranks (0, 3) apply the chiral free-surface ghost, interior ranks (1, 2)
# consume real neighbour layers.
_DMI_MAT = Material(ms=8e5, a_ex=1.3e-11, d=1.5e-3, ku=5e4, alpha=1.0)


def _dmi_worker(rank: int, world: int, init: str) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=init)
    try:
        nx, ny, nz = 4, 4, 8
        nzl = nz // world
        mesh = Mesh(n=(nx, ny, nz), dx=(2e-9, 2e-9, 2e-9))
        h_ext = (0.0, 0.0, 1e5)
        sl = slice(rank * nzl, (rank + 1) * nzl)
        torch.manual_seed(0)
        full = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        full = full / full.norm(dim=-1, keepdim=True)
        w = torch.randn(nx, ny, nz, 3, dtype=torch.float64)

        field = DistributedEffectiveField(mesh, _DMI_MAT, world, rank, h_ext=h_ext)
        ref_sys = System(mesh, _DMI_MAT, demag=True, h_ext=h_ext)  # chiral BC is the default

        got = field(full[:, :, sl].clone())
        ref = ref_sys.effective_field(full)[:, :, sl]
        torch.testing.assert_close(got, ref, rtol=1e-7, atol=1e-6)

        m_local = full[:, :, sl].clone().requires_grad_(True)
        all_reduce_sum((field(m_local) * w[:, :, sl]).sum()).backward()
        m_ref = full.clone().requires_grad_(True)
        (ref_sys.effective_field(m_ref) * w).sum().backward()
        assert m_ref.grad is not None
        torch.testing.assert_close(m_local.grad, m_ref.grad[:, :, sl], rtol=1e-6, atol=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.dist
def test_distributed_chiral_dmi_field_matches_single_process() -> None:
    """Reference-match for the skyrmion-capable field: exchange+DMI under the coupled chiral
    free-surface
    BC, distributed over 4 ranks, matches the single-process System forward AND adjoint."""
    world_size = 4
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(
            _dmi_worker,
            args=(world_size, init),
            nprocs=world_size,
            join=True,
            start_method="spawn",
        )


# global probe cells: chiral surfaces (z=0, z=7), the rank0|rank1 slab boundary (z=3, z=4), interior
_FD_CHECKS = [(1, 2, 0, 1), (0, 1, 7, 2), (2, 0, 3, 0), (1, 1, 4, 2), (3, 2, 5, 1)]


def _fd_worker(rank: int, world: int, init: str) -> None:
    """Coordinated cross-rank FD gradcheck through the chiral+DMI field: pins the halo adjoint at
    the slab boundary AND the chiral-ghost vjp at the global surfaces (where the discarded
    replicate ghost's zero cotangent and the direct chiral path must cancel exactly)."""
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=init)
    try:
        nx, ny, nz = 4, 4, 8
        nzl = nz // world
        mesh = Mesh(n=(nx, ny, nz), dx=(2e-9, 2e-9, 2e-9))
        sl = slice(rank * nzl, (rank + 1) * nzl)
        torch.manual_seed(0)
        full = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        full = full / full.norm(dim=-1, keepdim=True)
        w = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        field = DistributedEffectiveField(mesh, _DMI_MAT, world, rank)

        # forward gate first: without the DMI term the FD below would pass self-consistently
        ref = System(mesh, _DMI_MAT, demag=True).effective_field(full)[:, :, sl]
        torch.testing.assert_close(field(full[:, :, sl].clone()), ref, rtol=1e-7, atol=1e-6)

        def global_loss(m_all: torch.Tensor) -> torch.Tensor:
            h_local = field(m_all[:, :, sl].contiguous())
            return all_reduce_sum((h_local * w[:, :, sl]).sum())

        m_local = full[:, :, sl].clone().requires_grad_(True)
        all_reduce_sum((field(m_local) * w[:, :, sl]).sum()).backward()
        g_auto = m_local.grad
        assert g_auto is not None

        eps = 1e-6
        for ix, iy, iz, c in _FD_CHECKS:
            owner = iz // nzl
            with torch.no_grad():
                mp_plus, mp_minus = full.clone(), full.clone()
                if rank == owner:
                    mp_plus[ix, iy, iz, c] += eps
                    mp_minus[ix, iy, iz, c] -= eps
                lp, lm = float(global_loss(mp_plus)), float(global_loss(mp_minus))
            fd = (lp - lm) / (2.0 * eps)
            if rank == owner:
                g = float(g_auto[ix, iy, iz - owner * nzl, c])
                torch.testing.assert_close(torch.tensor(fd), torch.tensor(g), rtol=1e-5, atol=1e-4)
    finally:
        dist.destroy_process_group()


@pytest.mark.dist
def test_chiral_dmi_adjoint_matches_finite_difference() -> None:
    world_size = 2
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(
            _fd_worker,
            args=(world_size, init),
            nprocs=world_size,
            join=True,
            start_method="spawn",
        )
