"""The graded-anisotropy DESIGN term in the distributed solver (the big run's control knob).

Two gates, both on gloo/CPU:
  1. multi-rank matches the single-process reference: a distributed relax with a spatial delta-Ku
     map matches the reference (System field + the same design term) per slab.
  2. float64 gradcheck for the DESIGN VARIABLE: the gradient of a global loss w.r.t. dku entries
     -- the thing the optimizer consumes -- matches a coordinated cross-rank finite difference,
     probing
     interior AND slab-boundary cells. Gradcheck w.r.t. m is not enough: the optimizer steps on
     d(loss)/d(design), so that is the derivative that must be FD-true.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from spinforge.core.anisotropy import uniaxial_anisotropy_field
from spinforge.core.mesh import Mesh
from spinforge.core.system import Material, System, gilbert_rhs, rk4_step
from spinforge.dist.collectives import all_reduce_sum
from spinforge.dist.system import DistributedSystem

_MAT = Material(ms=8e5, a_ex=1.3e-11, ku=5e4, alpha=1.0)
_H_EXT = (0.0, 0.0, 1e5)


def _reference_relax(
    mesh: Mesh, m: torch.Tensor, dku: torch.Tensor, steps: int, dt: float
) -> torch.Tensor:
    """Single-process reference: System field + the same design term, same integrator."""
    sys_ = System(mesh, _MAT, demag=True, h_ext=_H_EXT)

    def rhs(mm: torch.Tensor) -> torch.Tensor:
        h = sys_.effective_field(mm) + uniaxial_anisotropy_field(
            mm, mesh, dku, _MAT.ku_axis, _MAT.ms
        )
        return gilbert_rhs(mm, h, _MAT.alpha)

    for _ in range(steps):
        m = rk4_step(rhs, m, dt)
    return m


def _worker(rank: int, world: int, init: str) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=init)
    try:
        nx, ny, nz = 4, 4, 8
        mesh = Mesh(n=(nx, ny, nz), dx=(2e-9, 2e-9, 2e-9))
        nzl = nz // world
        sl = slice(rank * nzl, (rank + 1) * nzl)
        torch.manual_seed(0)  # same tensors on every rank
        full = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        full = full / full.norm(dim=-1, keepdim=True)
        w = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        dku_full = 1e4 * torch.randn(nx, ny, nz, 1, dtype=torch.float64)
        dt, steps = 1e-14, 2
        dsys = DistributedSystem(mesh, _MAT, world, rank, h_ext=_H_EXT)

        # gate 1 (multi-rank matches reference): distributed relax with the design term matches the
        # reference per slab
        got = dsys.relax(
            full[:, :, sl].clone(), steps=steps, dt=dt, dku_map=dku_full[:, :, sl].clone()
        )
        ref = _reference_relax(mesh, full.clone(), dku_full, steps, dt)
        torch.testing.assert_close(got, ref[:, :, sl], rtol=1e-6, atol=1e-7)

        # gate 2 (float64 gradcheck on the design variable): d(global loss)/d(dku) vs coordinated
        # FD,
        # probing interior cells and both sides of the rank0|rank1 slab boundary
        dku_local = dku_full[:, :, sl].clone().requires_grad_(True)
        out = dsys.relax(
            full[:, :, sl].clone(), steps=steps, dt=dt, dku_map=dku_local, checkpoint_every=1
        )
        all_reduce_sum((out * w[:, :, sl]).sum()).backward()
        g = dku_local.grad
        assert g is not None

        def gloss(dku_all: torch.Tensor) -> float:
            o = dsys.relax(
                full[:, :, sl].clone(), steps=steps, dt=dt, dku_map=dku_all[:, :, sl].clone()
            )
            return float(all_reduce_sum((o * w[:, :, sl]).sum()))

        # FD-step sweep around eps ~ 1e-2 (dku ~ 1e4 J/m^3, so relative ~ 1e-6): the loss runs
        # through the checkpointed relax (stiff), where one hand-picked step can sit in the
        # cancellation-vs-truncation trough. Full sweep on every rank -- lockstep collectives.
        eps_sweep = (3e-2, 1e-2, 3e-3)
        for ix, iy, iz in ((0, 0, 0), (1, 2, 3), (2, 1, 4), (3, 3, 7)):
            owner = iz // nzl
            fds = []
            for eps in eps_sweep:
                with torch.no_grad():
                    dp, dm = dku_full.clone(), dku_full.clone()
                    if rank == owner:
                        dp[ix, iy, iz, 0] += eps
                        dm[ix, iy, iz, 0] -= eps
                fds.append((gloss(dp) - gloss(dm)) / (2.0 * eps))
            if rank == owner:
                auto = float(g[ix, iy, iz - owner * nzl, 0])
                # tolerance gate unchanged (rtol 1e-4 / atol 1e-12); the sweep relaxes only the step
                assert min(abs(fd - auto) for fd in fds) <= 1e-12 + 1e-4 * abs(auto), (
                    f"cell ({ix},{iy},{iz}): autograd {auto!r} vs FD "
                    f"{dict(zip(eps_sweep, fds, strict=True))!r}"
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.dist
def test_distributed_design_term_matches_and_gradchecks() -> None:
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(_worker, args=(2, init), nprocs=2, join=True, start_method="spawn")
