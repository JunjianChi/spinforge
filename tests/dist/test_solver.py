"""End-to-end distributed LLG solver: the two gates that define "deployable" (gloo/CPU, no
hardware).

Gate 1 (multi-rank matches the single-process reference): a full distributed relaxation (RK4 +
|m|=1 renorm, effective field = exchange-halo
+
transpose-FFT demag + zeeman + anisotropy) matches the single-process reference per slab, at a
non-trivial multi-slab size (32x32x16), for N=2 and N=4 ranks.

Gate 2 (full-solver cross-rank adjoint): the gradient through the WHOLE distributed solve
(multi-step
relax, not just the demag op in isolation) matches a coordinated cross-rank finite difference -- the
end-to-end distributed adjoint, the "not a clone" proof. One global input cell is perturbed; every
rank re-runs the full relax in lockstep (the collectives need all ranks); the owning rank compares
the
central difference of the global loss to its autograd gradient.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from spinforge.core.mesh import Mesh
from spinforge.core.system import Material, System
from spinforge.dist.collectives import a2a_calls, all_reduce_sum, reset_a2a_calls
from spinforge.dist.system import DistributedSystem

# exchange (a_ex) + demag + zeeman + uniaxial anisotropy (ku); d=0 -> decoupled exchange path, which
# the distributed exchange-halo reproduces. alpha=1 is the high-damping relaxation regime.
_MAT = Material(ms=8e5, a_ex=1.3e-11, ku=5e4, alpha=1.0)
_H_EXT = (0.0, 0.0, 1e5)


def _match_worker(rank: int, world: int, init: str, nx: int, ny: int, nz: int) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=init)
    try:
        mesh = Mesh(n=(nx, ny, nz), dx=(2e-9, 2e-9, 2e-9))
        nzl = nz // world
        sl = slice(rank * nzl, (rank + 1) * nzl)
        torch.manual_seed(0)
        full = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        full = full / full.norm(dim=-1, keepdim=True)
        dt, steps = 1e-14, 4

        dsys = DistributedSystem(mesh, _MAT, world, rank, h_ext=_H_EXT)
        got = dsys.relax(full[:, :, sl].clone(), steps=steps, dt=dt)

        ref = System(mesh, _MAT, demag=True, h_ext=_H_EXT).relax(full.clone(), steps=steps, dt=dt)
        torch.testing.assert_close(got, ref[:, :, sl], rtol=1e-6, atol=1e-7)
    finally:
        dist.destroy_process_group()


def _run_match(world: int, nx: int, ny: int, nz: int) -> None:
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(
            _match_worker,
            args=(world, init, nx, ny, nz),
            nprocs=world,
            join=True,
            start_method="spawn",
        )


@pytest.mark.dist
def test_distributed_solve_matches_single_process_n2() -> None:
    _run_match(world=2, nx=32, ny=32, nz=16)


@pytest.mark.dist
def test_distributed_solve_matches_single_process_n4() -> None:
    _run_match(world=4, nx=32, ny=32, nz=16)


# cross-rank finite-difference gradcheck: global cells spanning both slabs (nz=8, nzl=4)
_CHECKS = [(0, 0, 0, 0), (1, 1, 3, 2), (2, 0, 4, 1), (0, 1, 7, 0), (1, 2, 5, 2)]


def _grad_worker(rank: int, world: int, init: str) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=init)
    try:
        nx, ny, nz = 4, 4, 8
        mesh = Mesh(n=(nx, ny, nz), dx=(2e-9, 2e-9, 2e-9))
        nzl = nz // world
        sl = slice(rank * nzl, (rank + 1) * nzl)
        torch.manual_seed(0)  # same full, w on every rank
        full = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        full = full / full.norm(dim=-1, keepdim=True)
        w = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        dt, steps = 1e-14, 2
        dsys = DistributedSystem(mesh, _MAT, world, rank, h_ext=_H_EXT)

        def global_loss(m_all: torch.Tensor) -> torch.Tensor:
            out = dsys.relax(m_all[:, :, sl].contiguous(), steps=steps, dt=dt)
            return all_reduce_sum((out * w[:, :, sl]).sum())

        m_local = full[:, :, sl].clone().requires_grad_(True)
        all_reduce_sum((dsys.relax(m_local, steps=steps, dt=dt) * w[:, :, sl]).sum()).backward()
        g_auto = m_local.grad
        assert g_auto is not None

        # FD-step sweep (the multi-step relax is the STIFF loss where a single hand-picked eps
        # can sit in the cancellation-vs-truncation trough); every rank runs the full sweep --
        # no early exit -- so the collectives inside the loss stay in lockstep
        eps_sweep = (1e-5, 1e-6, 1e-7)
        for ix, iy, iz, c in _CHECKS:
            owner = iz // nzl
            fds = []
            for eps in eps_sweep:
                with torch.no_grad():
                    mp_plus, mp_minus = full.clone(), full.clone()
                    if rank == owner:
                        mp_plus[ix, iy, iz, c] += eps
                        mp_minus[ix, iy, iz, c] -= eps
                    lp, lm = float(global_loss(mp_plus)), float(global_loss(mp_minus))
                fds.append((lp - lm) / (2.0 * eps))
            if rank == owner:
                g = float(g_auto[ix, iy, iz - owner * nzl, c])
                # tolerance gate unchanged (rtol 1e-5 / atol 1e-7); the sweep relaxes only the step
                assert min(abs(fd - g) for fd in fds) <= 1e-7 + 1e-5 * abs(g), (
                    f"cell ({ix},{iy},{iz},{c}): autograd {g!r} vs FD "
                    f"{dict(zip(eps_sweep, fds, strict=True))!r}"
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.dist
def test_distributed_solve_cross_rank_gradcheck() -> None:
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(_grad_worker, args=(2, init), nprocs=2, join=True, start_method="spawn")


def _ckpt_worker(rank: int, world: int, init: str) -> None:
    """Temporal checkpointing of the distributed relax: same forward, same gradient, bounded
    extra collective traffic, and the recompute collectives stay coordinated across ranks (the
    whole run deadlocks if they don't -- lockstep is tested by termination)."""
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=init)
    try:
        nx, ny, nz = 4, 4, 8
        mesh = Mesh(n=(nx, ny, nz), dx=(2e-9, 2e-9, 2e-9))
        nzl = nz // world
        sl = slice(rank * nzl, (rank + 1) * nzl)
        torch.manual_seed(0)  # same full, w on every rank
        full = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        full = full / full.norm(dim=-1, keepdim=True)
        w = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        dt, steps = 1e-14, 4
        dsys = DistributedSystem(mesh, _MAT, world, rank, h_ext=_H_EXT)

        def run(ckpt: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
            m = full[:, :, sl].clone().requires_grad_(True)
            reset_a2a_calls()
            out = dsys.relax(m, steps=steps, dt=dt, checkpoint_every=ckpt)
            n_fwd = a2a_calls()
            reset_a2a_calls()
            all_reduce_sum((out * w[:, :, sl]).sum()).backward()
            n_bwd = a2a_calls()
            assert m.grad is not None
            return out.detach(), m.grad, n_fwd, n_bwd

        out_plain, g_plain, fwd_plain, bwd_plain = run(0)
        out_ck, g_ck, fwd_ck, bwd_ck = run(2)  # 2-step chunks

        # forward is the same computation in the same order -> bit-identical; gradient likewise
        # up to recompute round-off (same ops, tiny tolerance)
        torch.testing.assert_close(out_ck, out_plain, rtol=0.0, atol=0.0)
        torch.testing.assert_close(g_ck, g_plain, rtol=1e-12, atol=1e-14)

        # traffic pins: plain adjoint doubles the forward collectives exactly; checkpointing adds
        # recompute traffic bounded by one extra forward (> 2N, <= 3N). The realized point inside
        # that band is torch's selective-recompute behavior -- implementation-dependent, NOT a law
        # (a torch upgrade moving it within the band is fine; outside the band is a real bug).
        assert fwd_ck == fwd_plain  # checkpointing must not change forward traffic
        assert bwd_plain == fwd_plain
        assert fwd_plain < bwd_ck <= 2 * fwd_plain, (
            f"recompute band violated: fwd={fwd_plain}, ckpt bwd={bwd_ck} "
            f"(expected in (N, 2N] -- selective recompute is torch-version-dependent)"
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.dist
def test_checkpointed_relax_matches_and_bounds_traffic() -> None:
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(_ckpt_worker, args=(2, init), nprocs=2, join=True, start_method="spawn")
