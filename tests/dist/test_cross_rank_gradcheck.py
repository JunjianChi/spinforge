"""B2 coordinated cross-rank gradcheck: the distributed adjoint vs finite-difference ground truth.

test_adjoint matches the distributed gradient to the single-process autograd gradient (autograd vs
autograd). This is stronger: it finite-differences the DISTRIBUTED forward itself. One global input
cell is perturbed; every rank re-runs the distributed forward in lockstep (the transpose collective
needs all ranks), and the rank that owns the cell compares the central difference of the global loss
to its autograd gradient. That checks the whole forward+adjoint against ground truth, so a wrong
collective adjoint (the gap magnum.np.distributed leaves open) cannot pass. Runs on gloo/CPU.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from spinforge.core.mesh import Mesh
from spinforge.dist.collectives import all_reduce_sum, set_verify_conservation
from spinforge.dist.demag import DistributedDemagField

# global (ix, iy, iz, component) cells to probe -- spanning both rank slabs (iz 0..1 vs 2..3)
_CHECKS = [(0, 0, 0, 0), (1, 1, 1, 2), (1, 0, 2, 1), (0, 1, 3, 0), (1, 1, 2, 2)]

# FD step sweep: a single hand-picked eps can sit in the cancellation-vs-truncation trough for one
# cell and not another (worse for stiffer losses than this linear one). The check passes if ANY step
# reproduces autograd at the tight tolerance. Every rank runs the FULL sweep -- no early exit -- so
# the collective inside the loss stays in lockstep across ranks.
_EPS_SWEEP = (1e-4, 1e-5, 1e-6, 1e-7)


def _worker(rank: int, world_size: int, init: str, use_rfft: bool) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=init)
    set_verify_conservation(True)  # every transpose in the harness runs under the payload guard
    try:
        nx, ny, nz, ms = 2, 2, 4, 1.0  # ms=1 keeps the (linear) loss O(1) so FD is clean
        mesh = Mesh(n=(nx, ny, nz), dx=(2e-9, 2e-9, 2e-9))
        torch.manual_seed(0)  # same m_full, w on every rank
        m_full = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        m_full = m_full / m_full.norm(dim=-1, keepdim=True)
        w = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        nzl = nz // world_size
        sl = slice(rank * nzl, (rank + 1) * nzl)
        field = DistributedDemagField(mesh, world_size, rank, use_rfft=use_rfft)

        def global_loss(m_all: torch.Tensor) -> torch.Tensor:
            # every rank drives the same scalar loss; the collective inside keeps ranks in lockstep
            h_local = field(m_all[:, :, sl].contiguous(), ms)
            return all_reduce_sum((h_local * w[:, :, sl]).sum())

        # autograd gradient of the global loss w.r.t. this rank's slab
        m_local = m_full[:, :, sl].clone().requires_grad_(True)
        loss = all_reduce_sum((field(m_local, ms) * w[:, :, sl]).sum())
        loss.backward()
        g_auto = m_local.grad
        assert g_auto is not None

        for ix, iy, iz, c in _CHECKS:
            owner = iz // nzl
            fds = []
            for eps in _EPS_SWEEP:
                with torch.no_grad():
                    mp_plus = m_full.clone()
                    mp_minus = m_full.clone()
                    if rank == owner:  # only the owner perturbs; all ranks still run the forward
                        mp_plus[ix, iy, iz, c] += eps
                        mp_minus[ix, iy, iz, c] -= eps
                    lp = float(global_loss(mp_plus))
                    lm = float(global_loss(mp_minus))
                fds.append((lp - lm) / (2.0 * eps))
            if rank == owner:
                g = float(g_auto[ix, iy, iz - owner * nzl, c])
                errs = [abs(fd - g) for fd in fds]
                # explicit rtol/atol; the sweep only relaxes the FD-step choice, not the gate
                assert min(errs) <= 1e-6 + 1e-5 * abs(g), (
                    f"cell ({ix},{iy},{iz},{c}): autograd {g!r} vs FD "
                    f"{dict(zip(_EPS_SWEEP, fds, strict=True))!r}"
                )
    finally:
        set_verify_conservation(False)
        dist.destroy_process_group()


@pytest.mark.dist
def test_distributed_adjoint_matches_finite_difference() -> None:
    world_size = 2
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(
            _worker,
            args=(world_size, init, False),
            nprocs=world_size,
            join=True,
            start_method="spawn",
        )


@pytest.mark.dist
def test_rfft_distributed_adjoint_matches_finite_difference() -> None:
    """Float64 gradcheck for the rfft path: autograd's Hermitian-aware rfft2/irfft2 vjps must carry
    the
    x2 packing weight -- FD ground truth is what catches a mis-weighted adjoint, so this is the
    test that un-parks the rfft caveat (decisions 2026-07-06)."""
    world_size = 2
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(
            _worker,
            args=(world_size, init, True),
            nprocs=world_size,
            join=True,
            start_method="spawn",
        )
