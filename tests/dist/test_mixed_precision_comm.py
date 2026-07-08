"""Two-tier correctness for the mixed-precision-comm collective (gloo/CPU; the f32-wire row).

The collective's differentiability is inherited (it is casts composed around the already-gradchecked
all_to_all); what is NEW is the numerical effect of the f32 wire round-trip. Validated in the real
use -- injected into the distributed demag behind the swappable-backend collective seam -- two
tiers:

  Tier 1 (exact, STRICT): the f64-wire demag's adjoint matches a coordinated cross-rank finite
    difference at strict f64 tolerance (the canonical eps-swept harness lives in
    test_cross_rank_gradcheck.py; tier 1 repeats its single-eps form as the contrast anchor).
  Tier 2 (mixed, RELAXED): the f32-wire demag's adjoint matches the coordinated FD at a tolerance
    matched to the f32 round-trip (large eps: demag is linear, so FD truncation is ~0 and a coarse
    eps only lifts the f32 loss noise out of the difference). AND the mixed op's forward and
    gradient
    error vs the exact op is measured and bounded by ~u_f32 -- confirming it is genuinely
    reduced-precision (not secretly exact) and no worse than f32-scale (demag linear + self-adjoint,
    kappa~1, so the error does not amplify).

The bandwidth win (half the NVLink bytes on the wire) is device-blocked -> measured on real
multi-GPU hardware; the f32 round-trip's numerical effect is device-independent and fully exercised
here. Same seed on every rank so the coordinated FD stays in lockstep.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from spinforge.core.mesh import Mesh
from spinforge.dist.collectives import (
    SyncCollectiveA2A,
    all_reduce_sum,
    all_to_all_complex_mixed,
)
from spinforge.dist.demag import DistributedDemagField

U_F32 = 2**-24  # f32 unit roundoff ~5.96e-8: the expected error scale of one f64<->f32 round-trip
_CHECKS = [(0, 0, 0, 0), (1, 1, 1, 2), (1, 0, 2, 1), (0, 1, 3, 0), (1, 1, 2, 2)]


def _worker(rank: int, world_size: int, init: str) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=init)
    try:
        nx, ny, nz, ms = 2, 2, 4, 1.0  # ms=1 keeps the (linear) loss O(1) so FD is clean
        mesh = Mesh(n=(nx, ny, nz), dx=(2e-9, 2e-9, 2e-9))
        torch.manual_seed(0)  # same m_full, w on every rank (lockstep FD)
        m_full = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        m_full = m_full / m_full.norm(dim=-1, keepdim=True)
        w = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        nzl = nz // world_size
        sl = slice(rank * nzl, (rank + 1) * nzl)
        field_exact = DistributedDemagField(mesh, world_size, rank)
        field_mixed = DistributedDemagField(
            mesh, world_size, rank, backend=SyncCollectiveA2A(all_to_all_complex_mixed)
        )

        def global_loss(m_all: torch.Tensor, field: DistributedDemagField) -> torch.Tensor:
            h_local = field(m_all[:, :, sl].contiguous(), ms)
            return all_reduce_sum((h_local * w[:, :, sl]).sum())

        # autograd gradients of both demags (separate graphs)
        def autograd_grad(field: DistributedDemagField) -> torch.Tensor:
            m_local = m_full[:, :, sl].clone().requires_grad_(True)
            all_reduce_sum((field(m_local, ms) * w[:, :, sl]).sum()).backward()
            assert m_local.grad is not None
            return m_local.grad

        g_exact, g_mixed = autograd_grad(field_exact), autograd_grad(field_mixed)

        # forward + gradient error of the mixed op vs the exact op (this rank's slab)
        with torch.no_grad():
            h_exact = field_exact(m_full[:, :, sl].contiguous(), ms)
            h_mixed = field_mixed(m_full[:, :, sl].contiguous(), ms)
        fwd_rel = (h_mixed - h_exact).norm().item() / h_exact.norm().item()
        grad_rel = (g_mixed - g_exact).norm().item() / g_exact.norm().item()

        # coordinated cross-rank FD gradcheck: exact strict + mixed relaxed (all ranks lockstep)
        for eps, field, g_auto, rtol, atol in (
            (1e-6, field_exact, g_exact, 1e-5, 1e-6),  # tier 1: strict f64
            (1e-3, field_mixed, g_mixed, 2e-2, 1e-3),  # tier 2: relaxed to the f32 round-trip
        ):
            for ix, iy, iz, c in _CHECKS:
                owner = iz // nzl
                with torch.no_grad():
                    mp_plus, mp_minus = m_full.clone(), m_full.clone()
                    if rank == owner:  # only the owner perturbs; all ranks run the forward
                        mp_plus[ix, iy, iz, c] += eps
                        mp_minus[ix, iy, iz, c] -= eps
                    lp = float(global_loss(mp_plus, field))
                    lm = float(global_loss(mp_minus, field))
                fd = (lp - lm) / (2.0 * eps)
                if rank == owner:
                    g = float(g_auto[ix, iy, iz - owner * nzl, c])
                    torch.testing.assert_close(
                        torch.tensor(fd), torch.tensor(g), rtol=rtol, atol=atol
                    )

        # asserts last (no collectives after -> no divergent-failure deadlock): genuinely f32-lossy
        # (> f64 eps, so the downcast really happened) and no worse than a few x u_f32 through the
        # FFT
        assert 1e-10 < fwd_rel < 1e-5, f"fwd_rel={fwd_rel:.3e}"
        assert 1e-10 < grad_rel < 1e-5, f"grad_rel={grad_rel:.3e}"
        if rank == 0:
            print(
                f"\n[mixed-precision-comm] fwd_rel={fwd_rel:.3e} grad_rel={grad_rel:.3e} "
                f"(u_f32={U_F32:.3e}); "
                "coordinated FD gradcheck passed (exact strict, mixed relaxed)"
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.dist
def test_mixed_precision_comm_two_tier_correctness() -> None:
    world_size = 2
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(
            _worker, args=(world_size, init), nprocs=world_size, join=True, start_method="spawn"
        )
