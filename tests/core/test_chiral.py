"""Chiral exchange+DMI operator: reduces to the decoupled terms in the interior, gradchecks,
and at D=0 recovers pure exchange. The boundary differs by construction (the free-surface BC)."""

from __future__ import annotations

import torch
from torch.autograd import gradcheck

from spinforge.core.chiral import chiral_exchange_dmi_field
from spinforge.core.dmi import bulk_dmi_field
from spinforge.core.exchange import exchange_field
from spinforge.core.mesh import Mesh


def _rand_m(n: tuple[int, int, int]) -> torch.Tensor:
    m = torch.randn(*n, 3, dtype=torch.float64)
    return m / m.norm(dim=-1, keepdim=True)


def test_chiral_recovers_exchange_at_zero_dmi() -> None:
    mesh = Mesh(n=(6, 5, 4), dx=(2e-9, 2e-9, 2e-9))
    m = _rand_m(mesh.n)
    got = chiral_exchange_dmi_field(m, mesh, a=1.3e-11, d=0.0, ms=8e5)
    ref = exchange_field(m, mesh, 1.3e-11, 8e5)
    torch.testing.assert_close(got, ref, rtol=1e-10, atol=1e-6)


def test_chiral_matches_decoupled_in_interior() -> None:
    # away from the boundary the stencil is identical regardless of the ghost cells, so the
    # chiral field equals exchange + bulk DMI there; only the edge layer carries the BC.
    mesh = Mesh(n=(8, 7, 6), dx=(2e-9, 2e-9, 2e-9))
    m = _rand_m(mesh.n)
    a, d, ms = 1.3e-11, 1.5e-3, 8e5
    got = chiral_exchange_dmi_field(m, mesh, a=a, d=d, ms=ms)
    ref = exchange_field(m, mesh, a, ms) + bulk_dmi_field(m, mesh, d, ms)
    torch.testing.assert_close(got[1:-1, 1:-1, 1:-1], ref[1:-1, 1:-1, 1:-1], rtol=1e-9, atol=1e-3)


def test_chiral_gradcheck() -> None:
    # the field is ~1e7 A/m, so check an O(1)-scaled output (linear scaling preserves the
    # gradient-correctness test) -- otherwise finite-diff round-off swamps the default atol.
    mesh = Mesh(n=(5, 4, 3), dx=(2e-9, 2e-9, 2e-9))
    m = _rand_m(mesh.n).requires_grad_(True)
    assert gradcheck(
        lambda x: 1e-6 * chiral_exchange_dmi_field(x, mesh, a=1.3e-11, d=1.5e-3, ms=8e5),
        (m,),
        eps=1e-6,
        atol=1e-6,
        rtol=1e-4,
    )


def test_chiral_pad_z_slab_equals_full_grid() -> None:
    """pad_z: caller-owned z ghosts (the distributed halo's contract). A z-slab evaluated with the
    true neighbour layer (interior boundary) or the chiral ghost (global surface) must reproduce
    the full-grid field exactly -- and every axis size must come from m.shape, NOT mesh.n (the
    slab's z extent differs from the global mesh; a mesh.n[2] read passes world=1 and fails only
    distributed)."""
    from spinforge.core.chiral import chiral_ghost

    torch.manual_seed(0)
    nx, ny, nz = 4, 5, 8
    mesh = Mesh(n=(nx, ny, nz), dx=(1e-9, 2e-9, 3e-9))
    a, d, ms = 1.3e-11, 1.5e-3, 8e5
    xi, dz = d / (2.0 * a), mesh.dx[2]
    m = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
    m = m / m.norm(dim=-1, keepdim=True)
    ref = chiral_exchange_dmi_field(m, mesh, a=a, d=d, ms=ms)

    for lo_z, hi_z in ((0, 4), (4, 8)):
        slab = m[:, :, lo_z:hi_z]
        g_lo = (
            chiral_ghost(slab.narrow(2, 0, 1), 2, dz, xi, -1)
            if lo_z == 0
            else m[:, :, lo_z - 1 : lo_z]
        )
        g_hi = (
            chiral_ghost(slab.narrow(2, slab.shape[2] - 1, 1), 2, dz, xi, +1)
            if hi_z == nz
            else m[:, :, hi_z : hi_z + 1]
        )
        pad = torch.cat([g_lo, slab, g_hi], dim=2)
        got = chiral_exchange_dmi_field(slab, mesh, a=a, d=d, ms=ms, pad_z=pad)
        torch.testing.assert_close(got, ref[:, :, lo_z:hi_z], rtol=0.0, atol=0.0)
