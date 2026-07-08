"""Chiral exchange: exchange + bulk DMI sharing one consistent free-surface boundary condition.

Bulk (B20) DMI energy ``D m.(curl m)`` and exchange ``A|grad m|^2`` are not separable at a free
surface: minimizing their sum gives a single natural boundary condition, ``dm/dn = (D/2A)(n x m)``
(``n`` the outward normal). The standalone ``exchange_field`` (Neumann) and ``bulk_dmi_field``
(replicate) impose it inconsistently, which cants the edges. This operator pads BOTH the exchange
Laplacian and the DMI curl with the SAME chiral ghost cells, so the boundary is treated correctly.
At ``D = 0`` the ghost is the replicate (Neumann) cell, recovering the pure-exchange field.

Ref: free-surface DMI BC, A. N. Bogdanov / S. Rohart & A. Thiaville, PRB 88, 184422 (2013)
(interfacial form); the bulk-DMI analogue used here is ``dm/dn = (D/2A)(n x m)``.
"""

from __future__ import annotations

import torch
from scipy import constants

from .mesh import Mesh
from .stencil import central_diff, second_diff


def chiral_ghost(edge: torch.Tensor, ax: int, dx: float, xi: float, outward: int) -> torch.Tensor:
    """One chiral-BC ghost layer for an ``ax``-face edge layer: ``dm/dn = xi*(n_out x m_edge)``.

    ``outward = -1`` for the low face (normal ``-e_ax``), ``+1`` for the high face. At ``xi = 0``
    the ghost is the edge cell itself (replicate / Neumann). Pure tensor math, MPI-free: the
    distributed field calls this on the ranks owning the global z surfaces.
    """
    e = torch.zeros_like(edge)
    e[..., ax] = 1.0
    return edge + outward * dx * xi * torch.linalg.cross(e, edge, dim=-1)


def _pad_chiral(m: torch.Tensor, mesh: Mesh, ax: int, xi: float) -> torch.Tensor:
    """Pad ``m`` along ``ax`` with chiral-BC ghost cells on both faces.

    Sizes come from ``m.shape`` (never ``mesh.n``): ``m`` may be a z-slab of the global mesh.
    """
    n, dx = m.shape[ax], mesh.dx[ax]
    lo, hi = m.narrow(ax, 0, 1), m.narrow(ax, n - 1, 1)
    return torch.cat(
        [chiral_ghost(lo, ax, dx, xi, -1), m, chiral_ghost(hi, ax, dx, xi, +1)], dim=ax
    )


def chiral_exchange_dmi_field(
    m: torch.Tensor,
    mesh: Mesh,
    *,
    a: float,
    d: float,
    ms: float,
    pad_z: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exchange + bulk-DMI field (A/m) with the coupled free-surface BC (a: J/m, d: J/m^2).

    ``pad_z``: caller-owned z padding (``m`` with one ghost layer on each z face), for the
    distributed z-slab case -- the halo supplies real neighbour layers at interior slab
    boundaries and the surface-owning ranks supply the chiral ghost at the global faces. When
    omitted, both z ghosts are the chiral BC (the single-process case). All axis extents are
    taken from ``m.shape``, never from ``mesh.n``: a slab's z extent differs from the mesh.
    """
    xi = d / (2.0 * a)
    lap = torch.zeros_like(m)
    deriv: list[torch.Tensor] = [torch.zeros_like(m) for _ in range(3)]
    for ax in range(3):
        n = m.shape[ax]
        if n == 1:
            continue
        dx = mesh.dx[ax]
        if ax == 2 and pad_z is not None:
            if pad_z.shape[2] != n + 2:
                raise ValueError(f"pad_z z-extent {pad_z.shape[2]} != m z-extent {n} + 2")
            mp = pad_z
        else:
            mp = _pad_chiral(m, mesh, ax, xi)
        lap = lap + second_diff(mp, ax) / dx**2
        deriv[ax] = central_diff(mp, ax) / (2.0 * dx)

    dx_m, dy_m, dz_m = deriv
    curl = torch.stack(
        [
            dy_m[..., 2] - dz_m[..., 1],
            dz_m[..., 0] - dx_m[..., 2],
            dx_m[..., 1] - dy_m[..., 0],
        ],
        dim=-1,
    )
    pref = 1.0 / (constants.mu_0 * ms)
    return pref * (2.0 * a * lap - 2.0 * d * curl)
