"""Bulk (B20) DMI field: H = -(2 D/(mu0 Ms)) curl(m).

Central-difference curl with replicate boundaries -- the standalone term with a Neumann-like edge.
For edge-accurate work the bulk DMI shares ONE free-surface boundary condition with exchange; that
coupled operator lives in ``core/chiral.py`` and is what ``System`` uses when both A and D are on.
This standalone field is kept for the DMI-only case and as a reference.
"""

from __future__ import annotations

import torch
from scipy import constants

from .mesh import Mesh


def _central_diff(m: torch.Tensor, mesh: Mesh, ax: int) -> torch.Tensor:
    """Central derivative d m / d x_ax of each component; replicate-padded (Neumann-like) edge."""
    n = mesh.n[ax]
    if n == 1:
        return torch.zeros_like(m)
    lo, hi = m.narrow(ax, 0, 1), m.narrow(ax, n - 1, 1)
    mp = torch.cat([lo, m, hi], dim=ax)
    return (mp.narrow(ax, 2, n) - mp.narrow(ax, 0, n)) / (2.0 * mesh.dx[ax])


def _curl(m: torch.Tensor, mesh: Mesh) -> torch.Tensor:
    dx_m, dy_m, dz_m = (_central_diff(m, mesh, ax) for ax in range(3))
    cx = dy_m[..., 2] - dz_m[..., 1]
    cy = dz_m[..., 0] - dx_m[..., 2]
    cz = dx_m[..., 1] - dy_m[..., 0]
    return torch.stack([cx, cy, cz], dim=-1)


def bulk_dmi_field(m: torch.Tensor, mesh: Mesh, d: float, ms: float) -> torch.Tensor:
    """Bulk DMI field (A/m) for DMI constant ``d`` (J/m^2) and saturation ``ms`` (A/m)."""
    return -(2.0 * d / (constants.mu_0 * ms)) * _curl(m, mesh)
