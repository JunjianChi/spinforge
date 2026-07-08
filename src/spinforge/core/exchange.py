"""Exchange field: H_exch = (2 A / (mu0 Ms)) * laplacian(m), Neumann (zero-flux) boundaries."""

from __future__ import annotations

import torch
from scipy import constants

from .mesh import Mesh
from .stencil import replicate_pad, second_diff


def _laplacian(m: torch.Tensor, mesh: Mesh) -> torch.Tensor:
    """6-neighbour Laplacian of ``(nx, ny, nz, 3)``; replicate (zero-flux) boundaries."""
    lap = torch.zeros_like(m)
    for ax in range(3):
        if mesh.n[ax] == 1:
            continue
        lap = lap + second_diff(replicate_pad(m, ax), ax) / mesh.dx[ax] ** 2
    return lap


def exchange_field(m: torch.Tensor, mesh: Mesh, a_ex: float, ms: float) -> torch.Tensor:
    """Exchange field (A/m) for stiffness ``a_ex`` (J/m) and saturation ``ms`` (A/m)."""
    return (2.0 * a_ex / (constants.mu_0 * ms)) * _laplacian(m, mesh)
