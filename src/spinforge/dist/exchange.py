"""Distributed exchange field: local x,y Laplacian + a z-Laplacian that reads the halo ghost layer.

x and y are fully local (each rank owns the whole xy-plane); only z is slab-distributed, so the
z second-difference at a slab edge reads the neighbour's ghost layer from ``z_halo_exchange``. The
result matches the single-process ``exchange_field`` on each rank's slab, and the gradient crosses
ranks through the halo's adjoint -- so the LOCAL terms of a distributed LLG step are differentiable
too (the demag transpose-FFT covers the long-range term).
"""

from __future__ import annotations

import torch
from scipy import constants

from ..core.mesh import Mesh
from ..core.stencil import replicate_pad, second_diff
from .halo import z_halo_exchange


def distributed_exchange_field(
    m_local: torch.Tensor, mesh: Mesh, a: float, ms: float, world: int, rank: int
) -> torch.Tensor:
    """Exchange field on this rank's z-slab; ``m_local`` is ``[nx, ny, nz_local, 3]``."""
    lap = torch.zeros_like(m_local)
    for ax, h in ((0, mesh.dx[0]), (1, mesh.dx[1])):  # x, y are local (replicate / Neumann)
        lap = lap + second_diff(replicate_pad(m_local, ax), ax) / h**2
    if mesh.n[2] > 1:  # z is slab-distributed: the 2nd difference reads the neighbour ghost layers
        lap = lap + second_diff(z_halo_exchange(m_local, world, rank), 2) / mesh.dx[2] ** 2
    return (2.0 * a / (constants.mu_0 * ms)) * lap
