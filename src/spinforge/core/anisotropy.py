"""Uniaxial anisotropy field: H = (2 Ku/(mu0 Ms)) (m . u) u, easy axis u (energy -Ku (m.u)^2)."""

from __future__ import annotations

import torch
from scipy import constants

from .mesh import Mesh


def uniaxial_anisotropy_field(
    m: torch.Tensor,
    mesh: Mesh,
    ku: float | torch.Tensor,
    axis: tuple[float, float, float],
    ms: float,
) -> torch.Tensor:
    """Uniaxial anisotropy field (A/m) for ``ku`` (J/m^3) and easy ``axis``.

    ``ku`` may be a spatial tensor broadcastable against ``m[..., :1]`` (e.g. ``[nx, ny, 1, 1]``)
    -- a graded-anisotropy design profile; the field is then differentiable w.r.t. the profile.
    """
    del mesh  # local term; signature kept uniform with the stencil field terms
    u = torch.tensor(axis, dtype=m.dtype, device=m.device)
    u = u / u.norm()
    m_dot_u = (m * u).sum(-1, keepdim=True)
    return (2.0 * ku / (constants.mu_0 * ms)) * m_dot_u * u
