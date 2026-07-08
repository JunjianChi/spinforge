"""Topological charge via the Berg-Luscher lattice formula (exact integer for a closed texture)."""

from __future__ import annotations

import math

import torch


def _triangle_solid_angle(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Signed solid angle subtended by three unit vectors (per lattice triangle)."""
    triple = (a * torch.linalg.cross(b, c, dim=-1)).sum(-1)
    denom = 1.0 + (a * b).sum(-1) + (b * c).sum(-1) + (c * a).sum(-1)
    return 2.0 * torch.atan2(triple, denom)


def topological_charge(m: torch.Tensor) -> torch.Tensor:
    """Skyrmion number ``Q = (1/4pi) * sum of plaquette solid angles`` (Berg-Luscher).

    ``m`` is a tensor of (not necessarily exactly unit) vectors with shape ``(nx, ny, nz, 3)``
    or ``(nx, ny, 3)``; a 3D field is summed over its z-layers. The vectors are normalized
    before the solid-angle sum.
    """
    if m.dim() == 4:
        return torch.stack([topological_charge(m[:, :, k, :]) for k in range(m.shape[2])]).sum()
    m = m / m.norm(dim=-1, keepdim=True)
    a, b, c, d = m[:-1, :-1], m[1:, :-1], m[1:, 1:], m[:-1, 1:]
    plaquette = _triangle_solid_angle(a, b, c) + _triangle_solid_angle(a, c, d)
    return plaquette.sum() / (4.0 * math.pi)
