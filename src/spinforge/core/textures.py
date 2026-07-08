"""Analytic spin textures for initialization."""

from __future__ import annotations

import math

import torch

from .mesh import Mesh


def skyrmion(mesh: Mesh, radius: float, chirality: float = 0.0) -> torch.Tensor:
    """A skyrmion ansatz centred in the xy-plane: core down, background up, winding +1 (Q = -1).

    ``chirality`` = 0 gives a Neel skyrmion, pi/2 a Bloch skyrmion (the bulk-DMI case). The profile
    is constant along z (a straight skyrmion string for nz > 1).
    """
    nx, ny, nz = mesh.n
    xs = (torch.arange(nx, dtype=torch.float64) + 0.5) * mesh.dx[0]
    ys = (torch.arange(ny, dtype=torch.float64) + 0.5) * mesh.dx[1]
    cx, cy = nx * mesh.dx[0] / 2.0, ny * mesh.dx[1] / 2.0
    x, y = torch.meshgrid(xs - cx, ys - cy, indexing="ij")
    r = torch.sqrt(x * x + y * y)
    phi = torch.atan2(y, x)
    theta = (math.pi * (1.0 - r / radius)).clamp(0.0, math.pi)
    plane = torch.sin(theta)
    m = torch.empty(nx, ny, 1, 3, dtype=torch.float64)
    m[..., 0, 0] = plane * torch.cos(phi + chirality)
    m[..., 0, 1] = plane * torch.sin(phi + chirality)
    m[..., 0, 2] = torch.cos(theta)
    return m.expand(nx, ny, nz, 3).contiguous()
