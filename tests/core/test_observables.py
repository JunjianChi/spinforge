"""Oracle: the topological charge of a single skyrmion is Q = -1."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from spinforge.core.mesh import Mesh
from spinforge.core.observables import topological_charge
from spinforge.io.render import render_on_failure


def _neel_skyrmion(mesh: Mesh, radius: float) -> torch.Tensor:
    """Analytic Neel skyrmion: core down at the centre, up far away, winding +1 -> Q = -1."""
    nx, ny, _ = mesh.n
    xs = (torch.arange(nx, dtype=torch.float64) + 0.5) * mesh.dx[0]
    ys = (torch.arange(ny, dtype=torch.float64) + 0.5) * mesh.dx[1]
    cx, cy = nx * mesh.dx[0] / 2, ny * mesh.dx[1] / 2
    x, y = torch.meshgrid(xs - cx, ys - cy, indexing="ij")
    r = torch.sqrt(x**2 + y**2)
    phi = torch.atan2(y, x)
    theta = (math.pi * (1.0 - r / radius)).clamp(0.0, math.pi)
    m = torch.empty(nx, ny, 1, 3, dtype=torch.float64)
    m[..., 0, 0] = torch.sin(theta) * torch.cos(phi)
    m[..., 0, 1] = torch.sin(theta) * torch.sin(phi)
    m[..., 0, 2] = torch.cos(theta)
    return m


def test_single_skyrmion_charge_is_minus_one(artifacts_dir: Path) -> None:
    mesh = Mesh(n=(64, 64, 1), dx=(1e-9, 1e-9, 1e-9))
    m = _neel_skyrmion(mesh, radius=24e-9)
    q = topological_charge(m)
    with render_on_failure(m, "skyrmion_q", artifacts_dir):
        # Berg-Luescher is exact to roundoff (measured error ~2e-16); 1e-14 leaves wide margin
        torch.testing.assert_close(q, torch.tensor(-1.0, dtype=torch.float64), rtol=0.0, atol=1e-14)
