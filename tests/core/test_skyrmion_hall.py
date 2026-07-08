"""Skyrmion-Hall / Thiele (reduced, slow): a current-driven skyrmion deflects at
tan(theta_H)=alpha*D/G. Full alpha-sweep + figure are in experiments/skyrmion_hall.py +
results/skyrmion_hall.md; this coarse CPU version guards the STT drive + Thiele relation, no GPU."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from spinforge.core.mesh import Mesh
from spinforge.core.system import Material, System
from spinforge.core.textures import skyrmion
from spinforge.io.render import render_on_failure

MS, A_EX, D_DMI = 3.84e5, 8.78e-12, 1.58e-3


def _central(m: torch.Tensor, mesh: Mesh, ax: int) -> torch.Tensor:
    n = mesh.n[ax]
    mp = torch.cat([m.narrow(ax, 0, 1), m, m.narrow(ax, n - 1, 1)], dim=ax)
    return (mp.narrow(ax, 2, n) - mp.narrow(ax, 0, n)) / (2.0 * mesh.dx[ax])


@pytest.mark.slow
def test_skyrmion_hall_angle_matches_thiele(artifacts_dir: Path) -> None:
    n = 60
    mesh = Mesh(n=(n, n, 1), dx=(2e-9, 2e-9, 2e-9))
    hz = 0.4 / (4.0 * math.pi * 1e-7)
    alpha, u0 = 0.3, 100.0

    # relax a skyrmion in a clean (replicate-BC) uniform background
    relax = System(mesh, Material(ms=MS, a_ex=A_EX, d=D_DMI, alpha=1.0), demag=True,
                   h_ext=(0, 0, hz), chiral_bc=False)  # fmt: skip
    m = skyrmion(mesh, radius=n * 2e-9 * 0.22, chirality=math.pi / 2)
    m = relax.relax(m, steps=2500, dt=2e-13)

    # Thiele tensors over a central window (dA cancels in the alpha*D/G ratio)
    w = slice(n // 4, 3 * n // 4)
    dxm, dym = _central(m, mesh, 0)[w, w], _central(m, mesh, 1)[w, w]
    mm = m[w, w]
    d_tensor = 0.5 * float((dxm * dxm).sum() + (dym * dym).sum())
    g = float((mm * torch.linalg.cross(dxm, dym, dim=-1)).sum())
    th_pred = math.degrees(math.atan(alpha * d_tensor / abs(g)))

    # drive with an adiabatic STT current in +x; track the core centroid
    drive = System(mesh, Material(ms=MS, a_ex=A_EX, d=D_DMI, alpha=alpha), demag=True,
                   h_ext=(0, 0, hz), u=(u0, 0.0, 0.0), chiral_bc=False)  # fmt: skip
    xs = (torch.arange(n, dtype=torch.float64) + 0.5) * 2e-9
    xg, yg = torch.meshgrid(xs, xs, indexing="ij")

    def centroid(mm: torch.Tensor) -> tuple[float, float]:
        wt = torch.relu(-mm[:, :, 0, 2])
        s = wt.sum()
        return float((wt * xg).sum() / s), float((wt * yg).sum() / s)

    c0 = centroid(m)
    for _ in range(800):
        m = drive.step_rk4(m, 1e-13)
    c1 = centroid(m)
    vx, vy = c1[0] - c0[0], c1[1] - c0[1]
    th_meas = math.degrees(math.atan2(vy, vx))

    with render_on_failure(m, "skyrmion_hall_drive", artifacts_dir):
        assert vx > 0, "skyrmion should drift along the current (+x)"
        assert vy < 0, "Q<0 gyrovector deflects to -y (same-sign across the population)"
        assert abs(abs(th_meas) - th_pred) < 0.2 * th_pred, (
            f"Hall angle {abs(th_meas):.1f} deg off Thiele {th_pred:.1f} deg"
        )
