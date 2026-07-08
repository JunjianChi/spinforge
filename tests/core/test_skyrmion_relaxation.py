"""Integration oracle (slow): exchange + bulk DMI + demag + Zeeman together host a stable skyrmion.

A Bloch skyrmion in a FeGe-class chiral film under a perpendicular bias relaxes to a stable,
localized equilibrium: core down, a +z background ring around it, converged. The box-integrated Q is
substantial but not exactly -1 -- with the physically-correct free-surface DMI BC (the default; see
``core/chiral.py``) the boundary carries the real chiral surface twist, which reduces |Q| below 1.
That is genuine physics, not an artifact (``results/dmi_surface_twist.md``); the clamped-edge
"clean -1" of a replicate boundary is the artifact.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from spinforge.core.mesh import Mesh
from spinforge.core.observables import topological_charge
from spinforge.core.system import Material, System
from spinforge.core.textures import skyrmion
from spinforge.io.render import render_on_failure


@pytest.mark.slow
def test_chiral_film_hosts_a_stable_skyrmion(artifacts_dir: Path) -> None:
    n, dx = 50, 2e-9
    mesh = Mesh(n=(n, n, 1), dx=(dx, dx, dx))
    hz = 0.4 / (4.0 * math.pi * 1e-7)  # 0.4 T perpendicular bias
    mat = Material(ms=3.84e5, a_ex=8.78e-12, d=1.58e-3, alpha=1.0)  # FeGe-class, bulk DMI
    sys = System(mesh, mat, demag=True, h_ext=(0.0, 0.0, hz))

    m = skyrmion(mesh, radius=n * dx * 0.35, chirality=math.pi / 2)
    q0 = topological_charge(m)
    torch.testing.assert_close(float(q0), -1.0, rtol=0.0, atol=1e-2)

    m = sys.relax(m, steps=4000, dt=2e-13)
    m_next = sys.relax(m, steps=200, dt=2e-13)
    converged = float((m_next - m).abs().max())
    m = m_next

    with render_on_failure(m, "chiral_film_skyrmion", artifacts_dir):
        assert converged < 1e-3, f"relaxation not converged: max|dm| = {converged}"
        assert float(m[n // 2, n // 2, 0, 2]) < -0.9, "skyrmion core should point down"
        assert float(m[..., 2].max()) > 0.9, "a +z ring exists between core and edge twist"
        core = int((m[..., 2] < 0).sum())
        assert 0 < core < n * n, f"core must be localized, not empty or box-filling: {core}"
        assert -1.15 < float(topological_charge(m)) < -0.6, "topological content of one skyrmion"
