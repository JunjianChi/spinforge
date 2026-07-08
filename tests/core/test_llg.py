"""LLG integrator oracles: Larmor precession frequency, and damped relaxation to the easy axis."""

from __future__ import annotations

import math

import torch

from spinforge.core.constants import GAMMA
from spinforge.core.mesh import Mesh
from spinforge.core.system import Material, System


def test_larmor_precession_frequency() -> None:
    # Free spin (no damping) in H = Hz z: precesses in-plane at omega = GAMMA*Hz, mx=cos, my=sin.
    mesh = Mesh(n=(1, 1, 1), dx=(1e-9, 1e-9, 1e-9))
    sys = System(mesh, Material(ms=8e5, alpha=0.0), demag=False, h_ext=(0.0, 0.0, 1e5))
    omega = GAMMA * 1e5
    period = 2.0 * math.pi / omega
    dt = period / 2000.0
    m = torch.tensor([[[[1.0, 0.0, 0.0]]]], dtype=torch.float64)
    n_steps = 500
    m = sys.relax(m, steps=n_steps, dt=dt)
    t = n_steps * dt
    torch.testing.assert_close(float(m[0, 0, 0, 0]), math.cos(omega * t), rtol=0.0, atol=2e-3)
    torch.testing.assert_close(float(m[0, 0, 0, 1]), math.sin(omega * t), rtol=0.0, atol=2e-3)


def test_relaxes_to_easy_axis() -> None:
    # High damping + uniaxial easy-axis z: a tilted state must relax toward +z (not the hard plane).
    mesh = Mesh(n=(4, 4, 1), dx=(2e-9, 2e-9, 2e-9))
    mat = Material(ms=8e5, a_ex=1.3e-11, ku=5.0e5, ku_axis=(0.0, 0.0, 1.0), alpha=1.0)
    sys = System(mesh, mat, demag=False)
    m = torch.zeros(4, 4, 1, 3, dtype=torch.float64)
    m[..., 0], m[..., 2] = 0.6, 0.8  # tilted from the easy axis
    m = sys.relax(m, steps=4000, dt=1e-13)
    ones = torch.ones(4, 4, 1, dtype=torch.float64)
    torch.testing.assert_close(m[..., 2], ones, rtol=0.0, atol=1e-3)
