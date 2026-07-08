"""Adiabatic STT term: the directional derivative matches an analytic spiral, gradchecks, and
vanishes at zero current (so the LLG reduces to the field-only dynamics)."""

from __future__ import annotations

import math

import torch
from torch.autograd import gradcheck

from spinforge.core.mesh import Mesh
from spinforge.core.spin_torque import u_grad_m
from spinforge.core.system import Material, System


def test_u_grad_m_matches_analytic_spiral() -> None:
    # m(x) = (cos kx, sin kx, 0): (u.grad)m = u_x * k (-sin kx, cos kx, 0)
    n, dx = 64, 1e-9
    mesh = Mesh(n=(n, 1, 1), dx=(dx, dx, dx))
    k = 2.0 * math.pi / (16 * dx)
    x = (torch.arange(n, dtype=torch.float64) + 0.5) * dx
    m = torch.zeros(n, 1, 1, 3, dtype=torch.float64)
    m[..., 0], m[..., 1] = torch.cos(k * x)[:, None, None], torch.sin(k * x)[:, None, None]
    u0 = 50.0
    got = u_grad_m(m, mesh, (u0, 0.0, 0.0))
    want = torch.zeros_like(m)
    want[..., 0] = (u0 * k * -torch.sin(k * x))[:, None, None]
    want[..., 1] = (u0 * k * torch.cos(k * x))[:, None, None]
    # central diff is O(dx^2); compare the interior (boundaries use replicate)
    torch.testing.assert_close(got[2:-2], want[2:-2], rtol=2e-2, atol=1e-2 * u0 * k)


def test_u_grad_m_gradcheck() -> None:
    mesh = Mesh(n=(5, 4, 1), dx=(2e-9, 2e-9, 2e-9))
    m = torch.randn(5, 4, 1, 3, dtype=torch.float64, requires_grad=True)
    assert gradcheck(lambda x: u_grad_m(x, mesh, (30.0, -10.0, 0.0)), (m,), eps=1e-6, atol=1e-6)


def test_zero_current_recovers_field_dynamics() -> None:
    mesh = Mesh(n=(6, 6, 1), dx=(2e-9, 2e-9, 2e-9))
    mat = Material(ms=3.84e5, a_ex=8.78e-12, d=1.58e-3, alpha=0.3)
    no_u = System(mesh, mat, demag=True, h_ext=(0.0, 0.0, 1e5))
    with_u0 = System(mesh, mat, demag=True, h_ext=(0.0, 0.0, 1e5), u=(0.0, 0.0, 0.0))
    m = torch.randn(6, 6, 1, 3, dtype=torch.float64)
    m = m / m.norm(dim=-1, keepdim=True)
    # adding an exactly-zero STT term is bitwise neutral -> bit-identity IS the claim
    torch.testing.assert_close(no_u.llg_rhs(m), with_u0.llg_rhs(m), rtol=0.0, atol=0.0)


def test_llg_with_stt_gradcheck() -> None:
    mesh = Mesh(n=(5, 5, 1), dx=(2e-9, 2e-9, 2e-9))
    mat = Material(ms=3.84e5, a_ex=8.78e-12, d=1.58e-3, alpha=0.3)
    sys = System(mesh, mat, demag=True, h_ext=(0.0, 0.0, 1e5), u=(40.0, 0.0, 0.0))

    def solve(m0: torch.Tensor) -> torch.Tensor:
        m = m0
        for _ in range(2):
            m = sys.step_rk4(m, 1e-13)
        return m

    m0 = torch.randn(5, 5, 1, 3, dtype=torch.float64)
    m0 = (m0 / m0.norm(dim=-1, keepdim=True)).requires_grad_(True)
    assert gradcheck(solve, (m0,), eps=1e-6, atol=1e-4, rtol=1e-3)
