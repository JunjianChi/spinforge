"""Differentiability: autograd through the demag op and a short LLG solve, vs finite diff.

The whole project rests on the adjoint, so the single-GPU reference must be differentiable and its
gradients must match finite differences (float64 gradcheck) before any inverse-design use.
"""

from __future__ import annotations

import torch
from torch.autograd import gradcheck

from spinforge.core.demag import DemagField
from spinforge.core.mesh import Mesh
from spinforge.core.system import Material, System


def test_demag_field_gradcheck() -> None:
    # Ms = 1 so the field is O(1) and gradcheck tolerances are natural (demag is linear in Ms).
    mesh = Mesh(n=(4, 4, 2), dx=(2e-9, 2e-9, 2e-9))
    demag = DemagField(mesh)
    m = torch.randn(4, 4, 2, 3, dtype=torch.float64, requires_grad=True)
    assert gradcheck(lambda x: demag(x, 1.0), (m,), eps=1e-6, atol=1e-6, rtol=1e-4)


def test_short_llg_solve_gradcheck() -> None:
    mesh = Mesh(n=(4, 4, 1), dx=(2e-9, 2e-9, 2e-9))
    mat = Material(ms=3.84e5, a_ex=8.78e-12, d=1.58e-3, ku=1.0e5, alpha=1.0)
    sys = System(mesh, mat, demag=True, h_ext=(0.0, 0.0, 1.0e5))

    def solve(m0: torch.Tensor) -> torch.Tensor:
        m = m0
        for _ in range(2):
            m = sys.step_rk4(m, 1e-13)
        return m

    m0 = torch.randn(4, 4, 1, 3, dtype=torch.float64)
    m0 = (m0 / m0.norm(dim=-1, keepdim=True)).requires_grad_(True)
    assert gradcheck(solve, (m0,), eps=1e-6, atol=1e-4, rtol=1e-3)
