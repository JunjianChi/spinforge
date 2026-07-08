"""Adiabatic Zhang-Li spin-transfer torque (xi = 0): the current term that drives the texture.

The conduction-electron spin current advects the local magnetization: the Gilbert-form torque is
``-(u . grad) m`` where ``u`` is the spin-drift velocity (m/s, ~ proportional to the current
density). This is the adiabatic limit (non-adiabatic ``beta = 0``); it is what makes a skyrmion
move (and, via the gyrotropic coupling, deflect at the skyrmion-Hall angle). The non-adiabatic
term is deliberately out of scope.
"""

from __future__ import annotations

import torch

from .mesh import Mesh


def u_grad_m(m: torch.Tensor, mesh: Mesh, u: tuple[float, float, float]) -> torch.Tensor:
    """``(u . grad) m`` by central differences with replicate boundaries; shape matches ``m``."""
    out = torch.zeros_like(m)
    for ax in range(3):
        n = mesh.n[ax]
        if n == 1 or u[ax] == 0.0:
            continue
        lo, hi = m.narrow(ax, 0, 1), m.narrow(ax, n - 1, 1)
        mp = torch.cat([lo, m, hi], dim=ax)
        d = (mp.narrow(ax, 2, n) - mp.narrow(ax, 0, n)) / (2.0 * mesh.dx[ax])
        out = out + u[ax] * d
    return out
