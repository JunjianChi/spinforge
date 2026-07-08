"""muMAG Standard Problem 4 (reduced) — guards the dynamic LLG against the community reference.

Full-resolution validation (200x50x1, both fields, the 0.138 ns crossing) lives in
``experiments/mumag_sp4.py`` + ``results/mumag_sp4.md``. This is a coarse CPU version that still
reverses the film and lands the <mx>=0 crossing in the reference band, so it catches a regression
in precession/damping without a GPU. Marked slow (excluded from the fast CPU gate).
"""

from __future__ import annotations

import math

import pytest
import torch

from spinforge.core.mesh import Mesh
from spinforge.core.system import Material, System

MU0 = 4.0e-7 * math.pi
MS, A_EX = 8.0e5, 1.3e-11


def _crossing_ns(times: list[float], mx: list[float]) -> float:
    for i in range(1, len(mx)):
        if mx[i - 1] > 0.0 >= mx[i]:
            frac = mx[i - 1] / (mx[i - 1] - mx[i])
            return (times[i - 1] + (times[i] - times[i - 1]) * frac) * 1e9
    return float("nan")


@pytest.mark.slow
def test_sp4_field1_reversal() -> None:
    # coarse 50x13x1 ~ 500x125x3 nm (the fine 200x50x1 run is the committed result)
    mesh = Mesh(n=(50, 13, 1), dx=(10e-9, 125e-9 / 13, 3e-9))
    dev = torch.device("cpu")

    # S-state: saturate along [1,1,1], relax at zero field with alpha=1
    relax = System(mesh, Material(ms=MS, a_ex=A_EX, alpha=1.0), demag=True)
    m = torch.ones(*mesh.n, 3, dtype=torch.float64, device=dev)
    m = m / m.norm(dim=-1, keepdim=True)
    m = relax.relax(m, steps=4000, dt=2e-13)
    s_mx = float(m.reshape(-1, 3).mean(0)[0])
    assert s_mx > 0.9  # remanent S-state sits near +x saturation

    # field 1 = (-24.6, 4.3, 0) mT, applied instantaneously with the realistic damping
    h = tuple(v * 1e-3 / MU0 for v in (-24.6, 4.3, 0.0))
    dyn = System(mesh, Material(ms=MS, a_ex=A_EX, alpha=0.02), demag=True, h_ext=h)
    dt = 1e-13
    times, mx = [], []
    for step in range(2001):
        if step % 20 == 0:
            times.append(step * dt)
            mx.append(float(m.reshape(-1, 3).mean(0)[0]))
        m = dyn.step_rk4(m, dt)

    tc = _crossing_ns(times, mx)
    assert 0.10 < tc < 0.17, f"<mx>=0 crossing {tc:.3f} ns outside reference band"
    assert mx[-1] < 0.0  # the film has reversed
