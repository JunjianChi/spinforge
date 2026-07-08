"""Exchange field = (2A/(mu0 Ms)) * laplacian(m), 6-neighbour stencil, Neumann (zero-flux) BC.

Oracles: a uniform m gives zero; an integer profile m_x = i^2 has an exact discrete second
difference of 2, so the interior Laplacian is 2/dx^2 and H_x = (2A/(mu0 Ms)) * 2/dx^2 there.
"""

from __future__ import annotations

import torch
from scipy import constants

from spinforge.core.exchange import exchange_field
from spinforge.core.mesh import Mesh


def test_uniform_state_has_zero_exchange() -> None:
    mesh = Mesh(n=(8, 8, 8), dx=(1e-9, 1e-9, 1e-9))
    m = torch.zeros(8, 8, 8, 3, dtype=torch.float64)
    m[..., 2] = 1.0
    h = exchange_field(m, mesh, a_ex=1.3e-11, ms=8e5)
    torch.testing.assert_close(h, torch.zeros_like(h), rtol=0.0, atol=1e-6)


def test_quadratic_profile_laplacian() -> None:
    nx, dx, a_ex, ms = 8, 2e-9, 1.3e-11, 8e5
    mesh = Mesh(n=(nx, 1, 1), dx=(dx, dx, dx))
    m = torch.zeros(nx, 1, 1, 3, dtype=torch.float64)
    i = torch.arange(nx, dtype=torch.float64)
    # integer profile: discrete 2nd difference of i^2 is exactly 2 (no float cancellation), so the
    # interior Laplacian is 2/dx^2 -- a well-conditioned, O(1e7) target the tolerance can pin
    m[:, 0, 0, 0] = i**2
    h = exchange_field(m, mesh, a_ex=a_ex, ms=ms)
    expected = (2.0 * a_ex / (constants.mu_0 * ms)) * (2.0 / dx**2)
    # interior cells (exclude the two Neumann boundaries); atol tied to the expected scale so a
    # stencil sign/coefficient regression cannot slip through
    torch.testing.assert_close(
        h[1:-1, 0, 0, 0],
        torch.full((nx - 2,), expected, dtype=torch.float64),
        rtol=1e-9,
        atol=abs(expected) * 1e-9,
    )


def test_exchange_field_gradcheck() -> None:
    """Exchange is differentiated through inverse-design loops -> float64 gradcheck."""
    mesh = Mesh(n=(4, 4, 3), dx=(2e-9, 2e-9, 2e-9))
    m = torch.randn(4, 4, 3, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: exchange_field(x, mesh, a_ex=1.3e-11, ms=8e5), (m,))
