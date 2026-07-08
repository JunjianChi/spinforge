"""Unit oracles for the local field terms: anisotropy, bulk DMI, Zeeman."""

from __future__ import annotations

import torch
from scipy import constants

from spinforge.core.anisotropy import uniaxial_anisotropy_field
from spinforge.core.dmi import bulk_dmi_field
from spinforge.core.mesh import Mesh
from spinforge.core.zeeman import zeeman_field


def test_anisotropy_along_and_perp_to_easy_axis() -> None:
    mesh = Mesh(n=(4, 4, 4), dx=(1e-9, 1e-9, 1e-9))
    ku, ms = 5.1e5, 8e5
    pref = 2.0 * ku / (constants.mu_0 * ms)
    m = torch.zeros(4, 4, 4, 3, dtype=torch.float64)
    m[..., 2] = 1.0  # aligned with easy axis z
    h = uniaxial_anisotropy_field(m, mesh, ku, (0.0, 0.0, 1.0), ms)
    torch.testing.assert_close(h[..., 2], torch.full_like(h[..., 2], pref), rtol=1e-12, atol=1e-6)
    m[:] = 0.0
    m[..., 0] = 1.0  # perpendicular to easy axis -> zero
    h = uniaxial_anisotropy_field(m, mesh, ku, (0.0, 0.0, 1.0), ms)
    torch.testing.assert_close(h, torch.zeros_like(h), rtol=0.0, atol=1e-6)


def test_bulk_dmi_curl_of_linear_field() -> None:
    nx, dx, d, ms = 8, 2e-9, 1.58e-3, 3.84e5
    mesh = Mesh(n=(nx, 1, 1), dx=(dx, dx, dx))
    m = torch.zeros(nx, 1, 1, 3, dtype=torch.float64)
    m[:, 0, 0, 1] = torch.arange(nx, dtype=torch.float64) * dx  # m_y = x  -> curl_z = d(m_y)/dx = 1
    h = bulk_dmi_field(m, mesh, d, ms)
    expected = -(2.0 * d / (constants.mu_0 * ms)) * 1.0
    torch.testing.assert_close(
        h[1:-1, 0, 0, 2], torch.full((nx - 2,), expected, dtype=torch.float64), rtol=1e-9, atol=1e-3
    )


def test_bulk_dmi_field_is_float64_gradcheck_clean() -> None:
    """The standalone differentiable bulk-DMI term gets its own gradcheck (the chiral path
    covered its curl only indirectly)."""
    mesh = Mesh(n=(4, 4, 4), dx=(2e-9, 2e-9, 2e-9))
    torch.manual_seed(0)
    m = torch.randn(4, 4, 4, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x: bulk_dmi_field(x, mesh, d=1.58e-3, ms=3.84e5), (m,), atol=1e-6
    )


def test_zeeman_is_the_applied_field() -> None:
    m = torch.zeros(3, 3, 3, 3, dtype=torch.float64)
    h_ext = (1.0e4, -2.0e4, 3.0e4)
    h = zeeman_field(m, h_ext)
    assert h.shape == m.shape
    torch.testing.assert_close(
        h[1, 1, 1], torch.tensor(h_ext, dtype=torch.float64), rtol=0.0, atol=0.0
    )


def test_anisotropy_accepts_spatial_ku_tensor() -> None:
    """Graded-anisotropy inverse design (the inverse-design gate's control): ku may be a spatial
    tensor.

    A [nx, ny, 1, 1] profile must broadcast over z and match the scalar result cell-wise, and the
    field must be differentiable w.r.t. the PROFILE (the design parameter), f64-gradchecked."""
    mesh = Mesh(n=(3, 2, 2), dx=(2e-9, 2e-9, 2e-9))
    ms = 8e5
    torch.manual_seed(0)
    m = torch.randn(3, 2, 2, 3, dtype=torch.float64)
    m = m / m.norm(dim=-1, keepdim=True)

    ku_map = torch.tensor([1e4, 3e4, 5e4], dtype=torch.float64).reshape(3, 1, 1, 1)
    got = uniaxial_anisotropy_field(m, mesh, ku_map, (0.0, 0.0, 1.0), ms)
    for i, ku_i in enumerate((1e4, 3e4, 5e4)):
        want = uniaxial_anisotropy_field(m[i : i + 1], mesh, ku_i, (0.0, 0.0, 1.0), ms)
        torch.testing.assert_close(got[i : i + 1], want, rtol=1e-14, atol=0.0)

    ku_var = ku_map.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda k: 1e-5 * uniaxial_anisotropy_field(m, mesh, k, (0.0, 0.0, 1.0), ms),
        (ku_var,),
        rtol=1e-6,
        atol=1e-9,
    )
