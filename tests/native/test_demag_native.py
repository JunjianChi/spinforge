"""Native CUDA cuFFT demag matches the PyTorch reference and is differentiable (gradcheck)."""

from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck

from spinforge.core.demag import DemagField
from spinforge.core.mesh import Mesh
from spinforge.native import native_demag


def test_native_demag_rejects_quasi_2d_mesh() -> None:
    """The CUDA op pads every axis to 2*n, but DemagField collapses singleton axes -- a quasi-2D
    mesh must raise, not read out of bounds. No CUDA needed: the guard precedes the load."""
    mesh = Mesh(n=(8, 8, 1), dx=(2e-9, 2e-9, 2e-9))
    m = torch.zeros(8, 8, 1, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="3D"):
        native_demag(m, mesh, 8e5)


@pytest.mark.gpu
def test_native_demag_matches_reference_and_gradchecks() -> None:
    mesh = Mesh(n=(4, 4, 4), dx=(2e-9, 2e-9, 2e-9))
    ms = 8e5
    m = torch.randn(4, 4, 4, 3, dtype=torch.float64, device="cuda")
    m = m / m.norm(dim=-1, keepdim=True)

    ref = DemagField(mesh)(m, ms)  # PyTorch reference (on CUDA)
    got = native_demag(m, mesh, ms)
    torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-2)  # field ~ Ms scale

    # gradcheck with Ms=1 so the field is O(1) and tolerances are natural (demag is linear in Ms)
    mg = m.clone().requires_grad_(True)
    assert gradcheck(lambda x: native_demag(x, mesh, 1.0), (mg,), eps=1e-6, atol=1e-6, rtol=1e-4)


@pytest.mark.gpu
def test_native_demag_fp32_matches_fp64() -> None:
    """The FP32 (C2C) path agrees with FP64 (Z2Z) to single-precision tolerance."""
    mesh = Mesh(n=(16, 16, 16), dx=(2e-9, 2e-9, 2e-9))
    ms = 8e5
    m64 = torch.randn(16, 16, 16, 3, dtype=torch.float64, device="cuda")
    m64 = m64 / m64.norm(dim=-1, keepdim=True)

    h64 = native_demag(m64, mesh, ms)
    h32 = native_demag(m64.to(torch.float32), mesh, ms)
    assert h32.dtype == torch.float32
    # field scale ~ Ms; ~1e-5 relative is what FP32 cuFFT gives on a 32^3 padded transform
    torch.testing.assert_close(h32, h64.to(torch.float32), rtol=1e-4, atol=ms * 1e-5)
