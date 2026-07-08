"""Native smoke: the native CUDA op builds, dispatches, and is differentiable (float64
gradcheck)."""

from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck

from spinforge.native import scale


@pytest.mark.gpu
def test_native_scale_builds_and_gradchecks() -> None:
    x = torch.randn(16, dtype=torch.float64, device="cuda", requires_grad=True)
    # multiply-by-3 is the same IEEE op on both paths -> bit-identical
    torch.testing.assert_close(scale(x, 3.0), 3.0 * x, rtol=0.0, atol=0.0)
    assert gradcheck(lambda z: scale(z, 3.0), (x,))
