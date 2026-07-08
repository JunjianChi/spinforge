"""Oracle: the volume-averaged demag field of a uniformly magnetized cube is -Ms/3.

A cube has demag factors Nxx=Nyy=Nzz=1/3 by cubic symmetry and is exactly representable on a cubic
grid (no staircasing), so <H_dem> = -Ms/3 along the magnetization and ~0 transverse.
"""

from __future__ import annotations

import torch

from spinforge.core.demag import DemagField
from spinforge.core.mesh import Mesh


def test_uniform_cube_demag_is_minus_one_third() -> None:
    n = 16
    mesh = Mesh(n=(n, n, n), dx=(2e-9, 2e-9, 2e-9))
    ms = 8e5
    m = torch.zeros(n, n, n, 3, dtype=torch.float64)
    m[..., 2] = 1.0  # uniform along +z
    h = DemagField(mesh)(m, ms)
    h_avg = h.mean(dim=(0, 1, 2)) / ms
    # measured error ~2e-13 (float64 Newell + FFT); 1e-12 leaves 5x cross-platform margin
    torch.testing.assert_close(
        h_avg, torch.tensor([0.0, 0.0, -1.0 / 3.0], dtype=torch.float64), rtol=0.0, atol=1e-12
    )


def test_kernel_rebuilds_on_dtype_change() -> None:
    """A reused DemagField must not keep a stale-precision kernel: after a float64 call, a float32
    call has to rebuild (else the float64 kernel promotes the result back to float64)."""
    mesh = Mesh(n=(4, 4, 4), dx=(2e-9, 2e-9, 2e-9))
    ms = 8e5
    field = DemagField(mesh)
    m64 = torch.randn(4, 4, 4, 3, dtype=torch.float64)
    field(m64, ms)  # builds a float64 kernel
    h32 = field(m64.to(torch.float32), ms)
    assert h32.dtype == torch.float32
    # same deterministic op sequence on both paths -> bit-identity IS the claim
    torch.testing.assert_close(h32, DemagField(mesh)(m64.to(torch.float32), ms), rtol=0.0, atol=0.0)


def test_newell_component_permutation_table() -> None:
    """Pin the component/axis-permutation table via relabeling identities on ASYMMETRIC inputs:
    N_yy is N_xx with axes relabeled (x,y,z)->(y,z,x), etc. A typo'd permutation breaks equality;
    distinct grid sizes and cell aspect make shape/axis mixups loud."""
    from spinforge.core.demag import newell_component

    idx = [
        torch.arange(2, dtype=torch.float64).reshape(-1, 1, 1) * 1.0,
        torch.arange(3, dtype=torch.float64).reshape(1, -1, 1) * 1.5,
        torch.arange(4, dtype=torch.float64).reshape(1, 1, -1) * 2.0,
    ]
    gx, gy, gz = idx
    d = (1.0, 1.5, 2.0)

    def relabeled(ab: str, perm: tuple[int, int, int]) -> torch.Tensor:
        g = (gx, gy, gz)
        dp = (d[perm[0]], d[perm[1]], d[perm[2]])
        return newell_component(ab, g[perm[0]], g[perm[1]], g[perm[2]], dp)

    for want_ab, base_ab, perm in (
        ("yy", "xx", (1, 2, 0)),
        ("zz", "xx", (2, 0, 1)),
        ("xz", "xy", (0, 2, 1)),
        ("yz", "xy", (1, 2, 0)),
    ):
        # broadcasting is positional: both sides land on the same (nx, ny, nz) axes, so a correct
        # table makes the identity exact equality -- no output permute
        got = newell_component(want_ab, gx, gy, gz, d)
        want = relabeled(base_ab, perm)
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-15)
