"""Demagnetizing field via the Newell cell-averaged demag tensor and an FFT convolution.

Public formulas: A. J. Newell, W. Williams, D. J. Dunlop, J. Geophys. Res. 98, 9551 (1993);
cf. M. J. Donahue (OOMMF, NIST, public domain). The cell-to-cell averaged demag tensor is the triple
central second-difference of the analytic precursors f (diagonal) and g (off-diagonal); the field is
the linear convolution of the tensor with M = Ms*m, evaluated by a zero-padded FFT. Implemented
independently in PyTorch (full-complex FFT, so autograd differentiates it directly).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch

from .mesh import Mesh

_4PI = 4.0 * math.pi


def _f(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Newell diagonal (Nxx) precursor."""
    x, y, z = x.abs(), y.abs(), z.abs()
    x2, y2, z2 = x * x, y * y, z * z
    r = torch.sqrt(x2 + y2 + z2)
    out = (2.0 * x2 - y2 - z2) * r / 6.0
    out = out + (0.5 * y * (z2 - x2) * torch.asinh(y / torch.sqrt(x2 + z2))).nan_to_num()
    out = out + (0.5 * z * (y2 - x2) * torch.asinh(z / torch.sqrt(x2 + y2))).nan_to_num()
    out = out - (x * y * z * torch.atan(y * z / (x * r))).nan_to_num()
    return out


def _g(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Newell off-diagonal (Nxy) precursor."""
    z = z.abs()
    x2, y2, z2 = x * x, y * y, z * z
    r = torch.sqrt(x2 + y2 + z2)
    out = -x * y * r / 3.0
    out = out + (x * y * z * torch.asinh(z / torch.sqrt(x2 + y2))).nan_to_num()
    out = out + (y / 6.0 * (3.0 * z2 - y2) * torch.asinh(x / torch.sqrt(y2 + z2))).nan_to_num()
    out = out + (x / 6.0 * (3.0 * z2 - x2) * torch.asinh(y / torch.sqrt(x2 + z2))).nan_to_num()
    out = out - (z**3 / 6.0 * torch.atan(x * y / (z * r))).nan_to_num()
    out = out - (z * y2 / 2.0 * torch.atan(x * z / (y * r))).nan_to_num()
    out = out - (z * x2 / 2.0 * torch.atan(y * z / (x * r))).nan_to_num()
    return out


def _newell(func, x, y, z, d) -> torch.Tensor:
    """Tensor component = triple central second-difference of ``func`` over 4*pi*vol (Newell)."""
    dx, dy, dz = d
    stencil = ((-1, 1.0), (0, -2.0), (1, 1.0))
    acc = torch.zeros_like(x)
    for oi, wi in stencil:
        for oj, wj in stencil:
            for ok, wk in stencil:
                acc = acc + (wi * wj * wk) * func(x + oi * dx, y + oj * dy, z + ok * dz)
    return acc / (_4PI * dx * dy * dz)


# component -> (precursor, axis permutation): N_yy is N_xx with axes relabeled (x,y,z)->(y,z,x),
# etc. Pinned by the relabeling-identity test on asymmetric grids (a typo cannot hide).
_COMPONENTS: dict[str, tuple[Callable[..., torch.Tensor], tuple[int, int, int]]] = {
    "xx": (_f, (0, 1, 2)),
    "yy": (_f, (1, 2, 0)),
    "zz": (_f, (2, 0, 1)),
    "xy": (_g, (0, 1, 2)),
    "xz": (_g, (0, 2, 1)),
    "yz": (_g, (1, 2, 0)),
}


def newell_component(
    ab: str,
    gx: torch.Tensor,
    gy: torch.Tensor,
    gz: torch.Tensor,
    d: Sequence[float],
) -> torch.Tensor:
    """Real-space Newell tensor component ``ab`` on a (broadcastable) padded grid.

    Pure tensor math, MPI-free: the single-process build evaluates it on the full padded
    grid, the distributed build on this rank's z-slab only (owner-computes). ``gx, gy, gz`` may be
    broadcast views (e.g. ``[2nx,1,1]``); broadcasting is positional, so the output axes are always
    (x, y, z) regardless of the component's internal axis permutation.
    """
    func, perm = _COMPONENTS[ab]
    g = (gx, gy, gz)
    return _newell(func, g[perm[0]], g[perm[1]], g[perm[2]], (d[perm[0]], d[perm[1]], d[perm[2]]))


class DemagField:
    """Newell demag tensor for a fixed ``Mesh``, applied as a zero-padded FFT convolution."""

    def __init__(self, mesh: Mesh) -> None:
        self.mesh = mesh
        self._kernel: dict[str, torch.Tensor] | None = None
        self._kernel_key: tuple[torch.dtype, torch.device] | None = None
        self._adims: list[int] = [i for i in range(3) if mesh.n[i] > 1]
        self._shape: list[int] = [2 * mesh.n[i] if mesh.n[i] > 1 else 1 for i in range(3)]

    def _build(self, real_dtype: torch.dtype, device: torch.device) -> None:
        # Lengths are made dimensionless (scaled by the smallest cell): the tensor is
        # scale-invariant, and this avoids precision loss in f, g. Kernel built in float64.
        dmin = min(self.mesh.dx)
        d = [self.mesh.dx[i] / dmin for i in range(3)]
        # Build the grid (and so evaluate the f/g precursors) directly on the target device: the
        # padded (2N)^3 stencil of transcendentals is the build cost, and on CPU it dominates at
        # large N. fftfreq in float64 on-device keeps the kernel double-precise everywhere.
        idx = [
            torch.fft.fftfreq(self._shape[i], 1.0 / self._shape[i], device=device).to(torch.float64)
            * d[i]
            for i in range(3)
        ]
        # broadcastable views, not meshgrid: no full-domain coordinate arrays -- per component the
        # only full-size arrays alive are the accumulator and one stencil-shift evaluation
        gx = idx[0].reshape(-1, 1, 1)
        gy = idx[1].reshape(1, -1, 1)
        gz = idx[2].reshape(1, 1, -1)
        # Keep only the real (even) part of the kernel spectrum: the demag tensor is even, so its
        # transform is real; dropping the spurious odd part of the discrete off-diagonal precursors
        # makes the operator self-adjoint (gradcheck) and is the physically correct symmetric demag.
        # .contiguous() detaches the .real view from its complex parent buffer, which would
        # otherwise stay pinned at 2x the kernel's memory (the .to() casts no-op on the f64 path).
        self._kernel = {
            ab: torch.fft.fftn(newell_component(ab, gx, gy, gz, d), dim=self._adims)
            .real.to(real_dtype)
            .contiguous()
            for ab in _COMPONENTS
        }
        self._kernel_key = (real_dtype, device)

    def __call__(self, m: torch.Tensor, ms: float) -> torch.Tensor:
        """Demag field h (A/m), shape ``(nx, ny, nz, 3)``, for unit field ``m`` and ``Ms``."""
        if self._kernel is None or self._kernel_key != (m.dtype, m.device):
            self._build(m.dtype, m.device)  # rebuild on a changed precision/device, no stale reuse
        assert self._kernel is not None
        k, s = self._kernel, [self._shape[i] for i in self._adims]
        mf = [torch.fft.fftn(ms * m[..., a], dim=self._adims, s=s) for a in range(3)]
        hf = [
            k["xx"] * mf[0] + k["xy"] * mf[1] + k["xz"] * mf[2],
            k["xy"] * mf[0] + k["yy"] * mf[1] + k["yz"] * mf[2],
            k["xz"] * mf[0] + k["yz"] * mf[1] + k["zz"] * mf[2],
        ]
        crop = tuple(slice(0, self.mesh.n[i]) for i in range(3))
        h = [torch.fft.ifftn(hb, dim=self._adims).real[crop] for hb in hf]
        return torch.stack(h, dim=-1)
