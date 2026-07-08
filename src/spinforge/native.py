"""Loader + autograd wrappers for the native C++/CUDA ops (JIT-compiled on a CUDA host).

On a machine without a CUDA toolkit (e.g. the Mac), importing this and calling a native op raises;
callers fall back to the pure-PyTorch reference. JIT (``cpp_extension.load``) is for dev iteration;
the shipped build is CMake / scikit-build-core (decisions.md).
"""

from __future__ import annotations

import os

import torch

from .core.mesh import Mesh

_HERE = os.path.dirname(__file__)
_LOADED = False
_KERNEL_CACHE: dict[object, torch.Tensor] = {}


def _ensure_loaded() -> None:
    """JIT-compile and load the native library so its TORCH_LIBRARY ops register on torch.ops."""
    global _LOADED
    if not _LOADED:
        from torch.utils.cpp_extension import load

        load(
            name="spinforge_native",
            sources=[
                os.path.join(_HERE, "csrc", "scale_op.cu"),
                os.path.join(_HERE, "csrc", "demag.cu"),
            ],
            extra_ldflags=["-lcufft"],
            is_python_module=False,  # ops registered via TORCH_LIBRARY, no pybind module
            verbose=False,
        )
        _LOADED = True


# --- smoke op ---


class _Scale(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, x: torch.Tensor, s: float) -> torch.Tensor:
        ctx.s = s  # type: ignore[attr-defined]
        return torch.ops.spinforge.scale_forward(x, s)

    @staticmethod
    def backward(ctx: object, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return ctx.s * grad, None  # type: ignore[attr-defined]


def scale(x: torch.Tensor, s: float) -> torch.Tensor:
    """Native CUDA ``s * x`` (the smoke op): proves the build + dispatch + autograd chain."""
    _ensure_loaded()
    return _Scale.apply(x, s)


# --- native demag (cuFFT) ---


def _demag_kernel(mesh: Mesh, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Newell demag kernel (6 real FFT components) for the native op.

    Always built in float64 for accuracy, then cast to ``dtype`` (the FP32 path keeps the
    geometric tensor double-precise and only the convolution runs single-precision).
    """
    key = (mesh.n, mesh.dx, str(device), dtype)
    if key not in _KERNEL_CACHE:
        from .core.demag import DemagField

        base = DemagField(mesh)
        base._build(torch.float64, device)
        assert base._kernel is not None
        keys = ("xx", "xy", "xz", "yy", "yz", "zz")
        stacked = torch.stack([base._kernel[k] for k in keys]).contiguous()
        _KERNEL_CACHE[key] = stacked.to(dtype)
    return _KERNEL_CACHE[key]


class _DemagNative(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object, m: torch.Tensor, kernel: torch.Tensor, ms: float, n: tuple[int, int, int]
    ) -> torch.Tensor:
        ctx.kernel, ctx.ms, ctx.n = kernel, ms, n  # type: ignore[attr-defined]
        return torch.ops.spinforge.demag_forward(m, kernel, ms, n[0], n[1], n[2])

    @staticmethod
    def backward(ctx: object, grad_h: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        # demag is self-adjoint: grad_m = N * (ms * grad_h) = demag_forward(grad_h)
        k, ms, n = ctx.kernel, ctx.ms, ctx.n  # type: ignore[attr-defined]
        gm = torch.ops.spinforge.demag_forward(grad_h.contiguous(), k, ms, n[0], n[1], n[2])
        return gm, None, None, None


def native_demag(m: torch.Tensor, mesh: Mesh, ms: float) -> torch.Tensor:
    """Native cuFFT demag for ``m`` (``[nx,ny,nz,3]`` CUDA, float32 or float64); matches core.demag.

    Dispatches the cuFFT precision (C2C/Z2Z) on ``m.dtype``; the kernel is cast to match.
    """
    # The CUDA op pads every axis to 2*n, but DemagField collapses singleton axes -- so on a
    # quasi-2D mesh the passed kernel is short by 2x on that axis and the contract reads out of
    # bounds. Fail loudly (before the extension load) instead; use DemagField for quasi-2D meshes.
    if any(ni < 2 for ni in mesh.n):
        raise ValueError(f"native_demag requires a fully 3D mesh (all n>1); got n={mesh.n}")
    _ensure_loaded()
    kernel = _demag_kernel(mesh, m.device, m.dtype)
    return _DemagNative.apply(m, kernel, ms, mesh.n)
