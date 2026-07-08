"""Slab-decomposed (z) distributed Newell demag: local FFTs + all_to_all transpose.

Each rank owns m[:, :, z_slab, :] and gets h on the same slab. The 3D FFT is done by the transpose
method: local xy-FFT -> all_to_all transpose (z-distributed -> xy-distributed) -> local z-FFT, then
the kernel multiply and the mirrored inverse. Reproduces the single-process DemagField per slab.

The transpose backend is injected (swappable seam): the default is the split-phase ``TorchAsyncA2A``
(gloo/NCCL); instruments, the mixed-precision wire, and the from-scratch CUDA-aware-MPI collective
slot in via ``SyncCollectiveA2A`` with no change to this op. The ``schedule`` knob selects the
collective ordering: ``"sequential"`` (each transpose completed before the next component's FFT --
the naive schedule) or ``"pipelined"`` (component a+1's local FFT runs while component a's
transpose is in flight, forward and -- via the split-phase autograd pair -- adjoint too). Both
schedules are the same computation; overlap is a timing property, measured on real hardware.
Pipelined holds up to 3 in-flight send+recv buffer pairs per transpose stage (3 x 2 x plane_elems
complex) instead of sequential's 1 -- the memory cost of the overlap window.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist

from ..core.demag import newell_component
from ..core.mesh import Mesh
from .collectives import A2ABackend, TorchAsyncA2A, TransposeWaitable

_KEYS = ("xx", "xy", "xz", "yy", "yz", "zz")
_SCHEDULES = ("sequential", "pipelined")

Collective = Callable[[torch.Tensor], torch.Tensor]


def _raw_all_to_all_complex(x: torch.Tensor) -> torch.Tensor:
    """Build-time transpose: exact dtype, no autograd, no traffic counting.

    The apply-path collectives are the counted/injectable seam (mixed-precision wire, timing
    instruments); the kernel is a constant built once per (dtype, device), so its transpose must
    stay exact-precision and out of the traffic accounting. Runtime-detected carrier: the torch
    process group when one exists, else the native MPI collective (a plain ``mpirun`` launch has
    no torch PG at all -- exactly how the native demag runs).
    """
    xr = torch.view_as_real(x).contiguous()
    if dist.is_available() and dist.is_initialized():
        out = torch.empty_like(xr)
        dist.all_to_all_single(out, xr)
        return torch.view_as_complex(out)
    from .native_collectives import mpi_all_to_all  # lazy: breaks the module cycle

    return torch.view_as_complex(mpi_all_to_all(xr))


class DistributedDemagField:
    """Distributed Newell demag over a z-slab decomposition (``nz`` divisible by ``world_size``)."""

    def __init__(
        self,
        mesh: Mesh,
        world_size: int,
        rank: int,
        *,
        backend: A2ABackend | None = None,
        schedule: str = "sequential",
        use_rfft: bool = False,
    ) -> None:
        nx, ny, nz = mesh.n
        # rfft2 over (x, y) keeps only the Hermitian-non-redundant ny+1 columns: ~2x fewer wire
        # bytes on every transpose, fwd and adjoint (decisions 2026-07-06). z stays full-complex
        # (the z-FFT input is already complex after the xy transform). vjps come from the
        # rfft2/irfft2 autograd primitives -- no hand-written adjoint to mis-weight.
        self._plane = (2 * nx, ny + 1) if use_rfft else (2 * nx, 2 * ny)
        if nz % world_size or (self._plane[0] * self._plane[1]) % world_size:
            raise ValueError(
                f"need nz ({nz}) and the transposed plane rows "
                f"({self._plane[0]}*{self._plane[1]}) divisible by world_size ({world_size})"
            )
        if min(mesh.n) == 1:
            # a singleton axis drops out of the reference kernel's padding (core _adims), so the
            # hardcoded 3D padded layout here would silently build a different kernel
            raise ValueError(f"quasi-2D meshes are not supported by the slab demag; got n={mesh.n}")
        if schedule not in _SCHEDULES:
            raise ValueError(f"unknown schedule {schedule!r}; expected one of {_SCHEDULES}")
        self.mesh, self.p, self.rank = mesh, world_size, rank
        self._rfft = use_rfft
        self._schedule = schedule
        self._backend: A2ABackend = backend if backend is not None else TorchAsyncA2A()
        # The kernel dtype/device must follow the input m, but __init__ has no input yet. Defer the
        # build+per-rank slice to the first __call__ (keyed on (dtype, device)) instead of a
        # per-dtype rebuild-and-reslice: the slab slice is cheap and this keeps one build path.
        self._k: dict[str, torch.Tensor] | None = None
        self._k_key: tuple[torch.dtype, torch.device] | None = None

    def _build_k(self, real_dtype: torch.dtype, device: torch.device) -> None:
        # Owner-computes: this rank evaluates the padded Newell tensor ONLY on its own z-slab and
        # pushes it through the same fft2 -> transpose -> z-fft pipeline as m, landing each
        # component directly in the [XY/P, 2nz] slab the spectral multiply consumes. No rank ever
        # holds the full kernel; materializing the whole padded domain per rank would OOM before
        # the first step at the science target. Evaluated
        # in f64 (the f/g precursors cancel catastrophically below that), cast once at the end;
        # .real keeps the even part, so the operator stays exactly self-adjoint. The transpose is
        # the RAW collective: build traffic is uncounted (test_traffic pins apply-path counts) and
        # exact-precision (a mixed-precision apply collective must not corrupt the kernel).
        nx, ny, nz = self.mesh.n
        dmin = min(self.mesh.dx)
        d = tuple(self.mesh.dx[i] / dmin for i in range(3))
        shape = (2 * nx, 2 * ny, 2 * nz)
        idx = [
            torch.fft.fftfreq(shape[i], 1.0 / shape[i], device=device).to(torch.float64) * d[i]
            for i in range(3)
        ]
        twonzl = 2 * nz // self.p
        gx = idx[0].reshape(-1, 1, 1)
        gy = idx[1].reshape(1, -1, 1)
        gz = idx[2][self.rank * twonzl : (self.rank + 1) * twonzl].reshape(1, 1, -1)
        xy_fft = torch.fft.rfft2 if self._rfft else torch.fft.fft2
        k: dict[str, torch.Tensor] = {}
        for ab in _KEYS:
            v = newell_component(ab, gx, gy, gz, d)  # [2nx, 2ny, 2nz/P] f64, this slab only
            g = self._transpose_to_xy(xy_fft(v, dim=(0, 1)), _raw_all_to_all_complex)
            # .contiguous() detaches the .real view from its complex parent (else 2x resident)
            k[ab] = torch.fft.fft(g, dim=1).real.to(real_dtype).contiguous()
        self._k = k
        self._k_key = (real_dtype, device)

    def _transpose_to_xy(self, a: torch.Tensor, collective: Collective) -> torch.Tensor:
        # Kernel-build-only transpose (raw blocking collective). [2nx|packed, ..., L]
        # z-distributed -> [XY/P, P*L] xy-distributed; shape-generic in L.
        pl0, pl1, zl = a.shape
        xy, p = pl0 * pl1, self.p
        a = a.reshape(xy, zl).reshape(p, xy // p, zl)
        recv = collective(a)  # [P(src), XY/P, zl]
        return recv.permute(1, 0, 2).reshape(xy // p, p * zl)

    # apply-path transposes are split-phase: _start_* launches the collective, _finish_* completes
    # it; the schedule decides how much local FFT work runs in the window between them.
    def _start_xy(self, a: torch.Tensor) -> TransposeWaitable:
        pl0, pl1, zl = a.shape
        xy, p = pl0 * pl1, self.p
        return self._backend.start(a.reshape(xy, zl).reshape(p, xy // p, zl))

    @staticmethod
    def _finish_xy(h: TransposeWaitable) -> torch.Tensor:
        recv = h.wait()  # [P(src), XY/P, zl]
        p, xyp, zl = recv.shape
        return recv.permute(1, 0, 2).reshape(xyp, p * zl)

    def _start_z(self, g: torch.Tensor) -> TransposeWaitable:
        xyp, nz = g.shape
        nzl, p = nz // self.p, self.p
        return self._backend.start(g.reshape(xyp, p, nzl).permute(1, 0, 2).contiguous())

    def _finish_z(self, h: TransposeWaitable) -> torch.Tensor:
        recv = h.wait()  # [P, XY/P, nzl]
        p0, p1 = self._plane
        return recv.reshape(p0 * p1, recv.shape[2]).reshape(p0, p1, recv.shape[2])

    def __call__(self, m_local: torch.Tensor, ms: float) -> torch.Tensor:
        """Demag field on this rank's z-slab; ``m_local`` is ``[nx, ny, nz_local, 3]``."""
        nx, ny, nz = self.mesh.n
        if self._k is None or self._k_key != (m_local.dtype, m_local.device):
            self._build_k(m_local.dtype, m_local.device)  # dtype/device follow the input, no upcast
        assert self._k is not None
        big_m = ms * m_local
        xy_fft = torch.fft.rfft2 if self._rfft else torch.fft.fft2

        def fwd_plane(a: int) -> torch.Tensor:
            return xy_fft(big_m[..., a], s=(2 * nx, 2 * ny), dim=(0, 1))

        if self._schedule == "pipelined":
            # launch component a's transpose, compute component a+1's FFT while it flies
            pend = [self._start_xy(fwd_plane(a)) for a in range(3)]
            mf = [torch.fft.fft(self._finish_xy(h), n=2 * nz, dim=1) for h in pend]
        else:  # sequential: each transpose completed before the next FFT starts (naive schedule)
            mf = [
                torch.fft.fft(self._finish_xy(self._start_xy(fwd_plane(a))), n=2 * nz, dim=1)
                for a in range(3)
            ]
        k = self._k
        hf = [
            k["xx"] * mf[0] + k["xy"] * mf[1] + k["xz"] * mf[2],
            k["xy"] * mf[0] + k["yy"] * mf[1] + k["yz"] * mf[2],
            k["xz"] * mf[0] + k["yz"] * mf[1] + k["zz"] * mf[2],
        ]

        def inv_plane(b: int) -> torch.Tensor:
            return torch.fft.ifft(hf[b], n=2 * nz, dim=1)[:, :nz]  # [XY/P, nz]

        if self._schedule == "pipelined":
            pend_z = [self._start_z(inv_plane(b)) for b in range(3)]
            gzs = [self._finish_z(h) for h in pend_z]
        else:
            gzs = [self._finish_z(self._start_z(inv_plane(b))) for b in range(3)]
        h = []
        for gz in gzs:
            if self._rfft:
                h.append(torch.fft.irfft2(gz, s=(2 * nx, 2 * ny), dim=(0, 1))[:nx, :ny, :])
            else:
                h.append(torch.fft.ifft2(gz, dim=(0, 1))[:nx, :ny, :].real)
        return torch.stack(h, dim=-1)
