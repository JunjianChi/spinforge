"""Native CUDA-aware MPI collectives — the from-scratch distributed-transpose primitive.

Correctness-tested under mpirun on CUDA devices (multi-rank single GPU and a real multi-GPU node);
performance characterization is a separate concern. Loads a separate extension (so the single-GPU
box, which has no MPI, is unaffected). MPI compile/link flags come from ``mpicc`` (OpenMPI). Launch
with ``mpirun -np N python ...`` and ``import mpi4py.MPI`` first to initialize MPI.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import torch

from ..core.mesh import Mesh
from .collectives import SyncCollectiveA2A
from .demag import DistributedDemagField

_HERE = os.path.dirname(os.path.dirname(__file__))
_LOADED = False


def _mpi_flags() -> tuple[list[str], list[str]]:
    """Discover MPI include dirs and link flags via ``mpicc`` (OpenMPI ``--showme``)."""
    mpicc = shutil.which("mpicc") or "mpicc"
    inc = subprocess.check_output([mpicc, "--showme:incdirs"]).decode().split()
    libdirs = subprocess.check_output([mpicc, "--showme:libdirs"]).decode().split()
    libs = subprocess.check_output([mpicc, "--showme:libs"]).decode().split()
    ldflags = [f"-L{d}" for d in libdirs] + [f"-l{lib}" for lib in libs]
    return inc, ldflags


def _ensure_loaded() -> None:
    global _LOADED
    if not _LOADED:
        from torch.utils.cpp_extension import load

        inc, ldflags = _mpi_flags()
        load(
            name="spinforge_mpi",
            sources=[os.path.join(_HERE, "csrc", "mpi_collectives.cpp")],
            extra_include_paths=inc,
            extra_ldflags=ldflags,
            is_python_module=False,
            with_cuda=True,  # c10/cuda/CUDAStream.h needs the CUDA include dirs even in a .cpp
            verbose=False,
        )
        _LOADED = True


class _MpiAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.spinforge_mpi.mpi_all_to_all(x.contiguous())

    @staticmethod
    def backward(ctx: object, grad: torch.Tensor) -> torch.Tensor:
        # adjoint of all_to_all is all_to_all
        return torch.ops.spinforge_mpi.mpi_all_to_all(grad.contiguous())


def mpi_all_to_all(x: torch.Tensor) -> torch.Tensor:
    """Differentiable CUDA-aware MPI all-to-all over dim 0 (equal chunks)."""
    _ensure_loaded()
    return _MpiAllToAll.apply(x)


def mpi_ialltoall_probe(x: torch.Tensor) -> torch.Tensor:
    """RAW MPI_Ialltoall (launched + waited immediately): the per-machine CUDA-awareness probe.

    Stock OpenMPI implements nonblocking collectives via libnbc, which is NOT CUDA-aware even on
    a CUDA-aware build -- this op either matches the blocking result or crashes/corrupts, and
    that answer decides whether the native overlap variant uses Ialltoall or the pairwise path.
    Not differentiable, not injected anywhere: a measurement primitive.
    """
    _ensure_loaded()
    return torch.ops.spinforge_mpi.mpi_ialltoall(x.contiguous())


def mpi_pairwise_all_to_all_raw(x: torch.Tensor) -> torch.Tensor:
    """RAW pairwise Isend/Irecv all-to-all: the fallback when the Ialltoall probe fails.

    Point-to-point goes through UCX (CUDA-aware), and per-peer requests are the building block a
    hand-rolled overlapped native schedule would use. Not differentiable: a probe/benchmark
    primitive until the probe verdict picks the overlap implementation.
    """
    _ensure_loaded()
    return torch.ops.spinforge_mpi.mpi_pairwise_all_to_all(x.contiguous())


def mpi_all_to_all_complex(x: torch.Tensor) -> torch.Tensor:
    """Differentiable MPI all-to-all for a complex tensor (MPI has no complex type -> real view)."""
    return torch.view_as_complex(mpi_all_to_all(torch.view_as_real(x).contiguous()))


def native_distributed_demag(mesh: Mesh, world_size: int, rank: int) -> DistributedDemagField:
    """Distributed Newell demag using the from-scratch CUDA-aware-MPI transpose.

    The same op as ``DistributedDemagField`` with the native collective injected (swappable seam):
    local cuFFT via ``torch.fft`` + the hand-written ``mpi_all_to_all_complex`` transpose, in place
    of the ``torch.distributed`` one. The gloo path is validated for free
    (``test_cross_rank_gradcheck``);
    swapping only the collective here is what the AutoDL run turns on. Needs a CUDA-aware MPI node
    (``mpirun -np N``, ``import mpi4py.MPI`` first); the collective compiles lazily on first call.
    Blocking MPI_Alltoall -> sequential schedule only (the nonblocking MPI_Ialltoall variant is
    deferred: stock OpenMPI's libnbc is NOT CUDA-aware -- verify on the node first).
    """
    return DistributedDemagField(
        mesh, world_size, rank, backend=SyncCollectiveA2A(mpi_all_to_all_complex)
    )
