"""The native (CUDA-aware-MPI) distributed demag: free wiring proof + the AutoDL reference gate.

The wiring test runs anywhere (no MPI/CUDA): it proves the factory injects the native collective
into the same DistributedDemagField the gloo path validates (swappable-backend seam), so only the
collective differs
on the box. The reference-match test is the multi-GPU gate itself -- launched under mpirun on a
CUDA-aware-MPI node; it is box-pending (the native collective compiles + runs only there).
"""

from __future__ import annotations

import pytest
import torch

from spinforge.core.demag import DemagField
from spinforge.core.mesh import Mesh
from spinforge.dist.collectives import SyncCollectiveA2A
from spinforge.dist.demag import DistributedDemagField
from spinforge.dist.native_collectives import (
    mpi_all_to_all,
    mpi_all_to_all_complex,
    mpi_ialltoall_probe,
    mpi_pairwise_all_to_all_raw,
    native_distributed_demag,
)


def test_native_factory_wires_the_mpi_collective() -> None:
    """Swappable-backend wiring (free, no MPI/CUDA): the factory is the generic distributed demag
    with the
    native collective injected -- so the AutoDL run only swaps the transpose backend."""
    mesh = Mesh(n=(4, 4, 4), dx=(2e-9, 2e-9, 2e-9))
    field = native_distributed_demag(mesh, world_size=2, rank=0)
    assert isinstance(field, DistributedDemagField)
    assert isinstance(field._backend, SyncCollectiveA2A)
    assert field._backend.collective is mpi_all_to_all_complex


@pytest.mark.multigpu
def test_native_distributed_matches_reference() -> None:
    """AutoDL gate: the native distributed demag reproduces the single-process reference per slab.

    Launch as ``mpirun -np N python -m pytest -m multigpu tests/dist/test_native_distributed.py``.
    Box-pending: the CUDA-aware-MPI collective compiles + runs only on a multi-GPU node.
    """
    pytest.importorskip("mpi4py")
    from mpi4py import MPI  # importing initializes MPI (the native op checks MPI_Initialized)

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    comm = MPI.COMM_WORLD
    world_size, rank = comm.Get_size(), comm.Get_rank()
    if world_size < 2:
        pytest.skip("needs >=2 MPI ranks (mpirun -np 2+)")

    # nz scales with the world size (the slab op requires nz % world == 0) -- a fixed 4-cell
    # grid made this test fail at np=8 for grid reasons, not physics (found on the H20 node)
    nx, ny, nz, ms = 4, 4, max(8, world_size), 8e5
    mesh = Mesh(n=(nx, ny, nz), dx=(2e-9, 2e-9, 2e-9))
    torch.manual_seed(0)  # same m on every rank
    m_full = torch.randn(nx, ny, nz, 3, dtype=torch.float64, device="cuda")
    m_full = m_full / m_full.norm(dim=-1, keepdim=True)
    ref = DemagField(mesh)(m_full, ms)  # single-process reference on this rank's GPU

    nzl = nz // world_size
    m_local = m_full[:, :, rank * nzl : (rank + 1) * nzl].contiguous()
    h_local = native_distributed_demag(mesh, world_size, rank)(m_local, ms)
    torch.testing.assert_close(
        h_local, ref[:, :, rank * nzl : (rank + 1) * nzl], rtol=1e-9, atol=1e-6
    )


@pytest.mark.multigpu
def test_nonblocking_and_pairwise_match_blocking() -> None:
    """The Ialltoall CUDA-awareness probe + the pairwise fallback, against the blocking op.

    Launch under ``mpirun -np 2+``. Three outcomes matter: (a) all three agree -> the machine's
    MPI has a CUDA-aware nonblocking path and the overlap variant may use MPI_Ialltoall;
    (b) the probe crashes/corrupts while pairwise agrees -> build the overlap on Isend/Irecv;
    (c) pairwise disagrees too -> the MPI build is not CUDA-aware at all, stop here.
    """
    pytest.importorskip("mpi4py")
    from mpi4py import MPI

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    comm = MPI.COMM_WORLD
    world_size, rank = comm.Get_size(), comm.Get_rank()
    if world_size < 2:
        pytest.skip("needs >=2 MPI ranks (mpirun -np 2+)")

    torch.manual_seed(rank + 1)  # per-rank content so the transpose really permutes data
    x = torch.randn(world_size * 3, 5, dtype=torch.float64, device="cuda")

    y_blocking = mpi_all_to_all(x.clone())
    y_probe = mpi_ialltoall_probe(x.clone())
    y_pairwise = mpi_pairwise_all_to_all_raw(x.clone())
    torch.testing.assert_close(y_pairwise, y_blocking, rtol=0.0, atol=0.0)
    torch.testing.assert_close(y_probe, y_blocking, rtol=0.0, atol=0.0)
