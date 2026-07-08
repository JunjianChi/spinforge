"""The distributed slab demag reproduces the single-process demag on each rank's slab (gloo, CPU).

This is the reference-match gate for the distributed forward demag.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from spinforge.core.demag import DemagField
from spinforge.core.mesh import Mesh
from spinforge.dist.collectives import SyncCollectiveA2A, all_to_all_complex
from spinforge.dist.demag import DistributedDemagField


def _worker(rank: int, world_size: int, init: str, inject: bool) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=init)
    try:
        nx, ny, nz, ms = 4, 4, 4, 8e5
        mesh = Mesh(n=(nx, ny, nz), dx=(2e-9, 2e-9, 2e-9))
        torch.manual_seed(0)
        m_full = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        m_full = m_full / m_full.norm(dim=-1, keepdim=True)
        ref = DemagField(mesh)(m_full, ms)  # single-process reference, full grid

        # inject a wrapped collective backend to prove the swappable-backend seam routes through it
        # (the
        # native op slots in the same way); default path uses the split-phase torch backend
        calls = [0]

        def counting_collective(x: torch.Tensor) -> torch.Tensor:
            calls[0] += 1
            return all_to_all_complex(x)

        if inject:
            field = DistributedDemagField(
                mesh, world_size, rank, backend=SyncCollectiveA2A(counting_collective)
            )
        else:
            field = DistributedDemagField(mesh, world_size, rank)

        nzl = nz // world_size
        m_local = m_full[:, :, rank * nzl : (rank + 1) * nzl].contiguous()
        h_local = field(m_local, ms)
        ref_local = ref[:, :, rank * nzl : (rank + 1) * nzl]
        torch.testing.assert_close(h_local, ref_local, rtol=1e-9, atol=1e-6)
        if inject:
            assert calls[0] == 6, f"injected collective not used ({calls[0]} calls, expected 6)"
    finally:
        dist.destroy_process_group()


def _run(world_size: int, inject: bool) -> None:
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(
            _worker,
            args=(world_size, init, inject),
            nprocs=world_size,
            join=True,
            start_method="spawn",
        )


def _worker_f32(rank: int, world_size: int, init: str) -> None:
    """f32 input -> f32 output (no silent f64 upcast) AND matches the f32 single-process slab."""
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=init)
    try:
        nx, ny, nz, ms = 4, 4, 4, 8e5
        mesh = Mesh(n=(nx, ny, nz), dx=(2e-9, 2e-9, 2e-9))
        torch.manual_seed(0)
        m_full = torch.randn(nx, ny, nz, 3, dtype=torch.float32)
        m_full = m_full / m_full.norm(dim=-1, keepdim=True)
        ref = DemagField(mesh)(m_full, ms)
        assert ref.dtype == torch.float32
        field = DistributedDemagField(mesh, world_size, rank)
        nzl = nz // world_size
        m_local = m_full[:, :, rank * nzl : (rank + 1) * nzl].contiguous()
        h_local = field(m_local, ms)
        # output dtype must follow the input, not the internally-built kernel
        assert h_local.dtype == torch.float32, h_local.dtype
        # |H_demag| ~ Ms/3 ~ 2.7e5 A/m here; atol=2.0 A/m (~7e-6 rel) covers the f32 transpose-vs-
        # fftn reduction-order gap, rtol=1e-4 the mantissa (documented tolerances).
        torch.testing.assert_close(
            h_local, ref[:, :, rank * nzl : (rank + 1) * nzl], rtol=1e-4, atol=2.0
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.dist
def test_distributed_demag_f32_dtype_faithful_and_matches() -> None:
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(_worker_f32, args=(2, init), nprocs=2, join=True, start_method="spawn")


def _kernel_worker(rank: int, world_size: int, init: str) -> None:
    """Owner-computes kernel build: no full-domain FFT on any rank, slabs match the reference.

    Anisotropic mesh (nx != ny != nz AND dx != dy != dz) so a component-permutation typo cannot
    hide behind symmetry. The reference kernel is built BEFORE the fftn spy is installed; the
    distributed build must then never push a full padded (2nx, 2ny, 2nz) array through an FFT --
    that is the allocation that OOMs every rank at the science target.
    """
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=init)
    try:
        mesh = Mesh(n=(4, 6, 8), dx=(1e-9, 2e-9, 3e-9))
        nx, ny, nz = mesh.n
        ref = DemagField(mesh)
        ref._build(torch.float64, torch.device("cpu"))
        assert ref._kernel is not None

        full_shape = (2 * nx, 2 * ny, 2 * nz)
        seen: list[tuple[int, ...]] = []
        orig_fftn = torch.fft.fftn

        def spying_fftn(x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            seen.append(tuple(x.shape))
            return orig_fftn(x, *args, **kwargs)  # type: ignore[arg-type]

        torch.fft.fftn = spying_fftn  # type: ignore[assignment]
        try:
            field = DistributedDemagField(mesh, world_size, rank)
            torch.manual_seed(0)
            m_full = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
            m_full = m_full / m_full.norm(dim=-1, keepdim=True)
            nzl = nz // world_size
            field(m_full[:, :, rank * nzl : (rank + 1) * nzl].contiguous(), 8e5)
        finally:
            torch.fft.fftn = orig_fftn  # type: ignore[assignment]

        assert full_shape not in seen, f"full-domain FFT during distributed build: {seen}"

        # reference-match for the kernel itself: each slab equals the reference kernel reshaped +
        # sliced.
        # f64; the only difference is fft2*fft vs fftn reduction order -> tight tolerances.
        assert field._k is not None
        xy, xyp = (2 * nx) * (2 * ny), (2 * nx) * (2 * ny) // world_size
        for ab, k_slab in field._k.items():
            want = ref._kernel[ab].reshape(xy, 2 * nz)[rank * xyp : (rank + 1) * xyp]
            torch.testing.assert_close(k_slab, want, rtol=1e-9, atol=1e-12)
    finally:
        dist.destroy_process_group()


@pytest.mark.dist
def test_kernel_build_is_owner_computes_and_matches_reference() -> None:
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(
            _kernel_worker, args=(2, init), nprocs=2, join=True, start_method="spawn"
        )


def test_quasi_2d_mesh_rejected_at_construction() -> None:
    """A singleton axis silently changes the padded-kernel layout -- refuse it loudly, at init."""
    with pytest.raises(ValueError):
        DistributedDemagField(Mesh(n=(4, 1, 4), dx=(2e-9, 2e-9, 2e-9)), 2, 0)


def _rfft_worker(rank: int, world_size: int, init: str) -> None:
    """The rfft transpose path: half the wire bytes, same field, FD-checked adjoint elsewhere.

    Pins, on an anisotropic mesh (the packed [:, :ny+1, :] kernel index expression is the typo
    risk): (a) reference-match -- the rfft distributed demag matches the full-complex single-process
    reference at the pinned f64 tolerances; (b) the packed kernel slab equals the reference
    kernel's Hermitian-non-redundant block, reshaped and sliced; (c) every transposed operand
    carries 2nx*(ny+1) plane rows, not 4*nx*ny -- the ~2x wire-byte halving, observed not assumed.
    """
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=init)
    try:
        mesh = Mesh(n=(4, 6, 8), dx=(1e-9, 2e-9, 3e-9))
        nx, ny, nz, ms = *mesh.n, 8e5
        ref_field = DemagField(mesh)
        torch.manual_seed(0)
        m_full = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        m_full = m_full / m_full.norm(dim=-1, keepdim=True)
        ref = ref_field(m_full, ms)  # full-complex single-process reference

        shapes: list[tuple[int, ...]] = []

        def recording_collective(x: torch.Tensor) -> torch.Tensor:
            shapes.append(tuple(x.shape))
            return all_to_all_complex(x)

        field = DistributedDemagField(
            mesh, world_size, rank, backend=SyncCollectiveA2A(recording_collective), use_rfft=True
        )
        nzl = nz // world_size
        m_local = m_full[:, :, rank * nzl : (rank + 1) * nzl].contiguous()
        h_local = field(m_local, ms)

        # (a) reference-match vs the full-complex reference (pinned tolerances, decisions
        # 2026-07-06)
        torch.testing.assert_close(
            h_local, ref[:, :, rank * nzl : (rank + 1) * nzl], rtol=1e-9, atol=1e-6
        )

        # (b) packed kernel slab == reference kernel's [:, :ny+1, :] block, reshaped + sliced
        assert ref_field._kernel is not None and field._k is not None
        rows, rowsp = (2 * nx) * (ny + 1), (2 * nx) * (ny + 1) // world_size
        for ab, k_slab in field._k.items():
            packed = ref_field._kernel[ab][:, : ny + 1, :].reshape(rows, 2 * nz)
            want = packed[rank * rowsp : (rank + 1) * rowsp]
            torch.testing.assert_close(k_slab, want, rtol=1e-9, atol=1e-12)

        # (c) the apply-path wire operand is the packed plane: rows/P per chunk, never 4nxny/P
        full_rows_chunk = (2 * nx) * (2 * ny) // world_size
        assert shapes, "recording collective never used"
        assert all(s[1] == rowsp for s in shapes), (shapes, rowsp)
        assert all(s[1] != full_rows_chunk for s in shapes)
    finally:
        dist.destroy_process_group()


@pytest.mark.dist
def test_rfft_demag_matches_reference_and_halves_wire() -> None:
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(_rfft_worker, args=(2, init), nprocs=2, join=True, start_method="spawn")


def test_rfft_plane_indivisible_by_world_rejected() -> None:
    # 2nx*(ny+1) = 6*5 = 30, world 4: the packed plane cannot be chunked -> loud, at init
    with pytest.raises(ValueError):
        DistributedDemagField(Mesh(n=(3, 4, 8), dx=(2e-9, 2e-9, 2e-9)), 4, 0, use_rfft=True)


def _sched_worker(rank: int, world_size: int, init: str) -> None:
    """The pipelined schedule is the SAME computation as the sequential naive schedule, differently
    ordered:
    forward bit-identical, gradient equal to accumulation-order round-off. Overlap is a timing
    property (measured on the GPU node); correctness must not depend on it (gloo wait blocks)."""
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=init)
    try:
        mesh = Mesh(n=(4, 6, 8), dx=(1e-9, 2e-9, 3e-9))
        nx, ny, nz, ms = *mesh.n, 8e5
        torch.manual_seed(0)
        m_full = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        m_full = m_full / m_full.norm(dim=-1, keepdim=True)
        w = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
        nzl = nz // world_size
        sl = slice(rank * nzl, (rank + 1) * nzl)

        def run(schedule: str) -> tuple[torch.Tensor, torch.Tensor]:
            field = DistributedDemagField(mesh, world_size, rank, schedule=schedule)
            m = m_full[:, :, sl].clone().requires_grad_(True)
            h = field(m, ms)
            (h * w[:, :, sl]).sum().backward()  # symmetric per-rank loss keeps ranks in lockstep
            assert m.grad is not None
            return h.detach(), m.grad

        h_seq, g_seq = run("sequential")
        h_pipe, g_pipe = run("pipelined")
        torch.testing.assert_close(h_pipe, h_seq, rtol=0.0, atol=0.0)
        torch.testing.assert_close(g_pipe, g_seq, rtol=1e-13, atol=1e-14)
    finally:
        dist.destroy_process_group()


@pytest.mark.dist
def test_pipelined_schedule_matches_sequential() -> None:
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(_sched_worker, args=(2, init), nprocs=2, join=True, start_method="spawn")


def test_unknown_schedule_rejected() -> None:
    with pytest.raises(ValueError):
        DistributedDemagField(
            Mesh(n=(4, 4, 4), dx=(2e-9, 2e-9, 2e-9)), 2, 0, schedule="overlapped-magic"
        )


@pytest.mark.dist
def test_distributed_demag_matches_single_process() -> None:
    _run(world_size=2, inject=False)


@pytest.mark.dist
def test_injected_collective_is_used_and_correct() -> None:
    """Swappable backend: an injected collective is the only transpose path and preserves the
    reference match,
    so the native CUDA-aware-MPI collective slots in with no change to the demag op."""
    _run(world_size=2, inject=True)
