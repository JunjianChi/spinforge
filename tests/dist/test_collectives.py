"""Distributed collective gradients must survive the rank boundary (gloo, CPU multi-rank)."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from spinforge.dist.collectives import (
    a2a_calls,
    all_to_all,
    all_to_all_finish,
    all_to_all_start,
    reset_a2a_calls,
    set_verify_conservation,
)


def _worker(rank: int, world_size: int, init: str) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=init)
    try:
        torch.manual_seed(rank + 1)
        x = torch.randn(world_size, 4, dtype=torch.float64, requires_grad=True)
        y = all_to_all(x)
        # a permutation collective round-trips bit-exactly; state the zero tolerance explicitly
        torch.testing.assert_close(all_to_all(y), x, rtol=0.0, atol=0.0)
        w = torch.randn(world_size, 4, dtype=torch.float64)
        (y * w).sum().backward()
        # backward must equal the adjoint (another all_to_all of the cotangent)
        torch.testing.assert_close(x.grad, all_to_all(w), rtol=0.0, atol=1e-12)

        # fail-loud: a clear error, not the backend's cryptic split failure
        with pytest.raises(ValueError, match="divisible"):
            all_to_all(torch.randn(world_size + 1, 2, dtype=torch.float64))
    finally:
        dist.destroy_process_group()


@pytest.mark.dist
def test_all_to_all_is_differentiable_and_self_adjoint() -> None:
    world_size = 2
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        # spawn (not fork): the parent pytest process runs autograd backward in earlier tests, and
        # fork-based multiprocessing is incompatible with autograd's threads (the child's backward
        # errors). spawn re-imports the worker via its dotted name -> the tests package +
        # pythonpath.
        mp.start_processes(
            _worker, args=(world_size, init), nprocs=world_size, join=True, start_method="spawn"
        )


def _split_worker(rank: int, world_size: int, init: str) -> None:
    """Split-phase pair == the blocking collective, forward and adjoint; misuse fails loudly.

    Wait.backward LAUNCHES the adjoint all_to_all and Start.backward COMPLETES it (the DDP
    launch-early/await-last inversion) -- so the values and the traffic count must be exactly the
    blocking op's, and a second wait on the same handle must raise instead of corrupting."""
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=init)
    try:
        torch.manual_seed(rank + 1)
        x = torch.randn(world_size, 4, dtype=torch.float64, requires_grad=True)
        w = torch.randn(world_size, 4, dtype=torch.float64)
        ref_y = all_to_all(x.detach())
        ref_g = all_to_all(w)  # adjoint of the permutation is the permutation

        reset_a2a_calls()
        ph, slot = all_to_all_start(x)
        y = all_to_all_finish(ph, slot)
        assert a2a_calls() == 1  # start+wait = ONE collective
        torch.testing.assert_close(y, ref_y, rtol=0.0, atol=0.0)

        (y * w).sum().backward()
        assert a2a_calls() == 2  # the adjoint is the second
        torch.testing.assert_close(x.grad, ref_g, rtol=0.0, atol=1e-12)

        with pytest.raises(RuntimeError):
            all_to_all_finish(ph, slot)  # double wait
    finally:
        dist.destroy_process_group()


@pytest.mark.dist
def test_split_phase_all_to_all_matches_blocking_and_guards_misuse() -> None:
    world_size = 2
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(
            _split_worker,
            args=(world_size, init),
            nprocs=world_size,
            join=True,
            start_method="spawn",
        )


def _worker_noncontiguous(rank: int, world_size: int, init: str) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=init)
    try:
        torch.manual_seed(rank + 1)
        # non-contiguous-but-dense input (transpose): the output buffer must not inherit its strides
        xnc = torch.randn(4, world_size, dtype=torch.float64).t()
        assert not xnc.is_contiguous()
        got, ref = all_to_all(xnc), all_to_all(xnc.contiguous())
        torch.testing.assert_close(got, ref, rtol=0.0, atol=0.0)
    finally:
        dist.destroy_process_group()


def _conservation_worker(rank: int, world_size: int, init: str) -> None:
    """The conservation guard is silent on correct collectives and loud on corrupted ones.

    all_to_all permutes (rank, chunk) blocks, so the global payload sum and L1 are invariants of
    every physical collective -- forward, adjoint, and both split-phase legs. A dropped/zeroed chunk
    (the classic cross-rank bookkeeping bug) breaks the invariant and must raise, on every rank, in
    lockstep."""
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=init)
    set_verify_conservation(True)
    try:
        torch.manual_seed(rank + 1)
        x = torch.randn(world_size, 4, dtype=torch.float64, requires_grad=True)
        y = all_to_all(x)  # forward leg under the guard
        (y * y).sum().backward()  # adjoint leg under the guard
        x2 = torch.randn(world_size, 4, dtype=torch.float64, requires_grad=True)
        ph, slot = all_to_all_start(x2)  # split-phase forward leg
        all_to_all_finish(ph, slot).sum().backward()  # split-phase adjoint leg

        real_a2a = dist.all_to_all_single

        def dropped_chunk_a2a(out: torch.Tensor, inp: torch.Tensor, *args: object, **kw: object):
            work = real_a2a(out, inp, *args, **kw)
            out[0].zero_()  # simulate one lost (rank, chunk) block
            return work

        dist.all_to_all_single = dropped_chunk_a2a  # type: ignore[assignment]
        try:
            with pytest.raises(RuntimeError, match="not conserved"):
                all_to_all(x.detach())
        finally:
            dist.all_to_all_single = real_a2a  # type: ignore[assignment]
    finally:
        set_verify_conservation(False)
        dist.destroy_process_group()


@pytest.mark.dist
def test_conservation_guard_catches_corrupted_collective() -> None:
    world_size = 2
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(
            _conservation_worker,
            args=(world_size, init),
            nprocs=world_size,
            join=True,
            start_method="spawn",
        )


@pytest.mark.dist
def test_all_to_all_handles_noncontiguous_input() -> None:
    world_size = 2
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(
            _worker_noncontiguous,
            args=(world_size, init),
            nprocs=world_size,
            join=True,
            start_method="spawn",
        )
