"""The measurement variants: naive-baseline equivalence vehicle and high-watermark perturbations.

Three bench modes the rental protocol needs, pinned on gloo/CPU:

- ``uninstrumented``: sequential schedule on the DEFAULT split-phase backend -- the vehicle for
  the pre-registered naive-baseline equivalence check (blocking instrument vs async start;wait). Its
  comm split is not valid (nothing is instrumented) and collectives are counted by the a2a
  counter instead.
- ``--perturb no-comm``: identity "collective" -- full compute, zero wire. Wall time bounds the
  compute+pack share; measured collectives must be ZERO.
- ``--perturb no-fft``: shape-faithful, graph-preserving FFT stand-ins -- full wire traffic,
  trivial compute. Wall time bounds the comm share; the collective count must be UNCHANGED
  (12 for fwd+adjoint), including the backward ones (the stand-ins must not sever autograd).

Perturbed physics is wrong by construction, so the harness must SKIP the reference-match gate for
them
and say so in the summary (a timing bound, never a physics run).
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from benchmarks.bench_dist_demag import Config, build_field, no_fft_stand_ins
from spinforge.dist.collectives import TorchAsyncA2A, a2a_calls, all_reduce_sum, reset_a2a_calls


def _cfg(**kw: object) -> Config:
    base: dict = dict(
        n=(4, 4, 4), dtype=torch.float64, reps=1, warmup=0, seed=0, link_bw=None, device="cpu"
    )
    base.update(kw)
    return Config(**base)  # type: ignore[arg-type]


def _fwd_adj(field) -> None:  # noqa: ANN001
    m = torch.randn(4, 4, 2, 3, dtype=torch.float64).requires_grad_(True)
    all_reduce_sum((field(m, 8e5) ** 2).sum()).backward()


def _worker(rank: int, world: int, init: str) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=init)
    try:
        from benchmarks.bench_dist_demag import CommStats
        from spinforge.core.mesh import Mesh

        mesh = Mesh(n=(4, 4, 4), dx=(2e-9, 2e-9, 2e-9))

        # uninstrumented: default split-phase backend, comm split invalid, a2a-counted
        cs = CommStats()
        field, split_valid = build_field(_cfg(uninstrumented=True), cs, mesh, world, rank)
        assert split_valid is False
        assert isinstance(field._backend, TorchAsyncA2A)
        reset_a2a_calls()
        _fwd_adj(field)
        assert cs.calls() == 0 and a2a_calls() == 12

        # no-comm: identity collective -> zero measured collectives, compute intact
        cs = CommStats()
        field, split_valid = build_field(_cfg(perturb="no-comm"), cs, mesh, world, rank)
        assert split_valid is False
        reset_a2a_calls()
        _fwd_adj(field)
        assert cs.calls() == 0 and a2a_calls() == 0

        # mixed wire: the f32-wire collective is the backend; traffic still counted (12)
        from spinforge.dist.collectives import SyncCollectiveA2A, all_to_all_complex_mixed

        cs = CommStats()
        field, split_valid = build_field(_cfg(mixed_wire=True), cs, mesh, world, rank)
        assert split_valid is False
        assert isinstance(field._backend, SyncCollectiveA2A)
        assert field._backend.collective is all_to_all_complex_mixed
        reset_a2a_calls()
        _fwd_adj(field)
        assert a2a_calls() == 12

        # no-fft: stand-ins keep shapes AND the autograd graph -> traffic unchanged (12), incl bwd
        cs = CommStats()
        field, split_valid = build_field(_cfg(perturb="no-fft"), cs, mesh, world, rank)
        reset_a2a_calls()
        with no_fft_stand_ins():
            _fwd_adj(field)
        assert a2a_calls() == 12, a2a_calls()
    finally:
        dist.destroy_process_group()


@pytest.mark.dist
def test_bench_variants_traffic_contracts() -> None:
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(_worker, args=(2, init), nprocs=2, join=True, start_method="spawn")


def test_perturbed_configs_skip_the_gate() -> None:
    # the gate is physics; a perturbed run is a timing bound -- gating it would either fail
    # spuriously (no-fft) or validate nothing (no-comm)
    assert _cfg(perturb="no-fft").gate_required is False
    assert _cfg(perturb="no-comm").gate_required is False
    assert _cfg().gate_required is True
    assert _cfg(uninstrumented=True).gate_required is True  # real physics, still gated
    assert _cfg(mixed_wire=True).gate_required is True  # gated, at the f32-class tolerance
