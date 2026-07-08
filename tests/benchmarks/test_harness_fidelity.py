"""The timing harness's collective must be a faithful mirror of the shipped one (fwd + gradient).

``bench_dist_demag.py`` times a duplicated ``_TimedAllToAll`` (a copy of the shipped
``_AllToAllSingle``) so it can reach the BACKWARD collective, which runs inside autograd where an
outer wrapper cannot see it. The reference-match gate only exercises the forward, so this test pins
the
missing half: the timed collective must match ``all_to_all_complex`` in BOTH forward value and
gradient. A failure here is a real mirror-drift bug (the comm/backward timings would describe a
different computation), not a tolerance to relax.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from benchmarks.bench_dist_demag import CommStats, timed_collective
from spinforge.dist.collectives import all_to_all_complex


def _worker(rank: int, world_size: int, init: str) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=init)
    try:
        torch.manual_seed(rank)  # per-rank data so the all_to_all actually permutes content
        base = torch.randn(2 * world_size, 3, dtype=torch.float64) + 1j * torch.randn(
            2 * world_size, 3, dtype=torch.float64
        )

        def run(collective) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN001
            x = base.clone().requires_grad_(True)
            y = collective(x)
            torch.view_as_real(y).pow(2).sum().backward()  # real scalar loss -> complex grad
            assert x.grad is not None
            return y.detach(), x.grad.detach()

        y_ref, g_ref = run(all_to_all_complex)  # shipped
        y_t, g_t = run(timed_collective(CommStats()))  # harness mirror

        torch.testing.assert_close(y_t, y_ref, rtol=0.0, atol=0.0)  # bit-identical fwd expected
        torch.testing.assert_close(g_t, g_ref, rtol=0.0, atol=0.0)  # and identical gradient
    finally:
        dist.destroy_process_group()


@pytest.mark.dist
def test_timed_collective_matches_shipped_forward_and_gradient() -> None:
    world_size = 2
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(
            _worker, args=(world_size, init), nprocs=world_size, join=True, start_method="spawn"
        )
