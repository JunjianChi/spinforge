"""The distributed-adjoint quickstart example must actually run and prove its claim.

Spawns two CPU ranks and checks the example's core: the gradient computed through the distributed
adjoint matches the single-process reference, i.e. autograd survived the rank boundary. A README
example that silently rots is worse than none.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from examples.distributed_adjoint import distributed_adjoint_demo


def _worker(rank: int, world_size: int, init: str) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=init)
    try:
        result = distributed_adjoint_demo(rank, world_size)
        assert result["fwd_rel_err"] < 1e-10, result
        assert result["grad_rel_err"] < 1e-10, result
    finally:
        dist.destroy_process_group()


@pytest.mark.dist
def test_distributed_adjoint_example_matches_reference() -> None:
    world_size = 2
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{os.path.join(d, 'pg')}"
        mp.start_processes(
            _worker, args=(world_size, init), nprocs=world_size, join=True, start_method="spawn"
        )
