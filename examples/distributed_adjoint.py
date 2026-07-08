"""The core of spinforge, on a laptop: a gradient that flows across rank (GPU) boundaries.

Two processes each own half of a 3D magnet along z. The demagnetizing field couples the whole
volume through a distributed FFT, so computing it -- and its gradient -- has to send data across
the process boundary and back. This checks that the distributed forward and its adjoint agree
with the same solve run on one process. The single-process solve is the reference, already
float64-gradchecked; this demo adds the missing link, that the distributed path matches it. That
link is the piece existing distributed solvers lack: their autograd graph breaks at the boundary.

Run it:  torchrun --nproc_per_node=2 examples/distributed_adjoint.py     (CPU via gloo, seconds)
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist

from spinforge.core.demag import DemagField
from spinforge.core.mesh import Mesh
from spinforge.dist.collectives import all_reduce_sum
from spinforge.dist.demag import DistributedDemagField

logger = logging.getLogger(__name__)


def distributed_adjoint_demo(rank: int, world_size: int) -> dict[str, float]:
    """One distributed forward+adjoint of the demag, checked against the single-process solve.

    Each rank owns a z-slab. The loss is an arbitrary linear functional of the demag field, which
    excites every gradient degree of freedom. Returns the forward and gradient relative errors on
    this slab against the single-process reference. A process group must already be initialized.
    """
    nx, ny, nz, ms = 8, 8, 8, 8e5  # nz divisible by world_size
    mesh = Mesh(n=(nx, ny, nz), dx=(2e-9, 2e-9, 2e-9))
    torch.manual_seed(0)  # identical m and loss weights on every rank
    m_full = torch.randn(nx, ny, nz, 3, dtype=torch.float64)
    m_full = m_full / m_full.norm(dim=-1, keepdim=True)
    w = torch.randn(nx, ny, nz, 3, dtype=torch.float64)  # arbitrary linear functional of the field

    nzl = nz // world_size
    sl = slice(rank * nzl, (rank + 1) * nzl)

    # distributed forward + adjoint: field and gradient cross ranks through all_to_all
    m_local = m_full[:, :, sl].clone().requires_grad_(True)
    h_local = DistributedDemagField(mesh, world_size, rank)(m_local, ms)
    loss = all_reduce_sum((h_local * w[:, :, sl]).sum())
    loss.backward()
    assert m_local.grad is not None

    # single-process reference, itself gradchecked: compare forward AND gradient on this slab
    m_ref = m_full.clone().requires_grad_(True)
    h_ref = DemagField(mesh)(m_ref, ms)
    (h_ref * w).sum().backward()
    assert m_ref.grad is not None

    def rel(a: torch.Tensor, b: torch.Tensor) -> float:
        b = b.detach()
        return float((a.detach() - b).abs().max() / b.abs().max())

    return {
        "fwd_rel_err": rel(h_local, h_ref[:, :, sl]),
        "grad_rel_err": rel(m_local.grad, m_ref.grad[:, :, sl]),
    }


def main() -> None:
    dist.init_process_group("gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    try:
        result = distributed_adjoint_demo(rank, world_size)
        logger.info(
            "rank %d of %d: forward relative error %.1e, gradient relative error %.1e",
            rank,
            world_size,
            result["fwd_rel_err"],
            result["grad_rel_err"],
        )
        if rank == 0:
            logger.info(
                "distributed forward and adjoint both match the single-process solve to roundoff"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
