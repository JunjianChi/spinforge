"""Differentiable nearest-neighbor halo (ghost-layer) exchange along the distributed z-axis.

The transpose-FFT handles the long-range demag, but the LOCAL stencil terms (exchange Laplacian,
DMI curl) need each cell's z-neighbours -- and at a z-slab boundary those live on the neighbour
rank.
This exchanges one ghost z-layer with each neighbour (Neumann/replicate at the global ends), so a
rank can evaluate its stencil up to the slab edge. The adjoint sends the ghost-layer cotangents back
and accumulates them into the owner's boundary-layer gradient -- the local-term analogue of the
demag's differentiable ``all_to_all``. Non-blocking isend/irecv (irecv first) keeps it
deadlock-free.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


class _ZHalo(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, local: torch.Tensor, world: int, rank: int) -> torch.Tensor:
        ctx.world, ctx.rank = world, rank  # type: ignore[attr-defined]
        first = local[:, :, :1].contiguous()
        last = local[:, :, -1:].contiguous()
        ghost_lo, ghost_hi = torch.empty_like(first), torch.empty_like(last)
        reqs = []
        if rank > 0:
            reqs.append(dist.irecv(ghost_lo, rank - 1))
        if rank < world - 1:
            reqs.append(dist.irecv(ghost_hi, rank + 1))
        if rank > 0:
            reqs.append(dist.isend(first, rank - 1))
        if rank < world - 1:
            reqs.append(dist.isend(last, rank + 1))
        for r in reqs:
            if r is not None:
                r.wait()
        if rank == 0:
            ghost_lo = first  # Neumann (replicate) at the global low boundary
        if rank == world - 1:
            ghost_hi = last
        return torch.cat([ghost_lo, local, ghost_hi], dim=2)

    @staticmethod
    def backward(ctx: object, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        world, rank = ctx.world, ctx.rank  # type: ignore[attr-defined]
        g_lo = grad[:, :, :1].contiguous()  # cotangent of my low ghost (owned by rank-1's last)
        g_hi = grad[:, :, -1:].contiguous()  # cotangent of my high ghost (owned by rank+1's first)
        g_local = grad[:, :, 1:-1].clone()
        recv_lo, recv_hi = torch.zeros_like(g_lo), torch.zeros_like(g_hi)
        reqs = []
        if rank > 0:
            reqs.append(dist.irecv(recv_lo, rank - 1))  # rank-1's g_hi -> my first layer
        if rank < world - 1:
            reqs.append(dist.irecv(recv_hi, rank + 1))  # rank+1's g_lo -> my last layer
        if rank > 0:
            reqs.append(dist.isend(g_lo, rank - 1))
        if rank < world - 1:
            reqs.append(dist.isend(g_hi, rank + 1))
        for r in reqs:
            if r is not None:
                r.wait()
        # fold the ghost cotangents into the owning boundary layer (locally at the global ends)
        g_local[:, :, :1] += g_lo if rank == 0 else recv_lo
        g_local[:, :, -1:] += g_hi if rank == world - 1 else recv_hi
        return g_local, None, None


def z_halo_exchange(local: torch.Tensor, world: int, rank: int) -> torch.Tensor:
    """Pad ``local`` with one z-ghost layer from each neighbour (Neumann at the global ends).

    ``local`` is ``[nx, ny, nz_local, 3]``; output is ``[nx, ny, nz_local + 2, 3]``; differentiable
    (the adjoint returns the ghost cotangents to their owners).
    """
    if local.dim() != 4 or local.size(-1) != 3:
        raise ValueError(
            f"z_halo_exchange expects (nx, ny, nz_local, 3); got shape {tuple(local.shape)}"
        )
    return _ZHalo.apply(local, world, rank)
