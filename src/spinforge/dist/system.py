"""Distributed LLG solver on a z-slab: the full forward+adjoint solve, composed from the pieces.

``DistributedSystem`` is the deployable end-to-end object: it wraps the
``DistributedEffectiveField``
(exchange-halo + transpose-FFT demag + zeeman + anisotropy) with the pointwise Gilbert LLG RHS and
the
RK4 + |m|=1 relaxation -- reusing the SAME ``gilbert_rhs`` / ``rk4_step`` as the single-process
``System`` (the integrator arithmetic is MPI-free by design; only the effective field communicates).
So
a distributed relaxation is per-rank RK4 whose field evaluations exchange halos and all-to-all
transposes, and the gradient of the whole multi-step solve flows back across ranks through those
differentiable collectives -- the end-to-end distributed adjoint.

Backend-swappable seam: the demag transpose collective and the FFT backend are injected into the
field, so gloo->NCCL and torch-FFT->native-op are drop-in with no change here.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from ..core.mesh import Mesh
from ..core.system import Material, gilbert_rhs, rk4_step
from .field import DistributedEffectiveField


class DistributedSystem:
    """Full distributed LLG solve on a rank's z-slab (field + RK4 relax), forward and adjoint."""

    def __init__(
        self,
        mesh: Mesh,
        material: Material,
        world: int,
        rank: int,
        *,
        h_ext: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.mesh, self.material, self.world, self.rank = mesh, material, world, rank
        self._field = DistributedEffectiveField(mesh, material, world, rank, h_ext=h_ext)

    def llg_rhs(self, m_local: torch.Tensor, dku_map: torch.Tensor | None = None) -> torch.Tensor:
        """Gilbert LLG dm/dt on this rank's slab (no STT term)."""
        return gilbert_rhs(m_local, self._field(m_local, dku_map), self.material.alpha)

    def step_rk4(
        self, m_local: torch.Tensor, dt: float, dku_map: torch.Tensor | None = None
    ) -> torch.Tensor:
        """One distributed RK4 step + |m|=1 renorm; every field eval keeps the ranks in lockstep."""
        return rk4_step(lambda mm: self.llg_rhs(mm, dku_map), m_local, dt)

    def relax(
        self,
        m_local: torch.Tensor,
        steps: int,
        dt: float,
        *,
        checkpoint_every: int = 0,
        dku_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Relax this rank's slab for ``steps`` RK4 steps; differentiable end to end across ranks.

        ``checkpoint_every=k`` bounds the adjoint's activation memory: the unrolled graph is split
        into k-step chunks whose intermediates are recomputed in the backward instead of stored
        (the temporal analogue of ``dist/checkpoint.py``).
        The recompute re-runs each chunk's halo/transpose collectives INSIDE the backward pass;
        every rank derives identical chunk boundaries from (steps, k), so the recomputes stay in
        lockstep. Forward values and gradients are unchanged -- only memory and collective traffic
        (bounded by one extra forward per chunk) differ. ``0`` keeps the fully stored graph.
        """
        needs_grad = m_local.requires_grad or (dku_map is not None and dku_map.requires_grad)
        use_ckpt = checkpoint_every > 0 and torch.is_grad_enabled() and needs_grad
        if not use_ckpt:
            for _ in range(steps):
                m_local = self.step_rk4(m_local, dt, dku_map)
            return m_local

        # dku is an EXPLICIT checkpoint argument (not a closure capture) so the recompute's saved-
        # tensor tracking sees it and its design gradient survives the chunked backward.
        def run_chunk(m_in: torch.Tensor, k: int, dku: torch.Tensor | None) -> torch.Tensor:
            for _ in range(k):
                m_in = self.step_rk4(m_in, dt, dku)
            return m_in

        done = 0
        while done < steps:
            k = min(checkpoint_every, steps - done)
            m_local = checkpoint(run_chunk, m_local, k, dku_map, use_reentrant=False)
            done += k
        return m_local
