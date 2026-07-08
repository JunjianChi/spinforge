"""Distributed effective field on a z-slab: the demag (transpose-FFT, long-range) + the local
stencil terms (exchange, or the coupled exchange+bulk-DMI with the chiral free-surface BC) + zeeman
+ uniaxial anisotropy (pointwise), composed. This is where all the communication -- and the
distributed adjoint -- live; ``DistributedSystem`` wraps it with the purely-local ``m x H`` LLG RHS
+ RK4. It matches the single-process ``System.effective_field`` per slab, gradient included (the
decoupled d-without-exchange combination is refused at construction rather than approximated).

All radius-1 z-stencil terms share ONE depth-1 halo per field evaluation. The chiral free-surface
BC is purely local: the halo delivers replicate ghosts at the global z faces, and the ranks owning
those faces overwrite them with the differentiable chiral ghost -- interior slab boundaries carry
real neighbour data, so no cross-rank machinery beyond the existing halo adjoint is needed.
Adiabatic STT (another radius-1 z-stencil) composes the same way when the inverse design needs it
(deferred).
"""

from __future__ import annotations

import torch

from ..core.anisotropy import uniaxial_anisotropy_field
from ..core.chiral import chiral_exchange_dmi_field, chiral_ghost
from ..core.mesh import Mesh
from ..core.system import Material
from ..core.zeeman import zeeman_field
from .demag import DistributedDemagField
from .exchange import distributed_exchange_field
from .halo import z_halo_exchange


class DistributedEffectiveField:
    """Exchange (halo) + demag (transpose-FFT) + zeeman + anisotropy on a z-slab; differentiable."""

    def __init__(
        self,
        mesh: Mesh,
        material: Material,
        world: int,
        rank: int,
        *,
        h_ext: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        if material.d and not material.a_ex:
            raise ValueError(
                "distributed DMI without exchange is not implemented (the single-process field "
                f"would add a decoupled bulk-DMI term; got d={material.d}, a_ex={material.a_ex}) "
                "-- refusing rather than silently computing exchange-only physics"
            )
        self.mesh, self.mat, self.world, self.rank, self.h_ext = mesh, material, world, rank, h_ext
        self._demag = DistributedDemagField(mesh, world, rank)

    def _chiral_pad_z(self, m_local: torch.Tensor) -> torch.Tensor:
        """One shared depth-1 z-halo, with the chiral ghost applied at the global surfaces.

        The halo returns replicate ghosts at the global ends; the surface-owning ranks replace
        theirs with the chiral-BC ghost computed locally from the edge layer (differentiable, so
        its vjp flows to the edge directly, while the discarded replicate ghost's zero cotangent
        folds harmlessly through the halo adjoint -- pinned by the surface-cell FD probes).
        """
        mat = self.mat
        xi, dz = mat.d / (2.0 * mat.a_ex), self.mesh.dx[2]
        nzl = m_local.shape[2]
        hal = z_halo_exchange(m_local, self.world, self.rank)
        if self.rank == 0:
            ghost = chiral_ghost(m_local.narrow(2, 0, 1), 2, dz, xi, -1)
            hal = torch.cat([ghost, hal.narrow(2, 1, nzl + 1)], dim=2)
        if self.rank == self.world - 1:
            ghost = chiral_ghost(m_local.narrow(2, nzl - 1, 1), 2, dz, xi, +1)
            hal = torch.cat([hal.narrow(2, 0, nzl + 1), ghost], dim=2)
        return hal

    def __call__(self, m_local: torch.Tensor, dku_map: torch.Tensor | None = None) -> torch.Tensor:
        """Effective field on this rank's slab; ``m_local`` is ``[nx, ny, nz_local, 3]``.

        ``dku_map``: an optional graded-anisotropy DESIGN term (J/m^3), broadcastable against
        ``m_local[..., :1]`` -- THIS RANK'S z-slab slice of the global design. Pointwise (no halo),
        differentiable w.r.t. the map: the inverse design optimizes exactly this input.
        """
        mat = self.mat
        if mat.a_ex and mat.d:
            # coupled exchange + bulk DMI under the chiral free-surface BC (System's default)
            h = chiral_exchange_dmi_field(
                m_local,
                self.mesh,
                a=mat.a_ex,
                d=mat.d,
                ms=mat.ms,
                pad_z=self._chiral_pad_z(m_local),
            )
        else:
            h = distributed_exchange_field(
                m_local, self.mesh, mat.a_ex, mat.ms, self.world, self.rank
            )
        h = h + self._demag(m_local, mat.ms)
        if mat.ku:  # uniaxial anisotropy is pointwise (no z-derivative) -> per-slab, no halo
            h = h + uniaxial_anisotropy_field(m_local, self.mesh, mat.ku, mat.ku_axis, mat.ms)
        if dku_map is not None:  # the graded design term rides on top of the material constant
            h = h + uniaxial_anisotropy_field(m_local, self.mesh, dku_map, mat.ku_axis, mat.ms)
        if any(self.h_ext):
            h = h + zeeman_field(m_local, self.h_ext)
        return h
