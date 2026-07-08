"""LLG system: assemble the effective field from the active terms; integrate with fixed-step RK4.

The effective field sums the active terms (exchange, bulk DMI, uniaxial anisotropy, Zeeman, demag).
Time evolution is the Gilbert LLG with |m|=1 enforced each step; the inverse-design regime is
high-damping relaxation, where fixed-step RK4 is adequate and its adjoint is straightforward.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .anisotropy import uniaxial_anisotropy_field
from .chiral import chiral_exchange_dmi_field
from .constants import GAMMA
from .demag import DemagField
from .dmi import bulk_dmi_field
from .exchange import exchange_field
from .mesh import Mesh
from .spin_torque import u_grad_m
from .zeeman import zeeman_field


def gilbert_rhs(m: torch.Tensor, h: torch.Tensor, alpha: float) -> torch.Tensor:
    """Gilbert LLG dm/dt from a precomputed effective field ``h`` (pointwise; no derivatives/MPI).

    Shared by the single-process and distributed solvers -- the arithmetic is identical, only ``h``
    differs (local vs slab). The STT drive is added by the caller (it needs a spatial derivative).
    """
    inv = 1.0 / (1.0 + alpha * alpha)
    m_x_h = torch.linalg.cross(m, h, dim=-1)
    return -GAMMA * inv * (m_x_h + alpha * torch.linalg.cross(m, m_x_h, dim=-1))


def rk4_step(
    rhs: Callable[[torch.Tensor], torch.Tensor], m: torch.Tensor, dt: float
) -> torch.Tensor:
    """One fixed-step RK4 step of ``dm/dt = rhs(m)``, then renormalize |m| = 1 (pointwise; no MPI).

    ``rhs`` carries whatever communication the distributed field needs; the integrator itself is
    pure local arithmetic, so single-process and distributed steppers are one code path.
    """
    k1 = rhs(m)
    k2 = rhs(m + 0.5 * dt * k1)
    k3 = rhs(m + 0.5 * dt * k2)
    k4 = rhs(m + dt * k3)
    m = m + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return m / m.norm(dim=-1, keepdim=True)


@dataclass(frozen=True)
class Material:
    """Material parameters (SI). Zero values switch the corresponding term off."""

    ms: float
    a_ex: float = 0.0
    d: float = 0.0
    ku: float | torch.Tensor = 0.0
    ku_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    alpha: float = 1.0


class System:
    """A micromagnetic system on a ``Mesh`` with a ``Material``: field, LLG RHS, and relax."""

    def __init__(
        self,
        mesh: Mesh,
        material: Material,
        *,
        demag: bool = True,
        h_ext: tuple[float, float, float] = (0.0, 0.0, 0.0),
        u: tuple[float, float, float] = (0.0, 0.0, 0.0),
        chiral_bc: bool = True,
    ) -> None:
        self.mesh = mesh
        self.material = material
        self.h_ext = h_ext
        self.u = u  # adiabatic STT spin-drift velocity (m/s); the current drive
        # True: exchange+DMI share the free-surface BC (correct, default). False: decoupled
        # Neumann/replicate -- a clean uniform background, for idealized tests (e.g. Thiele/Hall).
        self.chiral_bc = chiral_bc
        self._demag = DemagField(mesh) if demag else None

    def effective_field(self, m: torch.Tensor) -> torch.Tensor:
        mat = self.material
        h = torch.zeros_like(m)
        if mat.a_ex and mat.d and self.chiral_bc:
            # exchange + bulk DMI share one free-surface BC; treat them together (edge-consistent)
            h = h + chiral_exchange_dmi_field(m, self.mesh, a=mat.a_ex, d=mat.d, ms=mat.ms)
        else:
            # decoupled terms (Neumann exchange / replicate DMI): a clean uniform background
            if mat.a_ex:
                h = h + exchange_field(m, self.mesh, mat.a_ex, mat.ms)
            if mat.d:
                h = h + bulk_dmi_field(m, self.mesh, mat.d, mat.ms)
        if isinstance(mat.ku, torch.Tensor) or mat.ku:
            h = h + uniaxial_anisotropy_field(m, self.mesh, mat.ku, mat.ku_axis, mat.ms)
        if any(self.h_ext):
            h = h + zeeman_field(m, self.h_ext)
        if self._demag is not None:
            h = h + self._demag(m, mat.ms)
        return h

    def llg_rhs(self, m: torch.Tensor) -> torch.Tensor:
        """Gilbert LLG right-hand side dm/dt, including adiabatic STT when ``u`` is set."""
        a = self.material.alpha
        rhs = gilbert_rhs(m, self.effective_field(m), a)
        if any(self.u):
            # adiabatic Zhang-Li (xi=0), explicit form: -inv*[(u.grad)m + a m x (u.grad)m]
            inv = 1.0 / (1.0 + a * a)
            s = u_grad_m(m, self.mesh, self.u)
            rhs = rhs - inv * (s + a * torch.linalg.cross(m, s, dim=-1))
        return rhs

    def step_rk4(self, m: torch.Tensor, dt: float) -> torch.Tensor:
        """One fixed-step RK4 step, then renormalize |m| = 1."""
        return rk4_step(self.llg_rhs, m, dt)

    def relax(self, m: torch.Tensor, steps: int, dt: float) -> torch.Tensor:
        for _ in range(steps):
            m = self.step_rk4(m, dt)
        return m
