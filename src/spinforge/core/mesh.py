"""A regular finite-difference grid."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Mesh:
    """A regular finite-difference grid: ``n`` cells with spacing ``dx`` (metres) from ``origin``.

    Fields live on cell centres; the magnetization tensor has shape ``(nx, ny, nz, 3)``.
    """

    n: tuple[int, int, int]
    dx: tuple[float, float, float]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def cell_volume(self) -> float:
        return self.dx[0] * self.dx[1] * self.dx[2]
