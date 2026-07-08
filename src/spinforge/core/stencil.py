"""Per-axis finite-difference arithmetic shared by every stencil field term.

One definition of the narrow-based differences so the three consumers (core exchange,
distributed exchange, chiral exchange+DMI) cannot drift apart. Padding OWNERSHIP deliberately
stays with the caller — replicate (Neumann), halo ghosts, and chiral-BC ghosts are exactly where
the sites legitimately differ, and the distributed layer must feed neighbour ghosts itself.
"""

from __future__ import annotations

import torch


def replicate_pad(m: torch.Tensor, ax: int) -> torch.Tensor:
    """Pad ``m`` by one copied edge layer on each side of ``ax`` (zero normal derivative)."""
    n = m.shape[ax]
    return torch.cat([m.narrow(ax, 0, 1), m, m.narrow(ax, n - 1, 1)], dim=ax)


def second_diff(mp: torch.Tensor, ax: int) -> torch.Tensor:
    """Second difference along ``ax`` of a tensor already padded by one layer per side."""
    n = mp.shape[ax] - 2
    return mp.narrow(ax, 2, n) - 2.0 * mp.narrow(ax, 1, n) + mp.narrow(ax, 0, n)


def central_diff(mp: torch.Tensor, ax: int) -> torch.Tensor:
    """Central first difference (times ``2*dx``) along ``ax`` of a one-layer-padded tensor."""
    n = mp.shape[ax] - 2
    return mp.narrow(ax, 2, n) - mp.narrow(ax, 0, n)
