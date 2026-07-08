"""Zeeman field: the externally applied field h_ext (A/m), independent of m."""

from __future__ import annotations

import torch


def zeeman_field(m: torch.Tensor, h_ext: tuple[float, float, float]) -> torch.Tensor:
    """Applied field (A/m) broadcast over the grid; shape matches ``m`` = ``(nx, ny, nz, 3)``."""
    h = torch.tensor(h_ext, dtype=m.dtype, device=m.device)
    return h.broadcast_to(m.shape)
