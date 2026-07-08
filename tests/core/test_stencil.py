"""The shared per-axis stencil arithmetic: one definition for every finite-difference site.

Three field terms (core exchange, distributed exchange, chiral exchange+DMI) apply the same
narrow-based differences to an already-padded tensor; this module pins the shared arithmetic so
the sites cannot drift apart. Padding OWNERSHIP stays at each site (replicate vs halo vs chiral
ghosts is exactly where they legitimately differ).
"""

from __future__ import annotations

import torch

from spinforge.core.stencil import central_diff, replicate_pad, second_diff


def test_replicate_pad_copies_the_edge_layers() -> None:
    m = torch.arange(24, dtype=torch.float64).reshape(4, 3, 2, 1)
    mp = replicate_pad(m, ax=0)
    assert mp.shape == (6, 3, 2, 1)
    torch.testing.assert_close(mp.narrow(0, 0, 1), m.narrow(0, 0, 1), rtol=0.0, atol=0.0)
    torch.testing.assert_close(mp.narrow(0, 5, 1), m.narrow(0, 3, 1), rtol=0.0, atol=0.0)


def test_second_diff_matches_the_manual_stencil() -> None:
    torch.manual_seed(0)
    m = torch.randn(5, 4, 3, 3, dtype=torch.float64)
    for ax in range(3):
        mp = replicate_pad(m, ax=ax)
        n = m.shape[ax]
        want = mp.narrow(ax, 2, n) - 2.0 * mp.narrow(ax, 1, n) + mp.narrow(ax, 0, n)
        torch.testing.assert_close(second_diff(mp, ax=ax), want, rtol=0.0, atol=0.0)


def test_central_diff_matches_the_manual_stencil() -> None:
    torch.manual_seed(1)
    m = torch.randn(4, 4, 4, 3, dtype=torch.float64)
    mp = replicate_pad(m, ax=1)
    want = mp.narrow(1, 2, 4) - mp.narrow(1, 0, 4)
    torch.testing.assert_close(central_diff(mp, ax=1), want, rtol=0.0, atol=0.0)


def test_replicate_pad_second_diff_is_zero_flux_at_the_boundary() -> None:
    """Replicate padding makes the boundary second difference one-sided (Neumann/zero-flux)."""
    m = torch.arange(5, dtype=torch.float64).reshape(5, 1, 1, 1).expand(5, 1, 1, 3).contiguous()
    d2 = second_diff(replicate_pad(m, ax=0), ax=0)
    # interior of a linear ramp: exactly zero; boundary: one-sided +-1 (the folded ghost)
    torch.testing.assert_close(d2[1:-1], torch.zeros_like(d2[1:-1]), rtol=0.0, atol=0.0)
    assert float(d2[0, 0, 0, 0]) == 1.0 and float(d2[-1, 0, 0, 0]) == -1.0
