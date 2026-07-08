"""Unit tests for the comm-roofline calculator (hand-computed oracles, explicit rtol/atol)."""

from __future__ import annotations

import pytest
import torch

from benchmarks.roofline import (
    COLLECTIVES_PER_APPLY,
    comm_time_lower_bound,
    elements_per_collective,
    fft_throughput_roofline,
    fit_alpha_beta,
    roofline_fraction,
    wire_bytes_per_rank,
)


def test_elements_per_collective_hand_value() -> None:
    # 4x4x4 grid, P=2: padded xy plane 8*8=64, local slab nz/P=2 -> 128 complex elements
    assert elements_per_collective((4, 4, 4), 2) == 128


def test_elements_rejects_indivisible_slab() -> None:
    with pytest.raises(ValueError):
        elements_per_collective((4, 4, 5), 2)


def test_wire_bytes_hand_value_f64_and_f32() -> None:
    # f64 -> complex128 (16 B): 128 el * 16 B = 2048 B local; (P-1)/P = 1/2 -> 1024 B on the wire
    assert wire_bytes_per_rank((4, 4, 4), 2, torch.float64) == 1024
    # f32 -> complex64 halves it
    assert wire_bytes_per_rank((4, 4, 4), 2, torch.float32) == 512


def test_wire_bytes_rejects_non_float() -> None:
    with pytest.raises(ValueError):
        wire_bytes_per_rank((4, 4, 4), 2, torch.complex128)


def test_comm_lower_bound_hand_value() -> None:
    # 6 collectives * 1024 B / 1e9 B/s = 6.144e-6 s
    lb = comm_time_lower_bound((4, 4, 4), 2, torch.float64, 6, 1e9)
    torch.testing.assert_close(torch.tensor(lb), torch.tensor(6.144e-6), rtol=1e-12, atol=1e-18)


def test_rfft_wire_accounting_hand_values() -> None:
    # rfft plane (2nx, ny+1) = (8, 5): 8*5*(4/2) = 80 complex elements (vs 128 full-complex);
    # f64 wire: 80*16 = 1280 B local * 1/2 off-rank = 640 B (vs 1024) -- the (ny+1)/(2ny) halving
    assert elements_per_collective((4, 4, 4), 2, use_rfft=True) == 80
    assert wire_bytes_per_rank((4, 4, 4), 2, torch.float64, use_rfft=True) == 640
    # default stays full-complex (backward compatible)
    assert elements_per_collective((4, 4, 4), 2) == 128


def test_comm_lower_bound_alpha_term_hand_value() -> None:
    # per collective: (P-1)*alpha + bytes/B = 1*1e-6 + 1024/1e9 = 2.024e-6 s; x6 = 1.2144e-5 s
    lb = comm_time_lower_bound((4, 4, 4), 2, torch.float64, 6, 1e9, alpha=1e-6)
    torch.testing.assert_close(torch.tensor(lb), torch.tensor(1.2144e-5), rtol=1e-12, atol=1e-18)


def test_comm_lower_bound_alpha_zero_is_beta_only() -> None:
    beta_only = comm_time_lower_bound((4, 4, 4), 2, torch.float64, 6, 1e9)
    with_zero = comm_time_lower_bound((4, 4, 4), 2, torch.float64, 6, 1e9, alpha=0.0)
    torch.testing.assert_close(torch.tensor(with_zero), torch.tensor(beta_only), rtol=0.0, atol=0.0)


def test_comm_lower_bound_rejects_negative_alpha() -> None:
    with pytest.raises(ValueError):
        comm_time_lower_bound((4, 4, 4), 2, torch.float64, 6, 1e9, alpha=-1e-6)


def test_fit_alpha_beta_recovers_synthetic_model() -> None:
    # exact t = alpha + b/beta data must recover (alpha, beta) up to float round-off
    alpha, beta = 5e-6, 100e9
    sizes = [1_000, 100_000, 10_000_000, 1_000_000_000]
    times = [alpha + b / beta for b in sizes]
    a_fit, b_fit = fit_alpha_beta(sizes, times)
    torch.testing.assert_close(torch.tensor(a_fit), torch.tensor(alpha), rtol=1e-6, atol=1e-12)
    torch.testing.assert_close(torch.tensor(b_fit), torch.tensor(beta), rtol=1e-6, atol=0.0)


def test_fit_alpha_beta_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError):
        fit_alpha_beta([1024], [1e-5])  # one point cannot fit two parameters
    with pytest.raises(ValueError):
        fit_alpha_beta([1024, 1024, 1024], [1e-5, 1.1e-5, 0.9e-5])  # no size spread


def test_roofline_fraction_and_structural_counts() -> None:
    torch.testing.assert_close(
        torch.tensor(roofline_fraction(2e-6, 1e-6)), torch.tensor(0.5), rtol=1e-12, atol=0.0
    )
    # adjoint doubles the forward traffic (test_traffic pins this on the real op)
    assert COLLECTIVES_PER_APPLY["fwd_adj"] == 2 * COLLECTIVES_PER_APPLY["fwd"]
    assert (
        COLLECTIVES_PER_APPLY["fwd"]
        < COLLECTIVES_PER_APPLY["fwd_adj_ckpt"]
        <= 3 * COLLECTIVES_PER_APPLY["fwd"]
    )


def test_fft_throughput_roofline_hand_value() -> None:
    # N=2^20, P=4, B=100e9 B/s, alpha=16 B, r=2: Psi = 5*20*4*100e9/(16*2) = 1.25e12 FLOP/s
    psi = fft_throughput_roofline(2**20, 4, 100e9, 16, 2)
    torch.testing.assert_close(torch.tensor(psi), torch.tensor(1.25e12), rtol=1e-12, atol=0.0)


def test_fft_throughput_roofline_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        fft_throughput_roofline(2**20, 4, -1.0, 16, 2)
