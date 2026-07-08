"""Unit tests for the memory-wall fit logic (no GPU; synthetic points, explicit rtol/atol)."""

from __future__ import annotations

import pytest
import torch

from benchmarks.mem_wall import MemPoint, fit_bytes_per_cell, largest_grid_within


def _synthetic(bytes_per_cell: float, overhead: float, cubes: list[int]) -> list[MemPoint]:
    # exact linear model bytes = a * n^3 + b, filled into the torch-peak slot
    return [
        MemPoint(
            n=(n, n, n), torch_peak_bytes=bytes_per_cell * n**3 + overhead, smi_delta_bytes=None
        )
        for n in cubes
    ]


def test_fit_recovers_known_slope_and_intercept() -> None:
    pts = _synthetic(bytes_per_cell=512.0, overhead=1e7, cubes=[32, 48, 64, 96])
    fit = fit_bytes_per_cell(pts)
    torch.testing.assert_close(
        torch.tensor(fit.bytes_per_cell), torch.tensor(512.0), rtol=1e-9, atol=1e-6
    )
    torch.testing.assert_close(
        torch.tensor(fit.overhead_bytes), torch.tensor(1e7), rtol=1e-9, atol=1e-3
    )
    # a clean linear model leaves ~zero residual
    assert fit.max_abs_residual_bytes < 1e-3


def test_largest_cubic_grid_within_budget() -> None:
    # a=512 B/cell, b=1e7 B overhead; budget 8 GB -> n^3 <= (8e9 - 1e7)/512 = 15605468.75
    # cube-root ~ 249.6 -> largest integer n that fits is 249 (249^3*512+1e7 = 7.91e9 <= 8e9)
    fit = fit_bytes_per_cell(_synthetic(512.0, 1e7, [32, 48, 64, 96]))
    n = largest_grid_within(fit, vram_bytes=8e9)
    assert n == 249
    assert fit.predict_bytes((n, n, n)) <= 8e9
    assert fit.predict_bytes((n + 1, n + 1, n + 1)) > 8e9


def test_largest_grid_honors_even_step() -> None:
    # step=2 (even grids, as the dist op prefers) -> largest even n <= 249 is 248
    fit = fit_bytes_per_cell(_synthetic(512.0, 1e7, [32, 48, 64, 96]))
    assert largest_grid_within(fit, vram_bytes=8e9, step=2) == 248


def test_predict_uses_smi_delta_when_present() -> None:
    # smi delta (the true wall incl. raw-cudaMalloc'd cuFFT buffers) overrides torch peak in the fit
    pts = [
        MemPoint(n=(n, n, n), torch_peak_bytes=100.0 * n**3, smi_delta_bytes=700.0 * n**3)
        for n in (32, 48, 64)
    ]
    fit = fit_bytes_per_cell(pts)
    torch.testing.assert_close(
        torch.tensor(fit.bytes_per_cell), torch.tensor(700.0), rtol=1e-9, atol=1e-6
    )


def test_fit_rejects_underdetermined_input() -> None:
    with pytest.raises(ValueError):
        fit_bytes_per_cell(_synthetic(512.0, 0.0, [64]))  # one point cannot fit a+b
