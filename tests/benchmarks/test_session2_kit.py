"""Session-2 kit: the single-GPU baseline meter and the true-speedup figure.

The rule this kit exists to satisfy: speedup S(P) = T(best single) / T(P) against the best
SINGLE-device implementation -- never the distributed code on one rank (whose collective/pack
overhead inflates the ratio). The single meter runs the core reference (no dist imports on its
path); the plot side computes S(P) per row and writes the speedup figure.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from benchmarks.bench_single import measure_single
from benchmarks.plot_c2 import collect_runs, plot_all, true_speedup
from tests.benchmarks.conftest import make_summary_run as _fake


def test_measure_single_cpu_smoke() -> None:
    out = measure_single(n=(8, 8, 8), dtype=torch.float64, reps=2, warmup=1, device="cpu", ms=8e5)
    for mode in ("fwd", "fwd_adj"):
        r = out[mode]
        assert r["wall_ms_mean"] > 0 and len(r["wall_ms_all"]) == 2


def test_true_speedup_uses_the_single_baseline(tmp_path: Path) -> None:
    _fake(tmp_path, "c2single_baseline", 1, 40.0, single=True)
    _fake(tmp_path, "c2_x_P2_strong_r0", 2, 25.0)
    _fake(tmp_path, "c2_x_P4_strong_r0", 4, 12.5)
    runs = collect_runs(tmp_path)
    s = true_speedup(runs, row="r0")
    assert s == {2: pytest.approx(40.0 / 25.0), 4: pytest.approx(40.0 / 12.5)}


def test_true_speedup_matches_the_single_baseline_grid(tmp_path: Path) -> None:
    """With baselines at several sizes, the denominator is the one on the STRONG grid --
    a smaller grid's (faster) single time must never inflate-or-deflate the ratio."""
    _fake(tmp_path, "c2single_small", 1, 10.0, single=True, n=[4, 4, 4])
    _fake(tmp_path, "c2single_match", 1, 40.0, single=True, n=[8, 8, 8])
    _fake(tmp_path, "c2_x_P2_strong_r0", 2, 25.0)  # strong grid is [8, 8, 8]
    s = true_speedup(collect_runs(tmp_path), row="r0")
    assert s == {2: pytest.approx(40.0 / 25.0)}


def test_plot_all_writes_size_scaling_figure(tmp_path: Path) -> None:
    """Strong runs at more than one grid size produce the classic wall-vs-P-per-size figure."""
    pytest.importorskip("matplotlib")
    for n, p, w in ((8, 2, 40.0), (8, 4, 20.0), (16, 2, 300.0), (16, 4, 150.0)):
        _fake(tmp_path, f"c2_x_P{p}_strong{n}_r0", p, w, n=[n, n, n])
    written = plot_all(tmp_path, tmp_path / "plots")
    assert any("size_scaling" in str(w) for w in written)


def test_plot_all_writes_speedup_figure(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    _fake(tmp_path, "c2single_baseline", 1, 40.0, single=True)
    for p, w in ((2, 25.0), (4, 12.5)):
        _fake(tmp_path, f"c2_x_P{p}_strong_r0", p, w)
    written = plot_all(tmp_path, tmp_path / "plots")
    assert (tmp_path / "plots" / "true_speedup.png").is_file()
    assert any("true_speedup" in str(w) for w in written)
