"""The figure builder: summary JSONs in, headline figures out (pure logic unit-tested).

The plotting itself is matplotlib (viz extra); what must be RIGHT is the arithmetic feeding it:
run-name parsing, row identification from config flags, and the parallel-efficiency computation
(eff(P) = T(Pmin)*Pmin / (T(P)*P) on the fwd_adj wall mean). An end-to-end run over a synthetic
runs/ tree must produce the figure files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.plot_c2 import collect_runs, plot_all, strong_efficiency, true_speedup
from tests.benchmarks.conftest import make_summary_run as _fake_run


def test_collect_and_strong_efficiency(tmp_path: Path) -> None:
    _fake_run(tmp_path, "c2_x_P2_strong_r0", 2, 10.0)
    _fake_run(tmp_path, "c2_x_P4_strong_r0", 4, 6.0)
    _fake_run(tmp_path, "c2_x_P4_strong_rfft", 4, 4.0, use_rfft=True)
    _fake_run(tmp_path, "c2_x_P4_weak_r0", 4, 12.0)

    runs = collect_runs(tmp_path)
    rows = {(r.column, r.row, r.world) for r in runs}
    assert ("strong", "r0", 2) in rows and ("strong", "rfft", 4) in rows
    assert ("weak", "r0", 4) in rows

    eff = strong_efficiency([r for r in runs if r.column == "strong" and r.row == "r0"])
    # eff(P) = T(2)*2 / (T(P)*P): P=2 -> 1.0; P=4 -> 20/24
    assert eff[2] == pytest.approx(1.0)
    assert eff[4] == pytest.approx(20.0 / 24.0)


def test_plot_all_writes_figures(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    for p, w in ((2, 10.0), (4, 6.0)):
        _fake_run(tmp_path, f"c2_x_P{p}_strong_r0", p, w)
        _fake_run(tmp_path, f"c2_x_P{p}_strong_rfft", p, 0.6 * w, use_rfft=True)
        _fake_run(tmp_path, f"c2_x_P{p}_weak_r0", p, 11.0)
        _fake_run(tmp_path, f"c2_x_P{p}_weak_rfft", p, 7.0, use_rfft=True)
    out = tmp_path / "plots"
    written = plot_all(tmp_path, out)
    assert (out / "strong_efficiency.png").is_file()
    assert (out / "weak_scaling.png").is_file()
    assert len(written) >= 2


def test_unmatched_run_names_stay_out_of_the_strong_column(tmp_path: Path) -> None:
    """overwall/energy runs carry no column tag and must not pollute the headline figures."""
    _fake_run(tmp_path, "c2_x_P2_strong_r0", 2, 10.0)
    _fake_run(tmp_path, "c2_x_P8_overwall", 8, 251.0)
    _fake_run(tmp_path, "c2_x_P8_energy_r0", 8, 23.8)
    runs = collect_runs(tmp_path)
    assert {r.column for r in runs} == {"strong", "other"}
    assert strong_efficiency([r for r in runs if r.column == "strong" and r.row == "r0"]) == {
        2: 1.0
    }


def test_row_series_prefers_the_meter_with_full_p_coverage(tmp_path: Path) -> None:
    """A clean rerun at one P must not splice into an instrumented row measured at every P --
    one meter per plotted series, coverage first."""
    from benchmarks.plot_c2 import row_series

    for p, w in ((2, 100.0), (4, 49.0), (8, 28.0)):
        _fake_run(tmp_path, f"c2_x_P{p}_strong_r0", p, w)
    _fake_run(tmp_path, "c2_y_P8_strong_r0_clean", 8, 23.7, uninstrumented=True)
    runs = [r for r in collect_runs(tmp_path) if r.column == "strong"]
    series = row_series(runs, "r0")
    assert sorted(r.world for r in series) == [2, 4, 8]
    assert all(not r.uninstrumented for r in series)  # the 3-point instrumented set wins


def test_weak_series_dedups_repeated_sessions(tmp_path: Path) -> None:
    """Two sessions' weak runs at the same P must yield one point per P, one meter throughout."""
    from benchmarks.plot_c2 import weak_series

    for pp, w in ((2, 13.0), (4, 13.5), (8, 14.0)):
        _fake_run(tmp_path, f"c2_a_P{pp}_weak_r0", pp, w)
    _fake_run(tmp_path, "c2_b_P8_weak_r0", 8, 13.2, uninstrumented=True)  # a rerun stamp
    series = weak_series(collect_runs(tmp_path), "r0")
    assert [r.world for r in series] == [2, 4, 8]
    assert all(not r.uninstrumented for r in series)


def test_size_scaling_uses_the_row_series_rules(tmp_path: Path) -> None:
    """The per-size figure must not splice meters either (one clean P=8 rerun must not join an
    instrumented 3-point line)."""
    from benchmarks.plot_c2 import row_series

    for pp, w in ((2, 40.0), (4, 20.0), (8, 10.0)):
        _fake_run(tmp_path, f"c2_x_P{pp}_strong_r0", pp, w, n=[8, 8, 8])
    _fake_run(tmp_path, "c2_y_P8_strong_r0_clean", 8, 9.0, n=[8, 8, 8], uninstrumented=True)
    series = row_series([r for r in collect_runs(tmp_path)], "r0", grid=(8, 8, 8))
    assert [r.wall_ms for r in series] == [40.0, 20.0, 10.0]


def test_headline_rows_share_the_modal_grid(tmp_path: Path) -> None:
    """A size-sweep dir must not put one row on 128-cubed and another on 256-cubed."""
    from benchmarks.plot_c2 import row_series

    for pp, w in ((2, 100.0), (4, 49.0), (8, 28.0)):
        _fake_run(tmp_path, f"c2_x_P{pp}_strong_r0", pp, w, n=[16, 16, 16])
        _fake_run(tmp_path, f"c2_x_P{pp}_strong_rfft", pp, w / 2, n=[16, 16, 16], use_rfft=True)
    for pp, w in ((2, 6.0), (8, 8.0)):  # a smaller size-sweep grid with fewer runs
        _fake_run(tmp_path, f"c2_s_P{pp}_strong8_rfft", pp, w, n=[8, 8, 8], use_rfft=True)
    runs = collect_runs(tmp_path)
    assert {r.n for r in row_series(runs, "rfft")} == {(16, 16, 16)}


def test_true_speedup_keeps_the_p1_cost_marker(tmp_path: Path) -> None:
    _fake_run(tmp_path, "c2single_base", 1, 40.0, single=True)
    _fake_run(tmp_path, "c2_x_P1_strong_r0_p1", 1, 55.0, uninstrumented=True)
    for pp, w in ((2, 30.0), (4, 15.0)):
        _fake_run(tmp_path, f"c2_x_P{pp}_strong_r0", pp, w)
    s = true_speedup(collect_runs(tmp_path), row="r0")
    assert s[1] == pytest.approx(40.0 / 55.0)  # the 'cost of going distributed' point survives
    assert s[2] == pytest.approx(40.0 / 30.0)
