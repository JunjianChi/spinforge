"""Figure builder: consume the benchmark summary JSONs, emit the headline figures.

Run after a benchmark run (offline -- analysis never burns rental time):

    PYTHONPATH=src:. python benchmarks/plot_c2.py --runs runs --out runs/plots

Figures: strong-scaling parallel efficiency per optimization row (the headline curve), weak-scaling
wall time per row, wall-vs-P per grid size (when several sizes exist), true speedup vs the
single-device baseline (when ``c2single_*`` runs exist), and -- when perturbation runs exist --
the elimination readout (full step vs the no-comm / no-fft bounds).
Colors are the validated default categorical palette in FIXED slot order
(row identity owns its hue in the row figures; the size figure colors by grid);
one axis per figure; text
in ink, never series color.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("plot_c2")

# validated default categorical palette (dataviz reference instance), fixed slot order
ROW_ORDER = ("r0", "rfft", "mixed", "overlap", "cum_rfft_overlap")
ROW_COLOR = {
    "r0": "#2a78d6",
    "rfft": "#1baf7a",
    "mixed": "#eda100",
    "overlap": "#008300",
    "cum_rfft_overlap": "#4a3aa7",
}
ROW_LABEL = {
    "r0": "R0 (naive)",
    "rfft": "rfft wire",
    "mixed": "f32 wire",
    "overlap": "overlap",
    "cum_rfft_overlap": "rfft + overlap",
}
_INK, _INK2, _GRID = "#3d3d3a", "#6b6b66", "#e5e5e0"


@dataclass(frozen=True)
class Run:
    row: str
    column: str  # strong | weak | bound | r0equiv | single
    world: int
    wall_ms: float
    ci95_ms: float
    perturb: str
    uninstrumented: bool
    n: tuple[int, ...]
    gpu: str


def _row_from_config(cfg: dict) -> str:
    if cfg.get("use_rfft") and cfg.get("schedule") == "pipelined":
        return "cum_rfft_overlap"
    if cfg.get("use_rfft"):
        return "rfft"
    if cfg.get("mixed_wire"):
        return "mixed"
    if cfg.get("schedule") == "pipelined":
        return "overlap"
    return "r0"


def collect_runs(runs_dir: Path, mode: str = "fwd_adj") -> list[Run]:
    """Parse every ``c2*/summary.json`` under ``runs_dir`` (row identity from config flags;
    ``c2single_*`` runs become the ``single`` column -- the true-speedup denominator)."""
    out: list[Run] = []
    for summary in sorted(runs_dir.glob("c2*/summary.json")):
        data = json.loads(summary.read_text())
        cfg, res = data["config"], data["results"]
        if mode not in res:
            continue
        m = re.search(r"_(strong|weak|bound|r0equiv)", summary.parent.name)
        # no recognized tag -> "other" (overwall/energy runs must never pollute the figures)
        column = "single" if cfg.get("single") else (m.group(1) if m else "other")
        out.append(
            Run(
                row="single" if cfg.get("single") else _row_from_config(cfg),
                column=column,
                world=int(cfg["world_size"]),
                wall_ms=float(res[mode]["wall_ms_mean"]),
                ci95_ms=float(res[mode].get("wall_ms_ci95", 0.0)),
                perturb=str(cfg.get("perturb", "none")),
                uninstrumented=bool(cfg.get("uninstrumented", False)),
                n=tuple(cfg.get("n", ())),
                gpu=str(cfg.get("env", {}).get("gpu", "")),
            )
        )
    return out


def modal_grid(runs: list[Run]) -> tuple[int, ...] | None:
    """The grid most strong runs share (ties -> more cells) -- ONE grid for every headline figure,
    so a size-sweep run dir cannot splinter rows onto different grids."""
    strong = [r for r in runs if r.column == "strong" and r.perturb == "none" and r.n]
    if not strong:
        return None
    counts: dict[tuple[int, ...], int] = {}
    for r in strong:
        counts[r.n] = counts.get(r.n, 0) + 1
    return max(counts, key=lambda n: (counts[n], n[0] * n[1] * n[2]))


def row_series(runs: list[Run], row: str, grid: tuple[int, ...] | None = None) -> list[Run]:
    """One row's strong-scaling series on ``grid`` (default: the modal grid): a single meter
    throughout (never splice instrumented and uninstrumented walls into one curve), the variant
    with the most P coverage winning (ties -> uninstrumented)."""
    cands = [r for r in runs if r.column == "strong" and r.row == row and r.perturb == "none"]
    if not cands:
        return []
    grid = grid or modal_grid(runs)
    return _one_meter(r for r in cands if r.n == grid)


def _one_meter(cands) -> list[Run]:  # noqa: ANN001
    """Pick ONE meter's points (most P coverage; ties -> uninstrumented), one point per P."""
    groups: dict[bool, dict[int, Run]] = {}
    for r in cands:
        groups.setdefault(r.uninstrumented, {})[r.world] = r
    if not groups:
        return []
    best = max(groups.values(), key=lambda g: (len(g), list(g.values())[0].uninstrumented))
    return sorted(best.values(), key=lambda r: r.world)


def weak_series(runs: list[Run], row: str) -> list[Run]:
    """One row's weak-scaling series: same one-meter/dedup rule (repeated sessions collide)."""
    cands = (r for r in runs if r.column == "weak" and r.row == row and r.perturb == "none")
    return _one_meter(cands)


def true_speedup(runs: list[Run], row: str) -> dict[int, float]:
    """S(P) = T(single-GPU reference) / T(P) for one row's strong-scaling runs. The denominator
    is our full-complex single-device reference, never the distributed code on one rank."""
    strong = row_series(runs, row)
    if not strong:
        return {}
    grid = strong[0].n
    # the P=1 distribution-cost marker: not part of the trend's meter, annotated in the reports,
    # so it may come from whichever meter measured it (there is no instrumented P=1 pair)
    if all(r.world != 1 for r in strong):
        p1 = [
            r
            for r in runs
            if r.column == "strong" and r.row == row and r.world == 1 and r.n == grid
        ]
        strong = sorted(strong + p1[:1], key=lambda r: r.world)
    # the denominator must sit on the SAME grid as the strong runs; baselines at other sizes
    # exist for context and must never leak into the ratio
    single = [r for r in runs if r.column == "single" and r.n == grid]
    if not single:
        return {}
    t1 = min(r.wall_ms for r in single)
    return {r.world: t1 / r.wall_ms for r in strong}


def strong_efficiency(runs: list[Run]) -> dict[int, float]:
    """eff(P) = T(Pmin)*Pmin / (T(P)*P) over one row's strong-scaling runs."""
    by_p = {r.world: r.wall_ms for r in runs}
    p0 = min(by_p)
    return {p: (by_p[p0] * p0) / (t * p) for p, t in sorted(by_p.items())}


def _style(ax) -> None:  # noqa: ANN001
    ax.grid(True, color=_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK2)
    ax.tick_params(colors=_INK2, labelcolor=_INK)


def plot_all(runs_dir: Path, out_dir: Path, mode: str = "fwd_adj") -> list[Path]:
    """Build every figure the collected runs support; returns the written paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = collect_runs(runs_dir, mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    # every figure names its machine (auditor rule: the reader must never have to guess which
    # hardware a curve came from); mixed-machine runs dirs would show as a joined label
    gpus = sorted({r.gpu for r in runs if r.gpu})
    hw = " + ".join(gpus) if gpus else "CPU (smoke)"

    def rows_in(column: str) -> list[str]:
        present = {r.row for r in runs if r.column == column and r.perturb == "none"}
        return [r for r in ROW_ORDER if r in present]

    # 1) the headline: strong-scaling parallel efficiency per row
    strong_rows = rows_in("strong")
    if strong_rows:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for row in strong_rows:
            eff = strong_efficiency(row_series(runs, row))
            ps = list(eff)
            ax.plot(
                ps,
                [100 * eff[p] for p in ps],
                color=ROW_COLOR[row],
                linewidth=2,
                marker="o",
                markersize=7,
                label=ROW_LABEL[row],
                zorder=3,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted({r.world for r in runs if r.column == "strong"}))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("GPUs (P)", color=_INK)
        ax.set_ylabel(f"parallel efficiency (%), {mode}", color=_INK)
        ax.set_title(f"Strong scaling efficiency \u2014 {hw}", color=_INK)
        ax.legend(frameon=False, labelcolor=_INK)
        _style(ax)
        p = out_dir / "strong_efficiency.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    # 2) weak scaling: wall time vs P per row (flat = perfect)
    weak_rows = rows_in("weak")
    if weak_rows:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for row in weak_rows:
            rs = weak_series(runs, row)
            ax.errorbar(
                [r.world for r in rs],
                [r.wall_ms for r in rs],
                yerr=[r.ci95_ms for r in rs],
                color=ROW_COLOR[row],
                linewidth=2,
                marker="o",
                markersize=7,
                capsize=3,
                label=ROW_LABEL[row],
                zorder=3,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted({r.world for r in runs if r.column == "weak"}))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_ylim(bottom=0)
        ax.set_xlabel("GPUs (P), per-rank slab fixed", color=_INK)
        ax.set_ylabel(f"wall time per {mode} step (ms, ±95% CI)", color=_INK)
        ax.set_title(f"Weak scaling (flat is perfect) \u2014 {hw}", color=_INK)
        ax.legend(frameon=False, labelcolor=_INK)
        _style(ax)
        p = out_dir / "weak_scaling.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    # 3) the classic size-x-P figure: wall vs P, one line per grid size (strong column)
    by_size: dict[tuple[int, ...], list[Run]] = {}
    for r in runs:
        if r.column == "strong" and r.perturb == "none" and r.n:
            by_size.setdefault(r.n, []).append(r)
    if len(by_size) > 1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sizes = sorted(by_size, key=lambda n: n[0] * n[1] * n[2])
        for i, n in enumerate(sizes):
            # prefer the optimized row where present, else r0 -- one line per size, stated
            rs = row_series(runs, "cum_rfft_overlap", grid=n) or row_series(runs, "r0", grid=n)
            color = list(ROW_COLOR.values())[i % len(ROW_COLOR)]
            label = "\u00d7".join(str(v) for v in n)
            ax.plot(
                [r.world for r in rs],
                [r.wall_ms for r in rs],
                color=color,
                linewidth=2,
                marker="o",
                markersize=7,
                label=label,
                zorder=3,
            )
            ideal = [rs[0].wall_ms * rs[0].world / r.world for r in rs]
            ax.plot(
                [r.world for r in rs],
                ideal,
                color=color,
                linewidth=1,
                linestyle="--",
                alpha=0.45,
                zorder=2,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(sorted({r.world for rs in by_size.values() for r in rs}))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("GPUs (P)", color=_INK)
        ax.set_ylabel(f"wall time per {mode} step (ms)", color=_INK)
        ax.set_title(f"Strong scaling by problem size \u2014 {hw}", color=_INK)
        ax.legend(frameon=False, labelcolor=_INK, title="grid", title_fontsize=9)
        _style(ax)
        p = out_dir / "size_scaling.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    # 4) TRUE speedup vs the single-device baseline, with the ideal line
    if any(r.column == "single" for r in runs) and strong_rows:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ps_all = sorted({r.world for r in runs if r.column == "strong"})
        ax.plot(ps_all, ps_all, color=_GRID, linewidth=1.5, linestyle="--", zorder=2)
        ax.text(ps_all[-1], ps_all[-1], " ideal", color=_INK2, fontsize=9, va="center")
        for row in strong_rows:
            s_of_p = true_speedup(runs, row)
            if not s_of_p:
                continue
            ps = list(s_of_p)
            ax.plot(
                ps,
                [s_of_p[p] for p in ps],
                color=ROW_COLOR[row],
                linewidth=2,
                marker="o",
                markersize=7,
                label=ROW_LABEL[row],
                zorder=3,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(ps_all)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_ylim(bottom=0)
        ax.set_xlabel("GPUs (P)", color=_INK)
        ax.set_ylabel(f"speedup vs the single-GPU reference, {mode}", color=_INK)
        ax.set_title(f"True speedup vs one GPU \u2014 {hw}", color=_INK)
        ax.legend(frameon=False, labelcolor=_INK)
        _style(ax)
        p = out_dir / "true_speedup.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    # 5) elimination readout: the full step against its high-watermark bounds
    bounds = {r.perturb: r for r in runs if r.column == "bound"}
    # the full-step bar must share the bounds' (uninstrumented) meter: prefer the clean
    # r0-equivalence run; the instrumented strong_r0 carries ~10% observer overhead
    clean = [r for r in runs if r.column == "r0equiv" and r.uninstrumented]
    full = clean or [r for r in runs if r.column == "strong" and r.row == "r0"]
    if bounds and full:
        ref = max(full, key=lambda r: r.world)
        fig, ax = plt.subplots(figsize=(6, 4))
        label = "full step (R0, clean)" if clean else "full step (R0, instrumented)"
        names = [label] + [f"bound: {k}" for k in bounds]
        vals = [ref.wall_ms] + [b.wall_ms for b in bounds.values()]
        colors = [ROW_COLOR["r0"]] + [_GRID] * len(bounds)
        bars = ax.bar(names, vals, color=colors, edgecolor=_INK2, linewidth=0.8, zorder=3)
        for b, v in zip(bars, vals, strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                color=_INK,
                fontsize=9,
            )
        ax.set_ylabel(f"wall time per {mode} step (ms)", color=_INK)
        ax.set_title(f"Comm dominance by elimination (P={ref.world})", color=_INK)
        _style(ax)
        p = out_dir / "elimination_bounds.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    logger.info("wrote %d figures to %s", len(written), out_dir)
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("runs/plots"))
    ap.add_argument("--mode", default="fwd_adj")
    args = ap.parse_args()
    for p in plot_all(args.runs, args.out, args.mode):
        logger.info("  %s", p)


if __name__ == "__main__":
    main()
