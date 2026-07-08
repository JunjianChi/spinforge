"""Distributed demag timing harness: gated, synchronized, repeated, with CIs.

The measurement discipline the neighbors lack (first-hand study 2026-07: magnum.np.distributed's
Timer has no cuda.synchronize/barrier and its distributed<->single "match" states no tolerances):

1. **Correctness gate BEFORE timing**: the distributed demag first matches the
   single-process reference on this rank's slab within explicit rtol/atol, running through the
   SAME instrumented collective that is then timed. No gate pass -> no numbers. The gate runs on
   its own SMALL gate grid (--gate-nx/ny/nz), decoupled from the timing grid: the timing grid may
   exceed one device in aggregate, where a single-process reference cannot exist by definition.
2. **Synchronized timing**: device drained (torch.cuda.synchronize), then a host barrier on a gloo
   CONTROL-PLANE subgroup before every timestamp; per-rep time = MAX over ranks (the step is as
   slow as its slowest rank), reduced over the same gloo control plane -- timing scalars are host
   floats and must never touch the compute backend (NCCL cannot reduce CPU tensors).
3. **Warmup + repetitions + 95% CI** (t-distribution), reported alongside mean/std.
4. **Comm breakdown + measured collective counts**: pack / forward-comm / backward-comm timed
   inside a differentiable instrumented all_to_all (sync-based -- valid for the NAIVE schedule
   only; overlapped variants must be profiled with nsys, wall time stays the headline metric).
5. **Roofline**: with --link-bw-gbs (a MEASURED per-rank bandwidth) and optionally --alpha-us (a
   MEASURED per-message latency, from benchmarks/bench_alpha_beta.py), reports the alpha-beta comm
   lower bound and the achieved fraction (the tie-vs-production-FFT instrument).
6. **Slab-only timing inputs**: each rank materializes only its own timing slab (content is
   irrelevant to timing; the gate grid carries the correctness burden), so the harness itself
   never re-introduces the full-grid allocation the project exists to avoid.

Run from the repo root (harness validation, Mac/CPU/gloo -- timings are NOT performance claims on
CPU); ``src`` supplies the package and ``.`` makes ``benchmarks`` importable:
    PYTHONPATH=src:. torchrun --nproc_per_node=2 benchmarks/bench_dist_demag.py \
        --nx 8 --ny 8 --nz 8 --dtype float64 --reps 10 --warmup 3
Run (AutoDL, NCCL): same, plus --link-bw-gbs <measured> and a real grid; add --run-name for
provenance capture under runs/ (seed + config + git commit).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from benchmarks.roofline import comm_time_lower_bound, roofline_fraction, wire_bytes_per_rank
from benchmarks.stats import timing_stats
from spinforge.core.demag import DemagField
from spinforge.core.mesh import Mesh
from spinforge.dist.checkpoint import checkpointed_demag
from spinforge.dist.collectives import (
    SyncCollectiveA2A,
    a2a_calls,
    all_reduce_sum,
    all_to_all_complex_mixed,
    reset_a2a_calls,
)
from spinforge.dist.demag import DistributedDemagField
from spinforge.io import set_seed

MODES = ("fwd", "fwd_adj", "fwd_adj_ckpt")
# High-watermark perturbations (comm-dominance by ELIMINATION, not assertion): each variant's
# wall time is a measured upper bound on one optimization class. Perturbed physics is wrong by
# construction -> the reference-match gate is skipped and the summary labels the run a timing bound.
PERTURBS = ("none", "no-comm", "no-fft")
# Gate tolerances by real dtype. f64 matches the committed gate (tests/dist/test_demag.py). f32:
# |H_demag| ~ Ms/3 ~ 2.7e5 A/m here, so atol=1.0 A/m is ~4e-6 relative -- revisit on GPU data.
GATE_TOL = {torch.float64: (1e-9, 1e-6), torch.float32: (1e-4, 1.0)}


def _environment() -> dict[str, object]:
    """Machine context every performance number must carry (the pyperf/Google-Benchmark habit):
    re-analysis months later must not depend on remembering which node produced the file."""
    env: dict[str, object] = {"hostname": platform.node(), "python": platform.python_version()}
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name()
        env["gpu_count"] = torch.cuda.device_count()
        env["capability"] = ".".join(map(str, torch.cuda.get_device_capability()))
        env["cuda_runtime"] = torch.version.cuda
        env["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
        try:
            env["driver"] = (
                subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                .stdout.split("\n")[0]
                .strip()
            )
        except (OSError, subprocess.TimeoutExpired):
            env["driver"] = "unknown"
    return env


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class CommStats:
    """Sync-based per-phase accumulators for the instrumented collective (naive schedule only)."""

    pack_s: float = 0.0
    comm_fwd_s: float = 0.0
    comm_bwd_s: float = 0.0
    n_fwd: int = 0
    n_bwd: int = 0

    def reset(self) -> None:
        self.pack_s = self.comm_fwd_s = self.comm_bwd_s = 0.0
        self.n_fwd = self.n_bwd = 0

    def calls(self) -> int:
        return self.n_fwd + self.n_bwd


class _TimedAllToAll(torch.autograd.Function):
    """Differentiable all_to_all with sync-based phase timing (mirrors dist/collectives.py).

    Duplicated here (not shipped) so the BACKWARD collective is timed too -- the shipped op's
    adjoint runs inside autograd where an outer wrapper cannot see it. Fidelity to the shipped
    op is enforced by running the reference-match gate through this instrument before any timing.
    """

    @staticmethod
    def forward(ctx: object, x: torch.Tensor, cs: CommStats) -> torch.Tensor:  # type: ignore[override]
        ctx.cs = cs  # type: ignore[attr-defined]
        _sync()
        t0 = time.perf_counter()
        xc = x.contiguous()
        _sync()
        t1 = time.perf_counter()
        out = torch.empty_like(xc)
        dist.all_to_all_single(out, xc)
        _sync()
        t2 = time.perf_counter()
        cs.pack_s += t1 - t0
        cs.comm_fwd_s += t2 - t1
        cs.n_fwd += 1
        return out

    @staticmethod
    def backward(ctx: object, grad: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        cs: CommStats = ctx.cs  # type: ignore[attr-defined]
        _sync()
        t0 = time.perf_counter()
        gc = grad.contiguous()
        _sync()
        t1 = time.perf_counter()
        grad_in = torch.empty_like(gc)
        dist.all_to_all_single(grad_in, gc)
        _sync()
        t2 = time.perf_counter()
        cs.pack_s += t1 - t0
        cs.comm_bwd_s += t2 - t1
        cs.n_bwd += 1
        return grad_in, None


def timed_collective(cs: CommStats):  # noqa: ANN201 - returns the injectable Collective
    def collective(x: torch.Tensor) -> torch.Tensor:
        real = torch.view_as_real(x).contiguous()  # gloo has no complex; byte-identical view
        return torch.view_as_complex(_TimedAllToAll.apply(real, cs))

    return collective


@dataclass
class Config:
    n: tuple[int, int, int]
    dtype: torch.dtype
    reps: int
    warmup: int
    seed: int
    link_bw: float | None  # bytes/s, measured, or None
    alpha: float = 0.0  # s/message, measured (bench_alpha_beta), 0 = bytes-only bound
    use_rfft: bool = False  # packed-Hermitian transpose plane (the rfft row) -- gate still applies
    schedule: str = "sequential"  # "pipelined" = the overlap row: wall-time only, no phase split
    uninstrumented: bool = False  # sequential on the DEFAULT backend: the naive-schedule vehicle
    mixed_wire: bool = False  # f32 wire payload (the mixed-precision row); gated at f32 tol
    perturb: str = "none"  # high-watermark variant (PERTURBS)
    gate_n: tuple[int, int, int] = (16, 16, 16)  # small full-reference gate grid
    modes: tuple[str, ...] = MODES
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    @property
    def gate_required(self) -> bool:
        """Perturbed runs are timing bounds with deliberately wrong physics -- gating them would
        fail spuriously (no-fft) or validate nothing (no-comm). Everything else is gated."""
        return self.perturb == "none"


def build_field(
    cfg: Config, cs: CommStats, mesh: Mesh, world: int, rank: int
) -> tuple[DistributedDemagField, bool]:
    """The measured op for this configuration; returns ``(field, comm_split_valid)``.

    - default sequential: the blocking instrumented collective (phase split valid -- the naive
    schedule's meter)
    - pipelined / --uninstrumented: the split-phase default backend (wall time only)
    - --perturb no-comm: identity 'collective' (full compute, zero wire)
    - --perturb no-fft: default backend (real wire; the FFT stand-ins are applied by the caller
      via ``no_fft_stand_ins()``)
    """
    if cfg.perturb not in PERTURBS:
        raise ValueError(f"unknown perturb {cfg.perturb!r}; expected one of {PERTURBS}")
    if cfg.perturb == "no-fft" and cfg.use_rfft:
        raise ValueError("--perturb no-fft supports the full-complex path only")
    kwargs: dict[str, object] = {"use_rfft": cfg.use_rfft, "schedule": cfg.schedule}
    if cfg.perturb == "no-comm":
        kwargs["backend"] = SyncCollectiveA2A(lambda x: x)  # differentiable identity, no wire
        split_valid = False
    elif cfg.mixed_wire:
        # blocking mixed collective (async composition is a pinned follow-up); wall-time row
        kwargs["backend"] = SyncCollectiveA2A(all_to_all_complex_mixed)
        split_valid = False
    elif cfg.perturb == "no-fft" or cfg.uninstrumented or cfg.schedule != "sequential":
        split_valid = False  # default split-phase backend; nothing is instrumented
    else:
        kwargs["backend"] = SyncCollectiveA2A(timed_collective(cs))
        split_valid = True
    return DistributedDemagField(mesh, world, rank, **kwargs), split_valid  # type: ignore[arg-type]


@contextmanager
def no_fft_stand_ins() -> Iterator[None]:
    """Replace the full-complex FFTs with shape-faithful, graph-preserving stand-ins.

    Wire traffic and autograd connectivity are UNCHANGED (zero-padding by concatenation is
    differentiable); only the transform arithmetic disappears -- the resulting wall time is the
    measured upper bound on what any local-FFT optimization could ever buy (the
    high-watermark method)."""

    def fake_fft2(x: torch.Tensor, s=None, dim=(-2, -1)):  # noqa: ANN001, ANN202
        out = torch.complex(x, torch.zeros_like(x)) if not x.is_complex() else x
        if s is None:  # already-padded operand (the kernel build's call shape)
            return out
        for d, target in zip(dim, s, strict=True):
            pad_shape = list(out.shape)
            pad_shape[d] = target - out.shape[d]
            if pad_shape[d]:
                pad = torch.zeros(pad_shape, dtype=out.dtype, device=out.device)
                out = torch.cat([out, pad], dim=d)
        return out

    def fake_fft(x: torch.Tensor, n=None, dim=-1):  # noqa: ANN001, ANN202
        if n is not None and n != x.shape[dim]:
            pad_shape = list(x.shape)
            pad_shape[dim] = n - x.shape[dim]
            pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=dim)
        return x

    def fake_inverse(x: torch.Tensor, n=None, dim=-1):  # noqa: ANN001, ANN202
        return x

    def fake_ifft2(x: torch.Tensor, dim=(-2, -1)):  # noqa: ANN001, ANN202
        return x

    saved = (torch.fft.fft2, torch.fft.fft, torch.fft.ifft, torch.fft.ifft2)
    torch.fft.fft2, torch.fft.fft = fake_fft2, fake_fft  # type: ignore[assignment]
    torch.fft.ifft, torch.fft.ifft2 = fake_inverse, fake_ifft2  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.fft.fft2, torch.fft.fft, torch.fft.ifft, torch.fft.ifft2 = saved  # type: ignore[assignment]


def make_ctrl_group() -> dist.ProcessGroup:
    """Gloo control-plane subgroup for host-side timing scalars and barriers.

    The compute backend may be NCCL, which cannot carry CPU tensors -- reducing a host timing
    float through it crashes on the real GPU node (while staying green on gloo-only CI).
    """
    group = dist.new_group(backend="gloo")
    assert group is not None  # every rank is a member
    return group


def reduce_max_seconds(dt: float, ctrl: dist.ProcessGroup) -> float:
    """MAX of a per-rank wall time over the gloo control plane (host tensors only)."""
    t = torch.tensor([dt], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=ctrl)
    return float(t.item())


def make_gate_inputs(cfg: Config, rank: int, world: int) -> tuple[Mesh, torch.Tensor, torch.Tensor]:
    """Gate grid only: identical seeded global m on every rank -> (mesh, m_full, this rank's slab).

    The full global m exists ONLY here, at the small gate grid where the single-process reference
    is computable; the timing grid never materializes it (make_timing_inputs).
    """
    set_seed(cfg.seed)
    nx, ny, nz = cfg.gate_n
    mesh = Mesh(n=cfg.gate_n, dx=(2e-9, 2e-9, 2e-9))
    m = torch.randn(nx, ny, nz, 3, dtype=cfg.dtype, device=cfg.device)
    m = m / m.norm(dim=-1, keepdim=True)
    nzl = nz // world
    return mesh, m, m[:, :, rank * nzl : (rank + 1) * nzl].contiguous()


def make_timing_inputs(cfg: Config, rank: int, world: int) -> tuple[Mesh, torch.Tensor]:
    """Timing grid: only this rank's [nx, ny, nz/world, 3] slab, seeded per (seed, rank).

    Timing needs a unit-normalized m of the right shape, not any particular field state; the
    correctness burden lives on the gate grid, so no rank ever holds the full timing-grid m.
    """
    set_seed(cfg.seed + rank)
    nx, ny, nz = cfg.n
    mesh = Mesh(n=cfg.n, dx=(2e-9, 2e-9, 2e-9))
    m = torch.randn(nx, ny, nz // world, 3, dtype=cfg.dtype, device=cfg.device)
    return mesh, m / m.norm(dim=-1, keepdim=True)


def correctness_gate(
    field_op: DistributedDemagField,
    mesh: Mesh,
    m_full: torch.Tensor,
    m_local: torch.Tensor,
    ms: float,
    rank: int,
    world: int,
    mixed_tol: bool = False,
) -> tuple[float, float]:
    """Distributed forward matches the single-process reference slab within tolerance. Raises on
    failure."""
    rtol, atol = GATE_TOL[torch.float32 if mixed_tol else m_full.dtype]
    ref = DemagField(mesh)(m_full, ms)
    nzl = mesh.n[2] // world
    with torch.no_grad():
        h = field_op(m_local, ms)
    torch.testing.assert_close(h, ref[:, :, rank * nzl : (rank + 1) * nzl], rtol=rtol, atol=atol)
    return rtol, atol


def run_mode(
    mode: str,
    field_op: DistributedDemagField,
    m_local: torch.Tensor,
    ms: float,
    cfg: Config,
    cs: CommStats,
    ctrl: dist.ProcessGroup,
) -> dict[str, float | int | list[float]]:
    """Time one mode: warmup reps discarded, per-rep wall = MAX over ranks, comm phases split."""

    def step() -> None:
        if mode == "fwd":
            with torch.no_grad():
                field_op(m_local, ms)
            return
        m = m_local.detach().clone().requires_grad_(True)
        if mode == "fwd_adj":
            loss = all_reduce_sum(field_op(m, ms).sum())
        else:  # fwd_adj_ckpt
            loss = all_reduce_sum(checkpointed_demag(field_op, m, ms).sum())
        loss.backward()

    wall: list[float] = []
    for i in range(cfg.warmup + cfg.reps):
        if i == cfg.warmup:
            cs.reset()  # phase accumulators cover only the measured reps
            reset_a2a_calls()  # split-phase counter (used when the collective is uninstrumented)
        _sync()  # drain this rank's device work BEFORE aligning hosts, so t0 starts a clean rep
        dist.barrier(group=ctrl)
        t0 = time.perf_counter()
        step()
        _sync()
        dt = time.perf_counter() - t0
        if i >= cfg.warmup:
            # slowest rank defines the step; reduced on the ctrl plane AFTER the timed window
            wall.append(reduce_max_seconds(dt, ctrl))
    n = len(wall)
    out: dict[str, float | int | list[float]] = timing_stats(wall)  # samples are max-over-ranks
    out.update(
        {
            # this-rank phase splits (per rep); cross-rank comm skew shows in wall vs comm gap.
            # Zero when the run is uninstrumented (pipelined schedule: wall time is the metric).
            "pack_ms": 1e3 * cs.pack_s / n,
            "comm_fwd_ms": 1e3 * cs.comm_fwd_s / n,
            "comm_bwd_ms": 1e3 * cs.comm_bwd_s / n,
            # instrumented runs count in cs; the split-phase default backend counts in a2a_calls
            "n_collectives": (cs.calls() or a2a_calls()) // n,
        }
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, required=True)
    p.add_argument("--ny", type=int, required=True)
    p.add_argument("--nz", type=int, required=True)
    p.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ms", type=float, default=8e5)
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument(
        "--link-bw-gbs",
        type=float,
        default=None,
        help="MEASURED per-rank ejection bandwidth, GB/s (enables roofline fraction)",
    )
    p.add_argument(
        "--alpha-us",
        type=float,
        default=0.0,
        help="MEASURED per-message latency, microseconds (bench_alpha_beta); 0 = bytes-only bound",
    )
    p.add_argument(
        "--gate-nx", type=int, default=16, help="reference-match gate grid (full-reference)"
    )
    p.add_argument("--gate-ny", type=int, default=16)
    p.add_argument("--gate-nz", type=int, default=16)
    p.add_argument(
        "--use-rfft",
        action="store_true",
        help="packed-Hermitian transpose plane (rfft wire, ~2x fewer wire bytes); gated as usual",
    )
    p.add_argument(
        "--schedule",
        choices=("sequential", "pipelined"),
        default="sequential",
        help="sequential = naive schedule (instrumented comm split); pipelined = the overlap "
        "row, timed by "
        "wall clock only (sync-based phase timing is invalid for an overlapped schedule)",
    )
    p.add_argument(
        "--uninstrumented",
        action="store_true",
        help="sequential on the DEFAULT split-phase backend (async start;wait back-to-back): the "
        "pre-registered naive-baseline equivalence check runs this against the default "
        "instrumented run",
    )
    p.add_argument(
        "--mixed-wire",
        action="store_true",
        help="f32 wire payload on the transpose (the mixed-precision row); the reference-match "
        "gate "
        "runs at the f32-class tolerance the wire rounding implies",
    )
    p.add_argument(
        "--perturb",
        choices=PERTURBS,
        default="none",
        help="high-watermark timing bound: no-comm = identity collective (compute share), "
        "no-fft = graph-preserving FFT stand-ins (comm share); gate skipped, physics disabled",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="capture provenance + summary under <repo>/runs/<name> (rank 0)",
    )
    p.add_argument("--repo-root", default=".")
    args = p.parse_args()
    if args.mixed_wire and args.dtype == "float32":
        p.error(
            "--mixed-wire at float32 compute is an identity cast (zero wire-byte reduction); "
            "run the row with --dtype float64"
        )

    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
    cfg = Config(
        n=(args.nx, args.ny, args.nz),
        dtype=getattr(torch, args.dtype),
        reps=args.reps,
        warmup=args.warmup,
        seed=args.seed,
        link_bw=args.link_bw_gbs * 1e9 if args.link_bw_gbs else None,
        alpha=args.alpha_us * 1e-6,
        use_rfft=args.use_rfft,
        schedule=args.schedule,
        uninstrumented=args.uninstrumented,
        mixed_wire=args.mixed_wire,
        perturb=args.perturb,
        gate_n=(args.gate_nx, args.gate_ny, args.gate_nz),
        modes=tuple(args.modes.split(",")),
    )
    try:
        ctrl = make_ctrl_group()  # collective: every rank must create it
        cs = CommStats()
        # reference-match gate on the small full-reference grid, through the same backend,
        # schedule, and
        # transpose packing as the timed op (an rfft/pipelined run gates that exact path);
        # perturbed runs are timing bounds with disabled physics -> gate skipped, summary says so
        gate_info: dict[str, object]
        if cfg.gate_required:
            gate_mesh, gm_full, gm_local = make_gate_inputs(cfg, rank, world)
            gate_field, _ = build_field(cfg, cs, gate_mesh, world, rank)
            rtol, atol = correctness_gate(
                gate_field,
                gate_mesh,
                gm_full,
                gm_local,
                args.ms,
                rank,
                world,
                mixed_tol=cfg.mixed_wire,
            )
            gate_info = {"n": cfg.gate_n, "rtol": rtol, "atol": atol, "passed": True}
        else:
            gate_info = {"skipped": f"perturb={cfg.perturb}: timing bound, physics disabled"}

        mesh, m_local = make_timing_inputs(cfg, rank, world)
        field_op, comm_split_valid = build_field(cfg, cs, mesh, world, rank)
        perturb_ctx = no_fft_stand_ins if cfg.perturb == "no-fft" else nullcontext

        results: dict[str, dict[str, float | int | list[float]]] = {}
        for mode in cfg.modes:
            if mode not in MODES:
                raise ValueError(f"unknown mode {mode!r}")
            cs.reset()
            with perturb_ctx():
                r = run_mode(mode, field_op, m_local, args.ms, cfg, cs, ctrl)
            # the mixed row moves an f32 payload regardless of the compute dtype
            wire_dtype = torch.float32 if cfg.mixed_wire else cfg.dtype
            wire = wire_bytes_per_rank(cfg.n, world, wire_dtype, cfg.use_rfft)
            n_coll = int(r["n_collectives"])  # type: ignore[arg-type]
            wall_mean_s = float(r["wall_ms_mean"]) / 1e3  # type: ignore[arg-type]
            r["wire_MB_per_rank_per_step"] = n_coll * wire / 1e6
            if cfg.link_bw is not None:
                lb = comm_time_lower_bound(
                    cfg.n,
                    world,
                    wire_dtype,
                    n_coll,
                    cfg.link_bw,
                    alpha=cfg.alpha,
                    use_rfft=cfg.use_rfft,
                )
                r["comm_lb_ms"] = 1e3 * lb
                r["roofline_fraction"] = roofline_fraction(wall_mean_s, lb)
            results[mode] = r

        if rank == 0:
            summary = {
                "config": {
                    "n": cfg.n,
                    "dtype": args.dtype,
                    "world_size": world,
                    "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                    "backend": dist.get_backend(),
                    "reps": cfg.reps,
                    "warmup": cfg.warmup,
                    "seed": cfg.seed,
                    "torch": torch.__version__,
                    "env": _environment(),
                    "gate": gate_info,
                    "link_bw_gbs": args.link_bw_gbs,
                    "alpha_us": args.alpha_us,
                    "use_rfft": cfg.use_rfft,
                    "schedule": cfg.schedule,
                    "uninstrumented": cfg.uninstrumented,
                    "mixed_wire": cfg.mixed_wire,
                    "perturb": cfg.perturb,
                    "comm_split_valid": comm_split_valid,
                    "cpu_run_disclaimer": None
                    if torch.cuda.is_available()
                    else "CPU/gloo run: harness validation only, NOT a performance claim",
                },
                "results": results,
            }
            print(json.dumps(summary, indent=2))
            if args.run_name:
                from spinforge.io import StorageLayout, capture_provenance

                run_dir = StorageLayout.at(args.repo_root).run_dir(args.run_name)
                capture_provenance(run_dir, summary["config"], cfg.seed)  # type: ignore[arg-type]
                (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
