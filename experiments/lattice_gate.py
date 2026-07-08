"""GATE (single GPU): must the bulk skyrmion-lattice inverse design be believed at all?

Three cheap pre-studies that decide whether the over-one-GPU science demo happens
(if any fails, the science claim is dropped and the pure-systems headline stands):

  respond  -- (i) do the strings RESPOND to a graded anisotropy profile? Relax a seeded lattice
              under delta_Ku(x) = theta * Ku_SCALE * (x/Lx - 1/2); the down-core population must
              shift toward the low-Ku side vs the theta=0 control, with the lattice intact
              (rendered m_z + per-layer Q).
  volume   -- (ii) is the ARRANGEMENT volume-sensitive? Relax the same seeded areal density and
              field in a box of side L and 2L (same cells); compare the interior nearest-neighbour
              spacing. A finite-size bias (spacings differ) is what makes the big volume genuinely
              necessary (a spatial, measured reason); no bias -> the demo could be coarsened and
              the claim dies.
  gradient -- (iii) is the GRADIENT well-behaved? L(theta) = squared miss of the population
              x-centroid vs a target; sweep theta, autograd dL/dtheta at each point, compare to
              the finite-difference secants of the same sweep and check for flat/stepped regions
              (the overfitting/snapping failure mode).

FeGe-like bulk DMI material (A = 8.78 pJ/m, D = 1.58 mJ/m^2, Ms = 0.384 MA/m, L_D = 4 pi A / D
~ 70 nm); relaxation at alpha = 1 under a +z field in the skyrmion-lattice window. Every verdict
is rendered and looked at before it is believed. Run:

    PYTHONPATH=src python experiments/lattice_gate.py --part respond --cuda
    PYTHONPATH=src python experiments/lattice_gate.py --part volume --cuda
    PYTHONPATH=src python experiments/lattice_gate.py --part gradient --cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import math

import torch

from spinforge.core.anisotropy import uniaxial_anisotropy_field
from spinforge.core.constants import GAMMA
from spinforge.core.mesh import Mesh
from spinforge.core.observables import topological_charge
from spinforge.core.system import Material, System
from spinforge.io import StorageLayout, capture_provenance, set_seed

logger = logging.getLogger("lattice_gate")

# FeGe-like (design.md oracle set); the graded design term modulates Ku around zero
FEGE = Material(ms=3.84e5, a_ex=8.78e-12, d=1.58e-3, ku=0.0, alpha=1.0)
CELL = 2.5e-9


def seed_lattice(mesh: Mesh, spacing_cells: int, radius: float) -> torch.Tensor:
    """Uniform-up background stamped with a triangular-ish grid of down-core skyrmions."""
    nx, ny, nz = mesh.n
    m = torch.zeros(nx, ny, nz, 3, dtype=torch.float64)
    m[..., 2] = 1.0
    xs = (torch.arange(nx, dtype=torch.float64) + 0.5) * mesh.dx[0]
    ys = (torch.arange(ny, dtype=torch.float64) + 0.5) * mesh.dx[1]
    x, y = torch.meshgrid(xs, ys, indexing="ij")
    half = spacing_cells // 2
    for i, cx_i in enumerate(range(half, nx, spacing_cells)):
        row_off = half if i % 2 else 0  # offset alternate rows -> near-triangular packing
        for cy_i in range(half + row_off, ny, spacing_cells):
            cx, cy = (cx_i + 0.5) * mesh.dx[0], (cy_i + 0.5) * mesh.dx[1]
            r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            phi = torch.atan2(y - cy, x - cx)
            theta_p = (math.pi * (1.0 - r / radius)).clamp(0.0, math.pi)
            core = r < radius
            plane = torch.sin(theta_p)
            for z in range(nz):
                m[..., z, 0] = torch.where(
                    core, plane * torch.cos(phi + math.pi / 2.0), m[..., z, 0]
                )
                m[..., z, 1] = torch.where(
                    core, plane * torch.sin(phi + math.pi / 2.0), m[..., z, 1]
                )
                m[..., z, 2] = torch.where(core, torch.cos(theta_p), m[..., z, 2])
    return m / m.norm(dim=-1, keepdim=True)


def relax(
    system: System,
    m: torch.Tensor,
    steps: int,
    dt: float,
    dku_map: torch.Tensor | None = None,
    checkpoint_every: int = 0,
) -> torch.Tensor:
    """Relax with an optional graded-anisotropy design term added to the effective field.

    The graded term rides ON TOP of the Material (the design parameter enters differentiably);
    chunked checkpointing bounds the adjoint memory exactly as in the transport experiment.
    """
    a = system.material.alpha
    pref = -GAMMA / (1.0 + a * a)

    def rhs(mm: torch.Tensor) -> torch.Tensor:
        h = system.effective_field(mm)
        if dku_map is not None:
            h = h + uniaxial_anisotropy_field(
                mm, system.mesh, dku_map, (0.0, 0.0, 1.0), system.material.ms
            )
        mxh = torch.linalg.cross(mm, h, dim=-1)
        return pref * (mxh + a * torch.linalg.cross(mm, mxh, dim=-1))

    def run_chunk(m_in: torch.Tensor, k: int) -> torch.Tensor:
        for _ in range(k):
            k1 = rhs(m_in)
            k2 = rhs(m_in + 0.5 * dt * k1)
            k3 = rhs(m_in + 0.5 * dt * k2)
            k4 = rhs(m_in + dt * k3)
            m_in = m_in + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            m_in = m_in / m_in.norm(dim=-1, keepdim=True)
        return m_in

    if checkpoint_every <= 0 or not (torch.is_grad_enabled() and m.requires_grad):
        return run_chunk(m, steps)
    done = 0
    while done < steps:
        k = min(checkpoint_every, steps - done)
        m = torch.utils.checkpoint.checkpoint(run_chunk, m, k, use_reentrant=False)
        done += k
    return m


def down_core_mask(m: torch.Tensor, margin: int, z: int | None = None) -> torch.Tensor:
    """Interior down-core mask of the mid layer: the strings' footprint, edge band EXCLUDED.

    The chiral free-surface twist puts an m_z < 0 halo around the box edge that a bare threshold
    counts as extra "strings" (a detection artifact a scalar check would miss, caught by the
    render); a margin of ~L_D/4 cells removes it, and the count must then agree with the mid-layer
    Q."""
    zi = m.shape[2] // 2 if z is None else z
    mask = m[:, :, zi, 2] < -0.3
    mask[:margin, :] = False
    mask[-margin:, :] = False
    mask[:, :margin] = False
    mask[:, -margin:] = False
    return mask


def half_weights(m: torch.Tensor, margin: int) -> tuple[float, float]:
    """relu(-m_z) population weight in the low-x vs high-x interior halves (size response)."""
    zi = m.shape[2] // 2
    w = torch.relu(-m[:, :, zi, 2]).clone()
    w[:margin, :] = 0.0
    w[-margin:, :] = 0.0
    w[:, :margin] = 0.0
    w[:, -margin:] = 0.0
    half = m.shape[0] // 2
    return float(w[:half].sum()), float(w[half:].sum())


def count_cores(mask: torch.Tensor) -> int:
    """Connected components (4-neighbour) of the core mask = number of strings in the layer."""
    lab = torch.zeros(mask.shape, dtype=torch.int32)
    nxt = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if bool(mask[i, j]) and lab[i, j] == 0:
                nxt += 1
                stack = [(i, j)]
                lab[i, j] = nxt
                while stack:
                    a, b = stack.pop()
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        p, q = a + da, b + db
                        if (
                            0 <= p < mask.shape[0]
                            and 0 <= q < mask.shape[1]
                            and bool(mask[p, q])
                            and lab[p, q] == 0
                        ):
                            lab[p, q] = nxt
                            stack.append((p, q))
    return nxt


def core_centroids(mask: torch.Tensor, mesh: Mesh) -> list[tuple[float, float]]:
    """Centroid (m) of each connected core component."""
    lab = torch.zeros(mask.shape, dtype=torch.int32)
    comps: list[list[tuple[int, int]]] = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if bool(mask[i, j]) and lab[i, j] == 0:
                comp = [(i, j)]
                lab[i, j] = 1
                stack = [(i, j)]
                while stack:
                    a, b = stack.pop()
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        p, q = a + da, b + db
                        if (
                            0 <= p < mask.shape[0]
                            and 0 <= q < mask.shape[1]
                            and bool(mask[p, q])
                            and lab[p, q] == 0
                        ):
                            lab[p, q] = 1
                            comp.append((p, q))
                            stack.append((p, q))
                comps.append(comp)
    return [
        (
            (sum(p[0] for p in c) / len(c) + 0.5) * mesh.dx[0],
            (sum(p[1] for p in c) / len(c) + 0.5) * mesh.dx[1],
        )
        for c in comps
    ]


def population_x_centroid(m: torch.Tensor, mesh: Mesh, margin: int) -> torch.Tensor:
    """Differentiable interior x-centroid of the down-core population (mid layer).

    relu(-m_z) weighting sees only down cores; the multiplicative interior mask keeps the edge
    twist band out of the population without breaking differentiability."""
    zi = m.shape[2] // 2
    w = torch.relu(-m[:, :, zi, 2])
    box = torch.zeros_like(w)
    box[margin:-margin, margin:-margin] = 1.0
    w = w * box
    xs = ((torch.arange(m.shape[0], dtype=m.dtype, device=m.device) + 0.5) * mesh.dx[0])[:, None]
    return (w * xs).sum() / w.sum()


def render(m: torch.Tensor, title: str, path: str) -> None:
    """The mid-layer m_z heatmap the verdict is read from (render and look)."""
    try:
        import matplotlib
    except ImportError:
        logger.warning(
            "matplotlib missing: NO render -> the verdict may not be trusted without a look at m_z"
        )
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zi = m.shape[2] // 2
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(m[:, :, zi, 2].T.cpu().numpy(), origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="m_z")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    logger.info("render -> %s", path)


def make_system(n: tuple[int, int, int], hz: float, device: torch.device) -> tuple[Mesh, System]:
    mesh = Mesh(n=n, dx=(CELL, CELL, CELL))
    return mesh, System(mesh, FEGE, demag=True, h_ext=(0.0, 0.0, hz))


def part_respond(args: argparse.Namespace, run_dir: str, device: torch.device) -> dict:
    """(i) strings respond to the graded profile; the lattice stays intact (render + Q)."""
    n = (args.nx, args.nx, args.nz)
    mesh, system = make_system(n, args.hz, device)
    m0 = seed_lattice(mesh, args.spacing, args.radius).to(device=device, dtype=args.torch_dtype)
    lx = n[0] * CELL
    xs = ((torch.arange(n[0], dtype=args.torch_dtype, device=device) + 0.5) * CELL / lx) - 0.5

    out: dict = {}
    for name, theta in (("control", 0.0), ("graded", 1.0)):
        dku = (theta * args.ku_scale * xs).reshape(-1, 1, 1, 1)
        with torch.no_grad():
            mf = relax(system, m0.clone(), args.steps, args.dt, dku_map=dku)
        mask = down_core_mask(mf, args.margin)
        q = float(topological_charge(mf[:, :, mf.shape[2] // 2]))
        w_lo, w_hi = half_weights(mf, args.margin)
        out[name] = {
            "count": count_cores(mask),
            "x_centroid_nm": 1e9 * float(population_x_centroid(mf, mesh, args.margin)),
            "mid_layer_Q": q,
            "weight_hi_over_lo": w_hi / w_lo,
        }
        title = f"{name}: dKu = {theta} * {args.ku_scale:g} * (x/Lx - 1/2)"
        render(mf, title, f"{run_dir}/{name}.png")
        logger.info("%s: %s", name, out[name])

    shift = out["graded"]["x_centroid_nm"] - out["control"]["x_centroid_nm"]
    asym = out["graded"]["weight_hi_over_lo"] / out["control"]["weight_hi_over_lo"]
    out["x_shift_nm"] = shift
    out["asymmetry_vs_control"] = asym
    # verdict criteria (numbers in the summary; the render is the arbiter of "intact"). The
    # response may appear as migration (centroid) or size modulation (half-weight asymmetry).
    out["verdict"] = {
        "responds": abs(shift) > 2.0 or abs(asym - 1.0) > 0.10,
        "count_matches_Q": abs(out["graded"]["count"] + out["graded"]["mid_layer_Q"]) <= 1.5,
        "count_stable": out["graded"]["count"] >= max(1, out["control"]["count"] - 1),
    }
    return out


def part_volume(args: argparse.Namespace, run_dir: str, device: torch.device) -> dict:
    """(ii) interior nearest-neighbour spacing at box L vs 2L (same cells, field, seeding)."""
    out: dict = {}
    for name, nx in (("L", args.nx), ("2L", 2 * args.nx)):
        mesh, system = make_system((nx, nx, args.nz), args.hz, device)
        m0 = seed_lattice(mesh, args.spacing, args.radius).to(device=device, dtype=args.torch_dtype)
        with torch.no_grad():
            mf = relax(system, m0, args.steps, args.dt)
        mask = down_core_mask(mf, args.margin)
        cents = core_centroids(mask, mesh)
        lx = nx * CELL
        margin = 0.3 * args.nx * CELL  # interior = safely away from the free edges
        interior = [c for c in cents if margin < c[0] < lx - margin and margin < c[1] < lx - margin]
        nn: list[float] = []
        for c in interior:
            ds = [math.dist(c, d) for d in cents if d != c]
            if ds:
                nn.append(min(ds))
        out[name] = {
            "count_total": len(cents),
            "count_interior": len(interior),
            "nn_spacing_nm": 1e9 * (sum(nn) / len(nn)) if nn else float("nan"),
        }
        render(mf, f"box {name} ({nx}x{nx}x{args.nz})", f"{run_dir}/box_{name}.png")
        logger.info("%s: %s", name, out[name])
    a, b = out["L"]["nn_spacing_nm"], out["2L"]["nn_spacing_nm"]
    out["spacing_rel_diff"] = abs(a - b) / b if b == b and b > 0 else float("nan")
    out["verdict"] = {"volume_sensitive_gt_3pct": out["spacing_rel_diff"] > 0.03}
    return out


def part_gradient(args: argparse.Namespace, run_dir: str, device: torch.device) -> dict:
    """(iii) L(theta) smooth; autograd dL/dtheta consistent with the sweep's FD secants."""
    n = (args.nx, args.nx, args.nz)
    mesh, system = make_system(n, args.hz, device)
    m0 = seed_lattice(mesh, args.spacing, args.radius).to(device=device, dtype=args.torch_dtype)
    lx = n[0] * CELL
    xs = ((torch.arange(n[0], dtype=args.torch_dtype, device=device) + 0.5) * CELL / lx) - 0.5
    target = 0.45 * lx

    thetas = [i / (args.npts - 1) * 2.0 - 1.0 for i in range(args.npts)]  # [-1, 1]
    losses: list[float] = []
    grads: list[float] = []
    for th in thetas:
        theta = torch.tensor(th, dtype=args.torch_dtype, device=device, requires_grad=True)
        dku = (theta * args.ku_scale * xs).reshape(-1, 1, 1, 1)
        mf = relax(
            system,
            m0.clone().requires_grad_(True),
            args.steps,
            args.dt,
            dku_map=dku,
            checkpoint_every=25,
        )
        loss = (population_x_centroid(mf, mesh, args.margin) - target).pow(2) * 1e18  # nm^2
        loss.backward()
        assert theta.grad is not None
        losses.append(float(loss))
        grads.append(float(theta.grad))
        logger.info("theta=%+.2f  L=%.4f nm^2  dL/dtheta=%+.4f", th, losses[-1], grads[-1])

    # FD secants of the sweep vs the mean of the endpoint autograd slopes
    fd_ok, pairs = [], []
    for i in range(len(thetas) - 1):
        h = thetas[i + 1] - thetas[i]
        secant = (losses[i + 1] - losses[i]) / h
        mean_grad = 0.5 * (grads[i] + grads[i + 1])
        pairs.append({"secant": secant, "mean_autograd": mean_grad})
        denom = max(abs(secant), abs(mean_grad), 1e-6)
        fd_ok.append(abs(secant - mean_grad) / denom < 0.35)  # loose: L(theta) is not quadratic
    nonflat = sum(1 for g in grads if abs(g) > 1e-3)
    out = {
        "thetas": thetas,
        "losses_nm2": losses,
        "grads": grads,
        "fd_pairs": pairs,
        "verdict": {
            "fd_consistent_fraction": sum(fd_ok) / len(fd_ok),
            "nonflat_fraction": nonflat / len(grads),
            "fd_ok": sum(fd_ok) / len(fd_ok) >= 0.7,
            "not_flat": nonflat / len(grads) >= 0.7,
        },
    }
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["respond", "volume", "gradient"], required=True)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--nz", type=int, default=8)
    ap.add_argument("--spacing", type=int, default=28, help="seed spacing (cells) ~ L_D")
    ap.add_argument("--radius", type=float, default=18e-9)
    ap.add_argument("--hz", type=float, default=1.4e5, help="+z field (A/m), SkX window")
    ap.add_argument(
        "--ku-scale",
        type=float,
        default=2.0e4,
        help="graded-dKu amplitude at theta=1 (J/m^3); ~20%% of mu0*Ms^2/2 for FeGe",
    )
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--dt", type=float, default=5e-14)
    ap.add_argument("--npts", type=int, default=5, help="theta sweep points (gradient part)")
    ap.add_argument("--margin", type=int, default=10, help="interior margin (cells), ~L_D/4")
    ap.add_argument("--f64", action="store_true", help="float64 (default f32 on GPU)")
    args = ap.parse_args()
    args.torch_dtype = torch.float64 if args.f64 or not args.cuda else torch.float32

    set_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    run = StorageLayout.at().run_dir(f"lattice_gate_{args.part}_seed{args.seed}")
    capture_provenance(run, {k: str(v) for k, v in vars(args).items()}, args.seed)

    part = {"respond": part_respond, "volume": part_volume, "gradient": part_gradient}[args.part]
    out = part(args, str(run), device)
    (run / "summary.json").write_text(json.dumps(out, indent=2))
    logger.info("VERDICT %s: %s", args.part, json.dumps(out.get("verdict", {})))


if __name__ == "__main__":
    main()
