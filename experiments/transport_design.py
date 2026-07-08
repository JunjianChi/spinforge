"""Inverse-design a current protocol that delivers a skyrmion past the skyrmion-Hall drift.

The skyrmion-Hall effect deflects a current-driven skyrmion sideways -- the core obstacle for
racetrack transport. This *designs* the drive: backprop through the STT-driven trajectory (the
differentiable solver's adjoint, gradient-checkpointed over step-chunks to bound memory) to find a
current ``u`` that lands the skyrmion on a straight-ahead target a naive +x drive would miss -- the
device-relevant payoff of the differentiable solver: gradient-based protocol design, not RL.

    PYTHONPATH=src python experiments/transport_design.py --out /tmp/tr.npz   # CPU (small)
    ... --cuda ... ; replot: --from-npz /tmp/tr.npz --plot fig.png
"""

from __future__ import annotations

import argparse
import logging
import math

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from spinforge.core.constants import GAMMA
from spinforge.core.mesh import Mesh
from spinforge.core.system import Material, System
from spinforge.core.textures import skyrmion
from spinforge.io import StorageLayout, capture_provenance, set_seed

MS, A_EX, D_DMI = 3.84e5, 8.78e-12, 1.58e-3
logger = logging.getLogger("transport_design")


def _central(m: torch.Tensor, mesh: Mesh, ax: int) -> torch.Tensor:
    n = mesh.n[ax]
    mp = torch.cat([m.narrow(ax, 0, 1), m, m.narrow(ax, n - 1, 1)], dim=ax)
    return (mp.narrow(ax, 2, n) - mp.narrow(ax, 0, n)) / (2.0 * mesh.dx[ax])


def centroid(
    m: torch.Tensor, xg: torch.Tensor, yg: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable core centroid, weighted by (1 - m_z)/2."""
    w = (1.0 - m[:, :, 0, 2]) * 0.5
    s = w.sum()
    return (w * xg).sum() / s, (w * yg).sum() / s


def driven_rhs(sys: System, m: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """LLG RHS with adiabatic STT, where ``u`` is a (differentiable) length-2 in-plane current."""
    a = sys.material.alpha
    inv = 1.0 / (1.0 + a * a)
    h = sys.effective_field(m)
    mxh = torch.linalg.cross(m, h, dim=-1)
    rhs = -GAMMA * inv * (mxh + a * torch.linalg.cross(m, mxh, dim=-1))
    s = u[0] * _central(m, sys.mesh, 0) + u[1] * _central(m, sys.mesh, 1)
    return rhs - inv * (s + a * torch.linalg.cross(m, s, dim=-1))


def _step(sys: System, m: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    k1 = driven_rhs(sys, m, u)
    k2 = driven_rhs(sys, m + 0.5 * dt * k1, u)
    k3 = driven_rhs(sys, m + 0.5 * dt * k2, u)
    k4 = driven_rhs(sys, m + dt * k3, u)
    m = m + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return m / m.norm(dim=-1, keepdim=True)


def drive(
    sys: System, m: torch.Tensor, u: torch.Tensor, steps: int, dt: float, chunk: int
) -> torch.Tensor:
    """Run ``steps`` driven RK4 steps, checkpointing each chunk so the adjoint fits in memory."""
    done = 0
    while done < steps:
        k = min(chunk, steps - done)

        def run(mm: torch.Tensor, uu: torch.Tensor, k: int = k) -> torch.Tensor:
            for _ in range(k):
                mm = _step(sys, mm, uu, dt)
            return mm

        m = checkpoint(run, m, u, use_reentrant=False)
        done += k
    return m


def relax_skyrmion(mesh: Mesh, dev: torch.device) -> torch.Tensor:
    hz = 0.4 / (4.0 * math.pi * 1e-7)
    mat = Material(ms=MS, a_ex=A_EX, d=D_DMI, alpha=1.0)
    sysr = System(mesh, mat, demag=True, h_ext=(0, 0, hz), chiral_bc=False)
    n = mesh.n[0]
    m = skyrmion(mesh, radius=n * mesh.dx[0] * 0.18, chirality=math.pi / 2).to(dev)
    return sysr.relax(m, steps=3000, dt=2e-13)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--u0", type=float, default=250.0)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--reg", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--from-npz", type=str, default="")
    ap.add_argument("--plot", type=str, default="")
    args = ap.parse_args()
    if args.from_npz:
        _plot(args.from_npz, args.plot or "tr.png")
        return
    dev = torch.device("cuda" if args.cuda else "cpu")
    set_seed(args.seed)
    run = StorageLayout.at().run_dir(f"transport_design_seed{args.seed}")
    capture_provenance(run, vars(args), args.seed)

    n, dx = args.n, 2e-9
    mesh = Mesh(n=(n, n, 1), dx=(dx, dx, dx))
    hz = 0.4 / (4.0 * math.pi * 1e-7)
    mat = Material(ms=MS, a_ex=A_EX, d=D_DMI, alpha=args.alpha)
    sysd = System(mesh, mat, demag=True, h_ext=(0, 0, hz), chiral_bc=False)

    m0 = relax_skyrmion(mesh, dev)
    xs = (torch.arange(n, dtype=torch.float64, device=dev) + 0.5) * dx
    xg, yg = torch.meshgrid(xs, xs, indexing="ij")
    c0 = centroid(m0, xg, yg)

    # baseline: the naive current (u0, 0)
    with torch.no_grad():
        mb = drive(sysd, m0.clone(), torch.tensor([args.u0, 0.0], dtype=torch.float64, device=dev),
                   args.steps, 1e-13, args.chunk)  # fmt: skip
        cb = centroid(mb, xg, yg)
    # well-posed target: the naive drive's final x, but back at the start's y. The only miss is the
    # Hall y-deflection -> the optimizer adds a gentle u_y to cancel it, u_x ~ u0 (no edge-crank).
    target = (cb[0].detach(), c0[1].detach())
    base_miss = math.hypot(float(cb[0] - target[0]), float(cb[1] - target[1])) * 1e9
    logger.info("baseline naive +x drive: Hall miss = %.2f nm", base_miss)

    # design u = (ux, uy) to land on target
    u = torch.tensor([args.u0, 0.0], dtype=torch.float64, device=dev, requires_grad=True)
    u0t = torch.tensor([args.u0, 0.0], dtype=torch.float64, device=dev)
    opt = torch.optim.Adam([u], lr=8.0)
    for it in range(args.iters):
        opt.zero_grad()
        m = drive(sysd, m0.clone(), u, args.steps, 1e-13, args.chunk)
        cx, cy = centroid(m, xg, yg)
        miss_sq = ((cx - target[0]) ** 2 + (cy - target[1]) ** 2) * 1e18
        # light penalty on deviating from the gentle baseline drive: prefer the minimal correction,
        # which keeps the skyrmion clean and central (no edge-cranking reward hack)
        reg = args.reg * ((u - u0t) / args.u0).pow(2).sum()
        loss = miss_sq + reg
        loss.backward()
        opt.step()
        if it % 5 == 0 or it == args.iters - 1:
            dx_nm = float(cx.detach() - target[0]) * 1e9
            dy_nm = float(cy.detach() - target[1]) * 1e9
            miss = math.hypot(dx_nm, dy_nm)
            ux, uy = float(u.detach()[0]), float(u.detach()[1])
            logger.info("iter %2d miss %6.2f nm u=(%.0f,%.0f)", it, miss, ux, uy)

    with torch.no_grad():
        m_final = drive(sysd, m0.clone(), u.detach(), args.steps, 1e-13, args.chunk)
        cf = centroid(m_final, xg, yg)
    fin_miss = math.hypot(float(cf[0] - target[0]), float(cf[1] - target[1])) * 1e9
    logger.info("designed u=(%.1f, %.1f): miss = %.2f nm", float(u[0]), float(u[1]), fin_miss)

    if args.out:
        np.savez(
            args.out,
            mz0=m0[..., 0, 2].cpu().numpy(),
            mzb=mb[..., 0, 2].cpu().numpy(),
            mzf=m_final[..., 0, 2].cpu().numpy(),
            c0=np.array([float(c0[0]), float(c0[1])]),
            cb=np.array([float(cb[0]), float(cb[1])]),
            cf=np.array([float(cf[0]), float(cf[1])]),
            target=np.array([float(target[0]), float(target[1])]),
            base_miss=np.array(base_miss),
            fin_miss=np.array(fin_miss),
            u=u.detach().cpu().numpy(),
        )
        logger.info("saved -> %s", args.out)
        if args.plot:
            _plot(args.out, args.plot)


def _plot(npz: str, png: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(npz)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    c0, cb, cf, tg = d["c0"] * 1e9, d["cb"] * 1e9, d["cf"] * 1e9, d["target"] * 1e9
    ax[0].imshow(d["mz0"].T, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, alpha=0.3,
                 extent=[0, d["mz0"].shape[0] * 2, 0, d["mz0"].shape[1] * 2])  # fmt: skip
    ax[0].plot(*c0, "ks", label="start")
    ax[0].plot(*tg, "g*", ms=15, label="target")
    ax[0].plot(*cb, "rx", ms=10, label=f"naive +x (miss {float(d['base_miss']):.1f} nm)")
    ax[0].plot(*cf, "bo", label=f"designed (miss {float(d['fin_miss']):.1f} nm)")
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].set_xlabel("x (nm)")
    ax[0].set_ylabel("y (nm)")
    ax[0].set_title(f"designed u=({d['u'][0]:.0f}, {d['u'][1]:.0f}) m/s counters the Hall drift")
    ax[1].imshow(d["mzb"].T, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, alpha=0.45)
    im = ax[1].imshow(d["mzf"].T, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, alpha=0.55)
    ax[1].set_title("m_z: naive (faint) vs designed endpoint")
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    fig.colorbar(im, ax=ax[1], shrink=0.8, label="m_z")
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    logger.info("saved figure -> %s", png)


if __name__ == "__main__":
    main()
