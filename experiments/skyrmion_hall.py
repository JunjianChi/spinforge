"""Current-driven skyrmion motion obeys the Thiele equation (skyrmion-Hall angle).

A relaxed FeGe-class skyrmion is driven by an adiabatic STT current ``u = (u0, 0, 0)``. The Thiele
equation (rigid, no pinning, xi=0) predicts a steady velocity deflected from the current by the
skyrmion-Hall angle ``tan(theta_H) = alpha * D / G``, where ``G = 4*pi*Q`` is the gyrocoupling and
``D = (D_xx + D_yy)/2`` is the dissipative tensor, both read off the relaxed texture
(`G`, `D` integrals share the area element + Ms/gamma prefactor, which cancel in the ratio). We
measure the velocity from the tracked core centroid and compare. Ref: A. A. Thiele, PRL 30, 230
(1973); skyrmion-Hall: Jiang et al., Nat. Phys. 13, 162 (2017).

    PYTHONPATH=src python experiments/skyrmion_hall.py --out /tmp/hall.npz
    ... --cuda ... ; replot: --from-npz /tmp/hall.npz --plot fig.png
"""

from __future__ import annotations

import argparse
import logging
import math

import numpy as np
import torch

from spinforge.core.mesh import Mesh
from spinforge.core.observables import topological_charge
from spinforge.core.system import Material, System
from spinforge.core.textures import skyrmion
from spinforge.io import StorageLayout, capture_provenance, set_seed

MS, A_EX, D_DMI = 3.84e5, 8.78e-12, 1.58e-3  # FeGe-class, bulk DMI
logger = logging.getLogger("skyrmion_hall")


def _central(m: torch.Tensor, mesh: Mesh, ax: int) -> torch.Tensor:
    n = mesh.n[ax]
    lo, hi = m.narrow(ax, 0, 1), m.narrow(ax, n - 1, 1)
    mp = torch.cat([lo, m, hi], dim=ax)
    return (mp.narrow(ax, 2, n) - mp.narrow(ax, 0, n)) / (2.0 * mesh.dx[ax])


def thiele_tensors(m: torch.Tensor, mesh: Mesh, win: slice) -> tuple[float, float]:
    """Dissipative D=(Dxx+Dyy)/2 and gyrocoupling G over a central window (the area element dA is
    kept so G ~ 4*pi*(windowed Q) is interpretable; dA cancels in the alpha*D/G Hall ratio)."""
    da = mesh.dx[0] * mesh.dx[1]
    dxm, dym = _central(m, mesh, 0), _central(m, mesh, 1)
    dxm, dym, mm = dxm[win, win], dym[win, win], m[win, win]
    dxx = float((dxm * dxm).sum()) * da
    dyy = float((dym * dym).sum()) * da
    g = float((mm * torch.linalg.cross(dxm, dym, dim=-1)).sum()) * da  # ~ 4*pi*Q_window
    return 0.5 * (dxx + dyy), g


def core_centroid(m: torch.Tensor, xg: torch.Tensor, yg: torch.Tensor) -> tuple[float, float]:
    """Centroid of the down-core, weighted by relu(-m_z) so the +z edge twist does not bias it."""
    w = torch.relu(-m[:, :, 0, 2])
    s = w.sum()
    return float((w * xg).sum() / s), float((w * yg).sum() / s)


def relax_skyrmion(mesh: Mesh, dev: torch.device, steps: int = 4000) -> torch.Tensor:
    hz = 0.4 / (4.0 * math.pi * 1e-7)
    mat = Material(ms=MS, a_ex=A_EX, d=D_DMI, alpha=1.0)
    # replicate BC: a clean uniform +z background so D, G and the core centroid are the skyrmion's
    # own (the chiral edge twist would contaminate this idealized test)
    sysr = System(mesh, mat, demag=True, h_ext=(0, 0, hz), chiral_bc=False)
    n = mesh.n[0]
    m = skyrmion(mesh, radius=n * mesh.dx[0] * 0.22, chirality=math.pi / 2).to(dev)
    return sysr.relax(m, steps=steps, dt=2e-13)


def drive(
    mesh: Mesh, m0: torch.Tensor, u0: float, alpha: float, dt: float, steps: int, rec: int
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    hz = 0.4 / (4.0 * math.pi * 1e-7)
    mat = Material(ms=MS, a_ex=A_EX, d=D_DMI, alpha=alpha)
    sysd = System(mesh, mat, demag=True, h_ext=(0, 0, hz), u=(u0, 0.0, 0.0), chiral_bc=False)
    n = mesh.n[0]
    xs = (torch.arange(n, dtype=torch.float64, device=m0.device) + 0.5) * mesh.dx[0]
    xg, yg = torch.meshgrid(xs, xs, indexing="ij")
    m = m0.clone()
    times, cent = [], []
    for step in range(steps + 1):
        if step % rec == 0:
            times.append(step * dt)
            cent.append(list(core_centroid(m, xg, yg)))
        if step < steps:
            m = sysd.step_rk4(m, dt)
    return np.array(times), np.array(cent), m


def _fit_velocity(t: np.ndarray, c: np.ndarray) -> tuple[float, float]:
    # skip the short turn-on transient, fit a line to the steady motion
    k = max(1, len(t) // 4)
    vx = float(np.polyfit(t[k:], c[k:, 0], 1)[0])
    vy = float(np.polyfit(t[k:], c[k:, 1], 1)[0])
    return vx, vy


def plot(npz: str, png: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(npz)
    sweep, dg = d["sweep"], float(d["dg"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    # (1) the headline: measured Hall angle vs alpha, against the Thiele line alpha*D/|G|
    al = sweep[:, 0]
    ax[0].plot(al, sweep[:, 1], "o", ms=7, label="measured |θ_H|")
    line = np.degrees(np.arctan(al * dg))
    ax[0].plot(al, line, "-", label="Thiele α·D/|G|")
    ax[0].set_xlabel("Gilbert damping α")
    ax[0].set_ylabel("skyrmion-Hall angle (deg)")
    ax[0].set_title("θ_H ∝ α (Thiele)")
    ax[0].legend()

    # (2) the core path at the reference alpha
    c = d["cent"] * 1e9
    ax[1].plot(c[:, 0] - c[0, 0], c[:, 1] - c[0, 1], "-o", ms=3)
    ax[1].axhline(0, color="0.7", lw=0.5)
    ax[1].set_xlabel("Δx (nm)")
    ax[1].set_ylabel("Δy (nm)")
    ax[1].set_aspect("equal")
    tm, tp = abs(float(d["th_meas"])), float(d["th_pred"])
    ax[1].set_title(f"core path: |θ_H| {tm:.1f}° vs {tp:.1f}°")

    # (3) the texture, start (faint) -> end
    ax[2].imshow(d["mz0"].T, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, alpha=0.4)
    im = ax[2].imshow(d["mz1"].T, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, alpha=0.6)
    ax[2].set_title("m_z: start (faint) → end")
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    fig.colorbar(im, ax=ax[2], shrink=0.8, label="m_z")
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    logger.info("saved figure -> %s", png)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--u0", type=float, default=80.0)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4])
    ap.add_argument(
        "--ref", type=float, default=0.3, help="alpha whose trajectory is saved/plotted"
    )
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--from-npz", type=str, default="")
    ap.add_argument("--plot", type=str, default="")
    args = ap.parse_args()
    if args.from_npz:
        plot(args.from_npz, args.plot or "hall.png")
        return
    dev = torch.device("cuda" if args.cuda else "cpu")
    set_seed(args.seed)
    run = StorageLayout.at().run_dir(f"skyrmion_hall_seed{args.seed}")
    capture_provenance(run, vars(args), args.seed)

    n = 100
    mesh = Mesh(n=(n, n, 1), dx=(2e-9, 2e-9, 2e-9))
    m0 = relax_skyrmion(mesh, dev)  # equilibrium is alpha-independent -> relax once, reuse
    win = slice(n // 4, 3 * n // 4)
    d_tensor, g = thiele_tensors(m0, mesh, win)
    q = float(topological_charge(m0))
    logger.info("relaxed: Q=%.3f  D=%.3f  G=%.3f (4piQ=%.3f)", q, d_tensor, g, 4 * math.pi * q)
    logger.info("Thiele coefficient D/|G| = %.4f", d_tensor / abs(g))

    rows = []
    ref_traj: dict[str, np.ndarray] = {}
    for alpha in args.alphas:
        # |tan(theta_H)| = alpha*D/|G|; deflection SIGN is set by Q, same-sign across the population
        th_pred = math.degrees(math.atan(alpha * d_tensor / abs(g)))
        times, cent, m_end = drive(mesh, m0, args.u0, alpha, 1e-13, args.steps, 50)
        vx, vy = _fit_velocity(times, cent)
        th_meas = math.degrees(math.atan2(vy, vx))
        rows.append((alpha, abs(th_meas), th_pred, th_meas, vx, vy))
        logger.info(
            "alpha=%.2f: |theta_H| meas %5.2f deg vs Thiele %5.2f deg  (deflect %+.1f, v=%.1f m/s)",
            alpha,
            abs(th_meas),
            th_pred,
            th_meas,
            math.hypot(vx, vy),
        )
        if abs(alpha - args.ref) < 1e-9:
            ref_traj = {
                "cent": cent,
                "times": times,
                "mz0": m0[..., 0, 2].cpu().numpy(),
                "mz1": m_end[..., 0, 2].cpu().numpy(),
                "th_meas": np.array(th_meas),
                "th_pred": np.array(th_pred),
            }

    if args.out and ref_traj:
        payload = {"sweep": np.array(rows), "dg": np.array(d_tensor / abs(g)), **ref_traj}
        np.savez(args.out, **payload)  # type: ignore[arg-type]  # numpy stub mistypes **kwargs
        logger.info("saved -> %s", args.out)
        if args.plot:
            plot(args.out, args.plot)


if __name__ == "__main__":
    main()
