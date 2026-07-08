"""Micromagnetic Standard Problem 4 (muMAG) — dynamic reversal of a Permalloy film.

Exercises the full LLG dynamics (exchange + demag + Zeeman + precession + low-damping ringing)
end-to-end against the community oracle, NOT just a static field term. Geometry 500x125x3 nm,
A = 1.3e-11 J/m, Ms = 8e5 A/m, K = 0: relax to the equilibrium S-state (saturate along [1,1,1],
relax at zero field), then apply field 1 = (-24.6, 4.3, 0) mT or field 2 = (-35.5, -6.3, 0) mT
instantaneously (alpha = 0.02) and record the spatially-averaged magnetization vs time.

Reference: <mx> first crosses zero at ~0.136 ns for field 1 (muMAG/OOMMF reference solutions,
https://www.ctcms.nist.gov/~rdm/std4/spec4.html); the canonical mumax3/OOMMF comparison uses this
same 200x50x1 grid (2.5 nm cells). Saves the trajectory to an .npz for plotting/scoring.

    PYTHONPATH=src python experiments/mumag_sp4.py --out /tmp/sp4.npz
    ... TORCH_CUDA_ARCH_LIST=8.9 ... python experiments/mumag_sp4.py --cuda --out /tmp/sp4.npz
"""

from __future__ import annotations

import argparse
import logging
import math

import numpy as np
import torch

from spinforge.core.mesh import Mesh
from spinforge.core.system import Material, System
from spinforge.io import StorageLayout, capture_provenance, set_seed

MU0 = 4.0e-7 * math.pi
MS = 8.0e5
A_EX = 1.3e-11
FIELDS_MT = {"1": (-24.6, 4.3, 0.0), "2": (-35.5, -6.3, 0.0)}

logger = logging.getLogger("mumag_sp4")


def h_from_mt(mt: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert an applied field given as mu0*H in mT to H in A/m."""
    return tuple(v * 1e-3 / MU0 for v in mt)  # type: ignore[return-value]


def s_state(
    mesh: Mesh,
    device: torch.device,
    *,
    dt: float = 2e-13,
    max_steps: int = 60000,
    tol: float = 1e-4,
) -> torch.Tensor:
    """Equilibrium S-state: saturate uniformly along [1,1,1], relax at zero field with alpha = 1.

    Returns when the max normalized torque |m x h|/Ms falls below ``tol`` (or at ``max_steps``).
    """
    mat = Material(ms=MS, a_ex=A_EX, alpha=1.0)
    system = System(mesh, mat, demag=True)
    m = torch.ones(*mesh.n, 3, dtype=torch.float64, device=device)
    m = m / m.norm(dim=-1, keepdim=True)
    for step in range(max_steps):
        m = system.step_rk4(m, dt)
        if step % 500 == 0:
            torque = torch.linalg.cross(m, system.effective_field(m), dim=-1)
            tmax = float(torque.abs().max()) / MS
            logger.info("s-state step %d  max torque %.2e", step, tmax)
            if tmax < tol:
                break
    return m


def run_field(
    mesh: Mesh,
    m0: torch.Tensor,
    mt: tuple[float, float, float],
    *,
    alpha: float = 0.02,
    dt: float = 1e-13,
    total_time: float = 1e-9,
    record_every: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the LLG under the applied field; return (times [s], m_avg [N, 3] normalized)."""
    system = System(mesh, Material(ms=MS, a_ex=A_EX, alpha=alpha), demag=True, h_ext=h_from_mt(mt))
    nsteps = int(round(total_time / dt))
    m = m0.clone()
    times: list[float] = []
    mavg: list[list[float]] = []
    for step in range(nsteps + 1):
        if step % record_every == 0:
            times.append(step * dt)
            mavg.append(m.reshape(-1, 3).mean(0).tolist())
        if step < nsteps:
            m = system.step_rk4(m, dt)
    return np.array(times), np.array(mavg)


def zero_crossing(times: np.ndarray, mx: np.ndarray) -> float:
    """First positive->non-positive crossing of <mx>, linearly interpolated (s; nan if none)."""
    for i in range(1, len(mx)):
        if mx[i - 1] > 0.0 >= mx[i]:
            frac = mx[i - 1] / (mx[i - 1] - mx[i])
            return float(times[i - 1] + (times[i] - times[i - 1]) * frac)
    return float("nan")


def plot(npz_path: str, png_path: str) -> None:
    """Render <mx>,<my>,<mz> vs time for both fields from a saved trajectory (lazy matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(npz_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, key, title in zip(axes, ("1", "2"), ("field 1", "field 2"), strict=True):
        t, m = data[f"t{key}"] * 1e9, data[f"m{key}"]
        for c, lbl in enumerate(("⟨mx⟩", "⟨my⟩", "⟨mz⟩")):
            ax.plot(t, m[:, c], label=lbl)
        tc = zero_crossing(data[f"t{key}"], m[:, 0]) * 1e9
        ax.axvline(tc, color="k", ls=":", lw=0.8)
        ax.axhline(0.0, color="0.7", lw=0.5)
        ax.set_title(f"{title}: ⟨mx⟩=0 at {tc:.3f} ns")
        ax.set_xlabel("time (ns)")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("⟨m⟩ / Ms")
    fig.suptitle("muMAG Standard Problem 4 — spinforge (200×50×1, 2.5 nm)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    logger.info("saved figure -> %s", png_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--field", choices=["1", "2", "both"], default="both")
    ap.add_argument("--dt", type=float, default=1e-13)
    ap.add_argument("--time", type=float, default=1e-9)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--from-npz", type=str, default="", help="skip the sim, just plot this .npz")
    ap.add_argument("--plot", type=str, default="", help="write the trajectory figure here (.png)")
    args = ap.parse_args()
    if args.from_npz:
        plot(args.from_npz, args.plot or "sp4.png")
        return
    dev = torch.device("cuda" if args.cuda else "cpu")
    set_seed(args.seed)
    run = StorageLayout.at().run_dir(f"mumag_sp4_seed{args.seed}")
    capture_provenance(run, vars(args), args.seed)

    mesh = Mesh(n=(200, 50, 1), dx=(2.5e-9, 2.5e-9, 3.0e-9))
    logger.info("relaxing S-state on %s ...", dev)
    m0 = s_state(mesh, dev)
    s_avg = m0.reshape(-1, 3).mean(0).tolist()
    logger.info("S-state <m> = (%.3f, %.3f, %.3f)", *s_avg)

    out: dict[str, np.ndarray] = {"s_state": np.array(s_avg)}
    for key in ["1", "2"] if args.field == "both" else [args.field]:
        times, mavg = run_field(mesh, m0, FIELDS_MT[key], dt=args.dt, total_time=args.time)
        tc = zero_crossing(times, mavg[:, 0])
        logger.info(
            "field %s: <mx> zero-crossing = %.4f ns  final <m> = (%.3f, %.3f, %.3f)",
            key,
            tc * 1e9,
            *mavg[-1].tolist(),
        )
        out[f"t{key}"] = times
        out[f"m{key}"] = mavg
    if args.out:
        np.savez(args.out, **out)  # type: ignore[arg-type]  # numpy stub mistypes **kwargs
        logger.info("saved trajectory -> %s", args.out)
        if args.plot:
            plot(args.out, args.plot)


if __name__ == "__main__":
    main()
