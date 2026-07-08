"""GATE (single GPU/CPU): does the differentiable inverse-design loop work on a skyrmion?

Design a spatially-varying perpendicular field delta_hz so that, after relaxing, the skyrmion
sits at a target position. Gradients flow by backprop through the relaxation (the differentiable
solve). The gate: the loss falls, the skyrmion reaches the target, the design beats a no-design
baseline, and -- verified by eye -- the moved object is a still-intact skyrmion (Q, min
m_z) confirmed by a rendered m_z, not an edge/detection artifact. This de-risks the science BEFORE
the over-one-GPU
bulk-lattice demo (AutoDL). Run:

    PYTHONPATH=src python experiments/inverse_design_gate.py            # CPU
    ... TORCH_CUDA_ARCH_LIST=8.9 ... python experiments/inverse_design_gate.py --cuda
"""

from __future__ import annotations

import argparse
import logging
import math

import torch

from spinforge.core.constants import GAMMA
from spinforge.core.mesh import Mesh
from spinforge.core.observables import topological_charge
from spinforge.core.system import Material, System
from spinforge.core.textures import skyrmion
from spinforge.io import StorageLayout, capture_provenance, set_seed

logger = logging.getLogger("inverse_design_gate")


def _centroid(
    m: torch.Tensor, xg: torch.Tensor, yg: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable down-core centroid, weighted by relu(-m_z).

    A (1 - m_z)/2 weight integrates the whole field, so the optimizer can fake progress by
    deepening an edge twist; relu(-m_z) only sees the down-pointing core (cf. skyrmion_hall).
    """
    w = torch.relu(-m[:, :, 0, 2])
    s = w.sum()
    return (w * xg).sum() / s, (w * yg).sum() / s


def _relax(
    system: System, m: torch.Tensor, design_hz: torch.Tensor, steps: int, dt: float
) -> torch.Tensor:
    a = system.material.alpha
    pref = -GAMMA / (1.0 + a * a)
    for _ in range(steps):

        def rhs(mm: torch.Tensor) -> torch.Tensor:
            h = system.effective_field(mm)
            hz = h[..., 2] + design_hz[:, :, None]  # add the design field to H_z
            h = torch.stack([h[..., 0], h[..., 1], hz], dim=-1)
            mxh = torch.linalg.cross(mm, h, dim=-1)
            return pref * (mxh + a * torch.linalg.cross(mm, mxh, dim=-1))

        k1 = rhs(m)
        k2 = rhs(m + 0.5 * dt * k1)
        k3 = rhs(m + 0.5 * dt * k2)
        k4 = rhs(m + dt * k3)
        m = m + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        m = m / m.norm(dim=-1, keepdim=True)
    return m


def _render(m0: torch.Tensor, m_final: torch.Tensor, design: torch.Tensor, path: str) -> None:
    """Render m_z before/after the design + the design field, so the moved object can be
    eyeballed as an intact skyrmion rather than trusted from a scalar alone."""
    try:
        import matplotlib
    except ImportError:
        logger.warning(
            "matplotlib not installed; skipping the m_z render (needed for the visual check)"
        )
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    mz0 = m0[:, :, 0, 2].detach().cpu().numpy()
    mz1 = m_final[:, :, 0, 2].detach().cpu().numpy()
    ax[0].imshow(mz0.T, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
    ax[0].set_title("m_z: start")
    im = ax[1].imshow(mz1.T, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
    ax[1].set_title("m_z: designed")
    ax[2].imshow(design.detach().cpu().numpy().T, origin="lower", cmap="viridis")
    ax[2].set_title("design field")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.colorbar(im, ax=ax[1], shrink=0.8, label="m_z")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    logger.info("saved render -> %s", path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", type=str, default="results/inverse_design_gate.png")
    args = ap.parse_args()
    dev = torch.device("cuda" if args.cuda else "cpu")
    set_seed(args.seed)
    run = StorageLayout.at().run_dir(f"inverse_design_gate_seed{args.seed}")
    capture_provenance(run, vars(args), args.seed)

    n, dx = 48, 2e-9
    mesh = Mesh(n=(n, n, 1), dx=(dx, dx, dx))
    hz = 0.4 / (4.0 * math.pi * 1e-7)
    mat = Material(ms=3.84e5, a_ex=8.78e-12, d=1.58e-3, alpha=1.0)
    # replicate BC (chiral_bc=False): a clean uniform +z background so the core centroid is the
    # skyrmion's own; the chiral edge twist would contaminate this idealized de-risk test
    system = System(mesh, mat, demag=True, h_ext=(0.0, 0.0, hz), chiral_bc=False)

    xs = (torch.arange(n, dtype=torch.float64, device=dev) + 0.5) * dx
    xg, yg = torch.meshgrid(xs, xs, indexing="ij")
    m0 = skyrmion(mesh, radius=n * dx * 0.3, chirality=math.pi / 2).to(dev)

    steps, dt = 300, 2e-13
    field_scale = 2e5  # A/m; optimize a dimensionless theta so gradients are O(1), not ~1e-22 in SI
    target = (xg.mean() + 8e-9, yg.mean())  # move 8 nm in +x

    # baseline: no design field
    with torch.no_grad():
        zero = torch.zeros(n, n, dtype=torch.float64, device=dev)
        mb = _relax(system, m0.clone(), zero, steps, dt)
        cb = _centroid(mb, xg, yg)
    base_err = math.hypot(float(cb[0] - target[0]), float(cb[1] - target[1])) * 1e9
    logger.info("baseline (no design): offset from target = %.2f nm", base_err)

    theta = torch.zeros(n, n, dtype=torch.float64, device=dev, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.3)
    for it in range(41):
        opt.zero_grad()
        m = _relax(system, m0.clone(), field_scale * theta, steps, dt)
        cx, cy = _centroid(m, xg, yg)
        loss = ((cx - target[0]) ** 2 + (cy - target[1]) ** 2) * 1e18  # nm^2 -> O(1) gradients
        loss = loss + 1e-3 * (theta**2).mean()  # mild magnitude regularization (scaled units)
        loss.backward()
        assert theta.grad is not None
        gnorm = float(theta.grad.norm())
        opt.step()
        if it % 5 == 0 or it == 40:
            err = math.hypot(float(cx - target[0]), float(cy - target[1])) * 1e9
            logger.info(
                "  iter %2d  err %6.2f nm  loss %8.3f  |grad| %.2e", it, err, float(loss), gnorm
            )

    # render and look: the moved object must be a still-intact skyrmion, not an edge/detection
    # artifact
    with torch.no_grad():
        m_final = _relax(system, m0.clone(), field_scale * theta, steps, dt)
        cx, cy = _centroid(m_final, xg, yg)
        final_err = math.hypot(float(cx - target[0]), float(cy - target[1])) * 1e9
        q = float(topological_charge(m_final))
        mz_min = float(m_final[:, :, 0, 2].min())
    logger.info(
        "designed: core offset %.2f nm (baseline %.2f), Q=%.3f, min m_z=%.3f",
        final_err,
        base_err,
        q,
        mz_min,
    )
    logger.info(
        "GATE: improves on baseline=%s, skyrmion intact=%s",
        final_err < base_err,
        q < -0.5 and mz_min < -0.7,
    )
    _render(m0, m_final, field_scale * theta, args.plot)


if __name__ == "__main__":
    main()
