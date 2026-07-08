"""The minimal spinforge design loop: optimize a field profile by backprop through the physics.

This is the template to copy for your own problem. The shape is exactly a training loop -- except
there is no dataset: the "forward pass" is a physical relaxation, the "model parameters" are your
design degrees of freedom, and the gradient comes from the adjoint of the solver itself.

    parameters (design)  ->  relax the magnetization under them  ->  loss on the outcome
                     ^                                                    |
                     +----------------- loss.backward() -----------------+

Everything is native torch, so the design variable can just as well be the output of an
nn.Module (a neural parametrization) and this loop can sit inside a larger ML pipeline.

Run it:  PYTHONPATH=src python examples/design_loop.py        (CPU, a few seconds)
"""

from __future__ import annotations

import logging

import torch

from spinforge.core.constants import GAMMA
from spinforge.core.mesh import Mesh
from spinforge.core.observables import topological_charge
from spinforge.core.system import Material, System
from spinforge.core.textures import skyrmion
from spinforge.io import set_seed

logger = logging.getLogger("design_loop")


def run_design_loop(
    n: int = 32,
    relax_steps: int = 60,
    design_iters: int = 12,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, float]:
    """Design a perpendicular field profile that parks a skyrmion on a target position.

    Returns the baseline (no-design) miss, the final miss, and the final topological charge --
    the last one because a scalar objective can always be gamed by destroying the object it
    measures; check the physics survived (full visual verification additionally demands a render).
    """
    set_seed(seed)
    dev = torch.device(device)

    # -- the physics: a FeGe-like chiral film hosting one skyrmion --------------------------------
    mesh = Mesh(n=(n, n, 1), dx=(2.5e-9, 2.5e-9, 2.5e-9))
    mat = Material(ms=3.84e5, a_ex=8.78e-12, d=1.58e-3, alpha=1.0)
    system = System(mesh, mat, demag=True, h_ext=(0.0, 0.0, 2.0e5))
    m0 = skyrmion(mesh, radius=10e-9).to(dev)

    # -- the design variable: a per-cell perpendicular field delta_hz(x, y) -----------------------
    design_hz = torch.zeros(n, n, dtype=torch.float64, device=dev, requires_grad=True)

    # -- a differentiable relax with the design field added to the effective field ----------------
    a = mat.alpha
    pref = -GAMMA / (1.0 + a * a)

    def relax(m: torch.Tensor, dt: float = 1e-13) -> torch.Tensor:
        def rhs(mm: torch.Tensor) -> torch.Tensor:
            h = system.effective_field(mm)
            hz = h[..., 2] + design_hz[:, :, None]
            h = torch.stack([h[..., 0], h[..., 1], hz], dim=-1)
            mxh = torch.linalg.cross(mm, h, dim=-1)
            return pref * (mxh + a * torch.linalg.cross(mm, mxh, dim=-1))

        for _ in range(relax_steps):
            k1 = rhs(m)
            k2 = rhs(m + 0.5 * dt * k1)
            k3 = rhs(m + 0.5 * dt * k2)
            k4 = rhs(m + dt * k3)
            m = m + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            m = m / m.norm(dim=-1, keepdim=True)
        return m

    # -- the objective: put the skyrmion centroid on a target 5 nm away ---------------------------
    xs = ((torch.arange(n, dtype=torch.float64, device=dev) + 0.5) * mesh.dx[0])[:, None]
    ys = ((torch.arange(n, dtype=torch.float64, device=dev) + 0.5) * mesh.dx[1])[None, :]

    def centroid(m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = torch.relu(-m[:, :, 0, 2])  # down-core weight only: the optimizer cannot fake it
        return (w * xs).sum() / w.sum(), (w * ys).sum() / w.sum()

    cx0, cy0 = centroid(m0)
    target = (cx0 + 5e-9, cy0.detach())

    def loss_of(m: torch.Tensor) -> torch.Tensor:
        cx, cy = centroid(m)
        return ((cx - target[0]) ** 2 + (cy - target[1]) ** 2) * 1e18  # nm^2

    with torch.no_grad():
        baseline = float(loss_of(relax(m0.clone()))) ** 0.5

    # -- the loop: exactly optimizer-step training, with physics as the forward pass --------------
    opt = torch.optim.Adam([design_hz], lr=1e5)
    for it in range(design_iters):
        opt.zero_grad()
        loss = loss_of(relax(m0.clone()))
        loss.backward()  # the adjoint: gradients flow through every RK4 step and field term
        opt.step()
        logger.info("iter %2d  miss = %.3f nm", it, float(loss.detach()) ** 0.5)

    with torch.no_grad():
        m_final = relax(m0.clone())
    final = float(loss_of(m_final)) ** 0.5
    q = float(topological_charge(m_final[:, :, 0]))
    logger.info("baseline %.3f nm -> final %.3f nm, Q = %.3f", baseline, final, q)
    return {"baseline_miss_nm": baseline, "final_miss_nm": final, "final_Q": q}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_design_loop()
