"""Direct cross-validation against mumax3 on the problem class (same state, same LLG, A/B).

The fidelity claim this pins: on the terms the inverse design uses (exchange + bulk DMI + demag +
Zeeman, FeGe-like), spinforge and mumax3 integrate the SAME physics. Protocol:

  1. Seed a multi-skyrmion lattice state in spinforge and write it to an OVF2 text file.
  2. Both codes evolve that IDENTICAL initial state under identical material/field for the same
     simulated time at alpha = 1 (a dynamic A/B, not two independent minimizations -- independent
     relaxes of a marginally stable many-defect state may land in different local minima and fake
     a mismatch).
  3. Compare endpoints pointwise (max/mean |dm|), plus physics observables (string count, interior
     nearest-neighbour spacing) -- interior-masked and full-box both, since the free-edge DMI BC
     implementations may differ in the edge layer.

mumax3 runs f32 with adaptive RK45; spinforge runs f64 fixed-step RK4 -- the comparison tolerance
therefore bounds BOTH the cross-code physics agreement and the integrator/precision gap, which is
the honest quantity a user cares about. Run (GPU box):

    PYTHONPATH=src:. python experiments/mumax_crossval.py --cuda \
        --mumax ~/opt/mumax3.12_linux_cuda12.9/mumax3
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
from pathlib import Path

import torch

from experiments.lattice_gate import (
    core_centroids,
    count_cores,
    down_core_mask,
    relax,
    seed_lattice,
)
from spinforge.core.mesh import Mesh
from spinforge.core.observables import topological_charge
from spinforge.core.system import Material, System
from spinforge.io import StorageLayout, capture_provenance, set_seed

logger = logging.getLogger("mumax_crossval")

FEGE = Material(ms=3.84e5, a_ex=8.78e-12, d=1.58e-3, ku=0.0, alpha=1.0)
CELL = 2.5e-9
MU0 = 4.0e-7 * math.pi


def write_ovf2_text(m: torch.Tensor, mesh: Mesh, path: Path) -> None:
    """Minimal OVF 2.0 text writer (x fastest, then y, then z -- the OOMMF/mumax layout)."""
    nx, ny, nz = mesh.n
    lines = [
        "# OOMMF OVF 2.0",
        "# Segment count: 1",
        "# Begin: Segment",
        "# Begin: Header",
        "# Title: spinforge seed",
        "# meshtype: rectangular",
        "# meshunit: m",
        "# xmin: 0\n# ymin: 0\n# zmin: 0",
        f"# xmax: {nx * mesh.dx[0]:.9e}",
        f"# ymax: {ny * mesh.dx[1]:.9e}",
        f"# zmax: {nz * mesh.dx[2]:.9e}",
        "# valuedim: 3",
        "# valuelabels: m_x m_y m_z",
        "# valueunits: 1 1 1",
        f"# xbase: {mesh.dx[0] / 2:.9e}",
        f"# ybase: {mesh.dx[1] / 2:.9e}",
        f"# zbase: {mesh.dx[2] / 2:.9e}",
        f"# xstepsize: {mesh.dx[0]:.9e}",
        f"# ystepsize: {mesh.dx[1]:.9e}",
        f"# zstepsize: {mesh.dx[2]:.9e}",
        f"# xnodes: {nx}\n# ynodes: {ny}\n# znodes: {nz}",
        "# End: Header",
        "# Begin: Data Text",
    ]
    arr = m.detach().cpu().double()
    body = [
        f"{arr[ix, iy, iz, 0]:.12e} {arr[ix, iy, iz, 1]:.12e} {arr[ix, iy, iz, 2]:.12e}"
        for iz in range(nz)
        for iy in range(ny)
        for ix in range(nx)
    ]
    lines += body + ["# End: Data Text", "# End: Segment", ""]
    path.write_text("\n".join(lines))


def read_ovf2_text(path: Path, n: tuple[int, int, int]) -> torch.Tensor:
    """Read an OVF 2.0 text file back into [nx, ny, nz, 3] (x fastest on disk)."""
    nx, ny, nz = n
    vals: list[list[float]] = []
    in_data = False
    for line in path.read_text().splitlines():
        if line.startswith("# Begin: Data Text"):
            in_data = True
            continue
        if line.startswith("# End: Data Text"):
            break
        if in_data and line and not line.startswith("#"):
            vals.append([float(v) for v in line.split()])
    if len(vals) != nx * ny * nz:
        raise ValueError(f"OVF cell count {len(vals)} != {nx * ny * nz}")
    flat = torch.tensor(vals, dtype=torch.float64)  # [nz*ny*nx, 3], x fastest
    return flat.reshape(nz, ny, nx, 3).permute(2, 1, 0, 3).contiguous()


def mumax_script(n: tuple[int, int, int], t_run: float, d_sign: float, out: Path) -> str:
    nx, ny, nz = n
    bz = MU0 * 1.4e5
    return f"""SetGridSize({nx}, {ny}, {nz})
SetCellSize({CELL:.3e}, {CELL:.3e}, {CELL:.3e})
Msat = {FEGE.ms}
Aex = {FEGE.a_ex}
Dbulk = {d_sign * FEGE.d}
alpha = {FEGE.alpha}
B_ext = vector(0, 0, {bz:.9e})
OutputFormat = OVF2_TEXT
m.LoadFile("{out / "seed.ovf"}")
Run({t_run:.3e})
SaveAs(m, "final")
"""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mumax", required=True, help="path to the mumax3 binary")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--nz", type=int, default=8)
    ap.add_argument("--spacing", type=int, default=28)
    ap.add_argument("--radius", type=float, default=18e-9)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--dt", type=float, default=5e-14)
    ap.add_argument("--d-sign", type=float, default=1.0, help="mumax Dbulk sign (convention probe)")
    ap.add_argument("--margin", type=int, default=10)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    n = (args.nx, args.nx, args.nz)
    run = StorageLayout.at().run_dir(f"mumax_crossval_seed{args.seed}")
    capture_provenance(run, {k: str(v) for k, v in vars(args).items()}, args.seed)

    mesh = Mesh(n=n, dx=(CELL, CELL, CELL))
    system = System(mesh, FEGE, demag=True, h_ext=(0.0, 0.0, 1.4e5))
    m0 = seed_lattice(mesh, args.spacing, args.radius)
    write_ovf2_text(m0, mesh, run / "seed.ovf")

    # --- spinforge side: f64 fixed-step RK4, t = steps * dt --------------------------------------
    t_run = args.steps * args.dt
    logger.info("spinforge: %d RK4 steps (%.0f ps, f64, %s)", args.steps, 1e12 * t_run, device)
    with torch.no_grad():
        m_sf = relax(system, m0.to(device=device, dtype=torch.float64), args.steps, args.dt).cpu()

    # --- mumax side: same seed, same simulated time ----------------------------------------------
    (run / "ab.mx3").write_text(mumax_script(n, t_run, args.d_sign, run))
    logger.info("mumax3: Run(%.0f ps)", 1e12 * t_run)
    subprocess.run([args.mumax, "-f", str(run / "ab.mx3")], check=True, capture_output=True)
    m_mx = read_ovf2_text(run / "ab.out" / "final.ovf", n)

    # --- compare ----------------------------------------------------------------------------------
    diff = (m_sf - m_mx).norm(dim=-1)
    inner = diff[args.margin : -args.margin, args.margin : -args.margin, :]
    mask_sf = down_core_mask(m_sf, args.margin)
    mask_mx = down_core_mask(m_mx, args.margin)

    def nn_spacing(mm: torch.Tensor) -> float:
        cents = core_centroids(down_core_mask(mm, args.margin), mesh)
        nn = [min(math.dist(c, d) for d in cents if d != c) for c in cents if len(cents) > 1]
        return 1e9 * sum(nn) / len(nn) if nn else float("nan")

    out = {
        "max_dm": float(diff.max()),
        "mean_dm": float(diff.mean()),
        "interior_max_dm": float(inner.max()),
        "interior_mean_dm": float(inner.mean()),
        "count_spinforge": count_cores(mask_sf),
        "count_mumax": count_cores(mask_mx),
        "Q_spinforge": float(topological_charge(m_sf[:, :, args.nz // 2])),
        "Q_mumax": float(topological_charge(m_mx[:, :, args.nz // 2])),
        "nn_spacing_spinforge_nm": nn_spacing(m_sf),
        "nn_spacing_mumax_nm": nn_spacing(m_mx),
    }
    (run / "summary.json").write_text(json.dumps(out, indent=2))
    logger.info("A/B: %s", json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
