# Three checks before the over-one-GPU lattice demo

Three single-GPU pre-studies (`experiments/lattice_gate.py`, RTX 4060) that decide whether a
large inverse-design run on a bulk skyrmion lattice is worth doing. Material: FeGe-like bulk
DMI, A = 8.78 pJ/m, D = 1.58 mJ/m^2, Ms = 0.384 MA/m, helical length about 70 nm, 2.5 nm cells,
+z field 1.4e5 A/m. All three verdicts were read off rendered m_z maps first and scalars second.

| check | criterion | measured | verdict |
|---|---|---:|---|
| respond | centroid shift over 2 nm or half-weight asymmetry over 10% | +4.9 nm and +16.1% | pass |
| volume | interior lattice spacing differs between box L and 2L by over 3% | 7.6% (63.4 to 68.6 nm) | pass |
| gradient | finite-difference-consistent secants at least 70%, non-flat at least 70% | 4/4 and 5/5 | pass |

## Respond: strings follow a graded anisotropy profile

delta Ku(x) = theta times 1e5 J/m^3 times (x/Lx - 1/2) over a 9-string lattice (96x96x8), 12000
RK4 steps (600 ps), theta = 1 against the theta = 0 control. The population x-centroid moves
+4.9 nm toward the high-Ku side and the high-to-low half-weight ratio grows 16%. The response is
partly migration and partly core-size modulation, with visibly larger cores on the high-Ku side
(`figures/lattice_gate_graded.png`). The lattice stays intact: the string count of 9 matches the
mid-layer |Q| of 7.8 within tolerance, and there is no stripe-out. One seeded string sits near
the top edge band in the control too, so that is seeding geometry, not the grading. At a weaker
grading (2e4, 150 ps) the same signature exists but small (+1.5 nm, +4.3%). A design loss at
scale should target density and size observables and budget relaxation times of nanoseconds.

## Volume: the arrangement is not box-size invariant

Same cells, field, and seeded areal density. Box L (96^2, 9 strings) against 2L (192^2, 46
strings, `figures/lattice_gate_box2L.png`, a clean near-triangular lattice). Interior
nearest-neighbour spacing with the same 72 nm absolute edge exclusion: 63.4 nm against 68.6 nm,
a 7.6% difference. The small box compresses the lattice through its boundary, so a small
surrogate gives the wrong arrangement. This measured finite-size bias is what makes the large
volume a physics requirement. Caveat: after the edge exclusion the L box holds a single interior
string, one spacing sample against 22 in the 2L box.

## Gradient: the adjoint through the relaxation is healthy

L(theta) is the squared miss of the interior population centroid against a target, theta in
[-1, 1], 5 points, 1500 checkpointed RK4 steps each, with autograd dL/dtheta at every point. The
loss sweep is smooth and monotone (22.8 to 41.7 nm^2), every gradient is nonzero (7.8 to 11.1),
and every interval's finite-difference secant matches the endpoint-gradient mean (8.22 against
8.22 on [-1, -0.5]). No snapping and no flat regions.

## A detector lesson

The first run counted 17 strings where the topological charge said about 8. A bare m_z < -0.3
threshold counts the chiral free-surface edge twist as strings. All observables are now
interior-masked (a margin of about a third of the helical length) and the verdict requires the count
to match |Q|. A scalar detector without a render would have shipped a false pass.

## Consequence

Cells are pinned at nanometer scale by the helical length. The arrangement is pinned by many
strings plus the long-range field, measured above rather than asserted. Gradients are
trustworthy through the checkpointed relaxation.
