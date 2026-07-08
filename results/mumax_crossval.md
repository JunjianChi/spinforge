# Cross-validation against mumax3, same state, same dynamics

Does spinforge integrate the same physics as the community-standard code? Direct A/B
(`experiments/mumax_crossval.py`, RTX 4060): a 9-string FeGe-like skyrmion-lattice state
(96x96x8, 2.5 nm cells, A = 8.78 pJ/m, D_bulk = 1.58 mJ/m^2, Ms = 0.384 MA/m, B_z = 0.176 T) is
seeded in spinforge, written to OVF2, loaded by mumax3.12 with `m.LoadFile`, and both codes
evolve the identical initial state for the same 150 ps at alpha = 1. A dynamic A/B rather than
two independent minimizations, so the local-minimum ambiguity of a many-defect state cannot fake
a mismatch. spinforge runs float64 fixed-step RK4. mumax3 runs its native float32 adaptive RK45.
The measured gap therefore bounds the cross-code physics difference and the integrator and
precision differences together.

| metric | spinforge (f64, RK4) | mumax3.12 (f32, RK45) |
|---|---:|---:|
| string count (interior) | 9 | 9 |
| mid-layer topological charge | -8.3412 | -8.3417 |
| interior nearest-neighbour spacing | 66.5338 nm | 66.5338 nm |

Pointwise endpoint difference between the two codes: max |dm| = 0.0024 over the whole box,
0.0008 in the interior (a margin of about a third of the helical length), mean 1.7e-4.

Agreement after 150 ps of full exchange, bulk-DMI, demag, and Zeeman dynamics is at the 0.1 to
0.25% level of the unit magnetization, the scale of mumax's own single-precision adaptive
tolerance. The bulk-DMI sign convention matched mumax's `Dbulk`. The
harness carries a `--d-sign` probe in case a future term differs. The largest deviations sit in
the edge layer, where the free-surface DMI discretizations differ in detail. The interior is 3x
tighter.

## What this claims and what it does not

- Claims: on the terms the inverse design uses, spinforge and mumax3 compute the same physics to
  within mumax's own numerical noise, on the exact problem class (a bulk skyrmion lattice) of
  the over-one-GPU capacity demonstration. Together with the muMAG standard problem 4 anchor,
  the simulator is cross-validated rather than merely self-consistent.
- Does not claim: feature parity. mumax has finite temperature, more materials and boundary
  conditions, and a faster forward path. This single A/B covers the design demo's regime, not
  all regimes.

Reproduce with `PYTHONPATH=src:. python experiments/mumax_crossval.py --cuda --mumax <mumax3
binary>` (mumax3.12 linux binary from github.com/mumax/3/releases).
