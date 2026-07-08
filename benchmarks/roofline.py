"""Communication roofline for the distributed demag transpose (Phase-C measurement instrument).

Two honest yardsticks, per the prior-art study (heFFTe / cuFFTMp / magnum.np.distributed,
first-hand, 2026-07):

1. **Comm lower bound** for OUR op: the all-to-all transpose moves a known number of bytes per
   rank per collective and issues (P-1) messages per rank per collective, so at a measured
   per-message latency alpha and per-rank link bandwidth B the wall time can never beat
   ``n_collectives * ((P-1) * alpha + wire_bytes_per_rank / B)`` (the alpha-beta transpose
   cost; alpha=0 recovers the bytes-only bound). alpha and B come from a measured payload sweep
   (``fit_alpha_beta``) -- the alpha term is what lets the regime map explain the small-message
   end of strong scaling, where the per-pair message shrinks as 1/P^2. The achieved fraction of
   this bound is the tie-instrument: production distributed FFTs sit near-but-below their own comm
   roofline (heFFTe reports reaching a large fraction of its Psi ceiling, ICCS 2020; cuFFTMp
   reports a high fraction of peak machine bandwidth in its NVIDIA docs) -- exact fractions are
   theirs to state, so this module computes only OUR fraction and leaves the cross-code numbers to
   their sources.
2. **heFFTe-style throughput roofline** ``Psi = 5 * log2(N) * P * B / (alpha * r)`` for the
   cross-code regime map (the same ceiling heFFTe defines, ICCS 2020, so the comparison is
   apples-to-apples).

Collective *counts* are measured, not assumed: forward = 6 (3 components x 2 transposes) and
forward+adjoint = 12 are structural (`tests/dist/test_traffic.py`); the checkpoint-recompute count
(15 = 2.5N at the tested config) is implementation-dependent (torch selective recompute), so the
harness counts collectives in situ rather than trusting a table.

All bandwidths are bytes/second; all times seconds. No CUDA required (pure arithmetic).
"""

from __future__ import annotations

import math

import torch

# Structural per-apply all_to_all counts (pinned by tests/dist/test_traffic.py). The ckpt value is
# a measured default for the current torch selective-recompute behavior, NOT a law -- the timing
# harness overrides it with the in-situ count.
COLLECTIVES_PER_APPLY = {"fwd": 6, "fwd_adj": 12, "fwd_adj_ckpt": 15}


def elements_per_collective(
    n: tuple[int, int, int], world_size: int, use_rfft: bool = False
) -> int:
    """Complex elements each rank holds in one transpose all_to_all (zero-padded 2x in x, y).

    The z-distributed operand is ``[2nx, 2ny, nz/P]`` complex (dist/demag.py::_to_xy); the
    xy-distributed operand ``[(2nx*2ny)/P, nz]`` has the same element count, so every collective
    in the forward and adjoint moves the same volume. ``use_rfft`` accounts for the packed
    Hermitian plane ``[2nx, ny+1, nz/P]`` (decisions 2026-07-06) -- the ~2x wire halving.
    """
    nx, ny, nz = n
    if nz % world_size:
        raise ValueError("nz must be divisible by world_size")
    plane = (2 * nx) * (ny + 1) if use_rfft else (2 * nx) * (2 * ny)
    return plane * (nz // world_size)


def wire_bytes_per_rank(
    n: tuple[int, int, int], world_size: int, dtype: torch.dtype, use_rfft: bool = False
) -> int:
    """Bytes each rank puts on the wire in ONE all_to_all (the (P-1)/P off-rank fraction).

    ``dtype`` is the REAL dtype of m (float32/float64); the transposed spectral tensor is the
    matching complex dtype (2x the real itemsize; the gloo real-view repack is byte-identical).
    """
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(f"expected a real float dtype, got {dtype}")
    complex_itemsize = 2 * dtype.itemsize
    local = elements_per_collective(n, world_size, use_rfft) * complex_itemsize
    return local * (world_size - 1) // world_size


def comm_time_lower_bound(
    n: tuple[int, int, int],
    world_size: int,
    dtype: torch.dtype,
    n_collectives: int,
    link_bw: float,
    alpha: float = 0.0,
    use_rfft: bool = False,
) -> float:
    """Seconds the collectives alone must take: ``n_coll * ((P-1)*alpha + bytes/link_bw)``.

    ``alpha`` is the measured per-message latency (s) -- each rank exchanges with P-1 peers per
    all_to_all; ``alpha=0`` is the bytes-only bound. Assumes a full-bisection intra-node fabric
    (NVLink), i.e. the bottleneck is each rank's own ejection bandwidth -- state this assumption
    wherever the bound is reported, alongside the hardware, precision, and scaling type.
    """
    if link_bw <= 0:
        raise ValueError("link_bw must be positive")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    wire = wire_bytes_per_rank(n, world_size, dtype, use_rfft)
    return n_collectives * ((world_size - 1) * alpha + wire / link_bw)


def fit_alpha_beta(payload_bytes: list[int], times_s: list[float]) -> tuple[float, float]:
    """Least-squares fit of ``t = alpha + b/beta`` to a measured payload sweep.

    Returns ``(alpha, beta)`` = (per-message latency in s, bandwidth in B/s). Needs at least two
    distinct payload sizes; noise can drive the fitted alpha slightly negative -- that is reported
    as-is (the bound rejects it), never clamped.
    """
    if len(payload_bytes) != len(times_s):
        raise ValueError("payload_bytes and times_s must have the same length")
    m = len(payload_bytes)
    if m < 2 or len(set(payload_bytes)) < 2:
        raise ValueError("need at least two distinct payload sizes to fit (alpha, beta)")
    sb = float(sum(payload_bytes))
    st = float(sum(times_s))
    sbb = float(sum(b * b for b in payload_bytes))
    sbt = float(sum(b * t for b, t in zip(payload_bytes, times_s, strict=True)))
    slope = (m * sbt - sb * st) / (m * sbb - sb * sb)  # dt/dbyte = 1/beta
    if slope <= 0:
        raise ValueError("fitted bandwidth is non-positive; the sweep data is not alpha-beta-like")
    alpha = (st - slope * sb) / m
    return alpha, 1.0 / slope


def roofline_fraction(measured_time: float, lower_bound: float) -> float:
    """Achieved fraction of the comm roofline (1.0 = wall time equals the comm lower bound)."""
    if measured_time <= 0:
        raise ValueError("measured_time must be positive")
    return lower_bound / measured_time


def fft_throughput_roofline(
    n_total: int, world_size: int, link_bw: float, itemsize: int, n_reshapes: int
) -> float:
    """heFFTe-style comm-bound FFT throughput ceiling Psi = 5*log2(N)*P*B/(alpha*r), in FLOP/s.

    ``n_total`` = total complex points of the (padded) 3D FFT, ``itemsize`` = alpha bytes per
    element, ``n_reshapes`` = r global transposes per FFT. Derivation: FFT work ~ 5*N*log2(N)
    flops; comm-bound time ~ r*alpha*N/(P*B); their ratio is Psi. Used only for the cross-code
    regime map (a tie claim vs production FFTs) -- never as a win metric.
    """
    if min(n_total, world_size, itemsize, n_reshapes) <= 0 or link_bw <= 0:
        raise ValueError("all arguments must be positive")
    return 5.0 * math.log2(n_total) * world_size * link_bw / (itemsize * n_reshapes)
