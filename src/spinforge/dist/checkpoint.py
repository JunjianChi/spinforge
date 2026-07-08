"""Cross-rank gradient checkpointing for the distributed demag: activation memory for recompute.

The distributed forward stores its FFT + transpose intermediates for the backward; for a problem
spanning many GPUs that per-rank activation set is itself a memory wall. Checkpointing runs the
forward WITHOUT saving them and re-runs it in the backward to regenerate what the adjoint needs.

The subtlety is distributed, not local: the recompute re-executes the transpose ``all_to_all``
collectives *inside the backward pass*, so every rank must reach them in lockstep or the job
deadlocks. Because all ranks drive an identical (all-reduced) loss and run symmetric code, their
backward recomputes stay coordinated. This mechanism carries the distributed adjoint into the
over-one-GPU regime; the single-GPU temporal analogue is in ``experiments/transport_design.py``.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from .demag import DistributedDemagField


def checkpointed_demag(
    field: DistributedDemagField, m_local: torch.Tensor, ms: float
) -> torch.Tensor:
    """Distributed demag whose forward activations are recomputed in the backward (memory-bounded).

    Numerically identical to ``field(m_local, ms)`` forward and gradient; only the activation memory
    differs. ``use_reentrant=False`` so the saved-tensor set is tracked correctly across the
    collective boundary.
    """
    return checkpoint(lambda mm: field(mm, ms), m_local, use_reentrant=False)
