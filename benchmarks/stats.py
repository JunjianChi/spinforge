"""Shared timing statistics: one definition of mean/std/CI + raw samples for every meter.

The distributed harness and the single-GPU baseline must summarize walls identically, or their
numbers are not comparable -- so the summary lives in exactly one place.
"""

from __future__ import annotations

import math
from typing import Any

from scipy import stats as _scipy_stats


def timing_stats(wall_s: list[float]) -> dict[str, Any]:
    """Summarize wall-clock samples (seconds in, milliseconds out) with a 95% t-CI.

    Raw samples are always included (pyperf discipline: statistics stay derivable forever).
    """
    n = len(wall_s)
    mean = sum(wall_s) / n
    std = math.sqrt(sum((t - mean) ** 2 for t in wall_s) / (n - 1)) if n > 1 else 0.0
    ci95 = float(_scipy_stats.t.ppf(0.975, n - 1)) * std / math.sqrt(n) if n > 1 else float("nan")
    return {
        "wall_ms_mean": 1e3 * mean,
        "wall_ms_std": 1e3 * std,
        "wall_ms_ci95": 1e3 * ci95,
        "wall_ms_all": [1e3 * t for t in wall_s],
        "reps": n,
    }
