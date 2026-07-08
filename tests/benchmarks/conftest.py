"""Shared fixtures for the benchmark tests."""

from __future__ import annotations

import json
from pathlib import Path


def make_summary_run(root: Path, name: str, world: int, wall: float, **flags: object) -> None:
    """Write a minimal benchmark summary.json a plot/collect test can consume."""
    d = root / name
    d.mkdir(parents=True)
    config: dict = {
        "world_size": world,
        "schedule": "sequential",
        "use_rfft": False,
        "mixed_wire": False,
        "uninstrumented": False,
        "perturb": "none",
        "n": [8, 8, 8],
    }
    config.update(flags)
    (d / "summary.json").write_text(
        json.dumps(
            {"config": config, "results": {"fwd_adj": {"wall_ms_mean": wall, "wall_ms_ci95": 0.1}}}
        )
    )
