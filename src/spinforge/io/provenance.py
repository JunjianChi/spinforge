"""Reproducibility: one seed drives every RNG; each run records seed+config+git.

A result with no captured provenance does not count -- ``capture_provenance`` pins config, seed, and
the exact git commit + working diff into the run's directory so a committed number is reproducible.
"""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch from one value."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# run git against the source tree (this file's dir), not the cwd: a run directory is often not
# itself a checkout (the AutoDL case), and binding the commit is the whole point of provenance
_SRC_TREE = Path(__file__).resolve().parent


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(_SRC_TREE), *args], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""  # not a git checkout / git absent -> provenance still records config + seed


def capture_provenance(run_dir: Path, config: dict[str, Any], seed: int) -> Path:
    """Write provenance.json (seed, config, git commit) + git.diff into ``run_dir``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {"seed": seed, "config": config, "git_commit": _git("rev-parse", "HEAD")}
    (run_dir / "provenance.json").write_text(json.dumps(record, indent=2, default=str))
    (run_dir / "git.diff").write_text(_git("diff"))
    return run_dir
