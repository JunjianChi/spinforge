"""set_seed is deterministic (one seed drives every RNG) and capture_provenance records the run
(seed+config+git)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from spinforge.io import capture_provenance, set_seed


def test_set_seed_is_deterministic() -> None:
    set_seed(0)
    a = torch.randn(8)
    set_seed(0)
    b = torch.randn(8)
    torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)


def test_capture_provenance_writes_record(tmp_path: Path) -> None:
    rd = capture_provenance(tmp_path / "run", {"n": 8, "alpha": 0.3}, seed=7)
    record = json.loads((rd / "provenance.json").read_text())
    assert record["seed"] == 7
    assert record["config"] == {"n": 8, "alpha": 0.3}
    assert "git_commit" in record
    assert (rd / "git.diff").exists()


def test_capture_provenance_records_commit_from_a_non_checkout_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    """The git commit must bind even when the run dir (the cwd) is not a checkout --
    e.g. an AutoDL run directory outside the source checkout."""
    import subprocess

    monkeypatch.chdir(tmp_path)  # a fresh dir with no .git, like an AutoDL run directory
    rd = capture_provenance(tmp_path / "run", {"n": 8}, seed=1)
    record = json.loads((rd / "provenance.json").read_text())
    head = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head:  # only assert when the source tree really is a checkout
        assert record["git_commit"] == head
