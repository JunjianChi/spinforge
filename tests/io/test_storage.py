"""StorageLayout gives the five decoupled buckets and a per-run directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from spinforge.io import StorageLayout


def test_buckets_created_on_demand(tmp_path: Path) -> None:
    layout = StorageLayout.at(tmp_path)
    for name in ("data", "checkpoints", "runs", "results", "scratch"):
        p = layout.bucket(name)
        assert p == tmp_path / name
        assert p.is_dir()


def test_unknown_bucket_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown bucket"):
        StorageLayout.at(tmp_path).bucket("weights")


def test_run_dir_under_runs(tmp_path: Path) -> None:
    rd = StorageLayout.at(tmp_path).run_dir("skyrmion_hall_seed0")
    assert rd == tmp_path / "runs" / "skyrmion_hall_seed0"
    assert rd.is_dir()
