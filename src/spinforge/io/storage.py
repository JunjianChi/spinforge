"""One source of truth for output paths: five decoupled buckets.

data / checkpoints / runs / scratch are gitignored; only results/ is committed. Every
experiment routes its writes through here instead of hand-building paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_BUCKETS = ("data", "checkpoints", "runs", "results", "scratch")


@dataclass(frozen=True)
class StorageLayout:
    """Bucketed output paths rooted at ``root`` (``Path.cwd()`` by default via ``at``)."""

    root: Path

    @classmethod
    def at(cls, root: str | Path | None = None) -> StorageLayout:
        return cls(Path(root) if root is not None else Path.cwd())

    def bucket(self, name: str) -> Path:
        """The path of one bucket, created on demand."""
        if name not in _BUCKETS:
            raise ValueError(f"unknown bucket {name!r}; expected one of {_BUCKETS}")
        p = self.root / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def run_dir(self, name: str) -> Path:
        """A directory under runs/ for one experiment run (its provenance + outputs land here)."""
        p = self.bucket("runs") / name
        p.mkdir(parents=True, exist_ok=True)
        return p
