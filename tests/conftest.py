"""Shared fixtures for the suite."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def artifacts_dir() -> Path:
    """Where ``render_on_failure`` renders land: gitignored, uploaded by CI on a red run."""
    return Path(__file__).parent / "test_artifacts"
