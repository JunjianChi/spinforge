"""The failure-render helper: a failing physics assertion leaves an m_z picture behind.

Scalars alone can be fabricated by a detection misfire a scalar check would not catch; the render
is the evidence. ``render_mz`` draws the heatmap + cross-section; ``render_on_failure`` renders only
when
the guarded block raises, then re-raises -- so every red oracle test ships its own visual evidence
(uploaded by CI as an artifact) and a green run writes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from spinforge.io.render import render_mz, render_on_failure


def _texture() -> torch.Tensor:
    m = torch.zeros(8, 6, 4, 3, dtype=torch.float64)
    m[..., 2] = 1.0
    m[2:5, 2:4, :, 2] = -1.0  # a localized down-core so the render has structure
    return m


def test_render_mz_writes_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    out = tmp_path / "sub" / "tex.png"  # parent dirs are created
    assert render_mz(_texture(), out, title="tex") is True
    assert out.is_file() and out.stat().st_size > 0


def test_render_mz_degrades_without_matplotlib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # None in sys.modules makes ``import matplotlib`` raise ImportError -> warn + False, no crash
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    out = tmp_path / "tex.png"
    assert render_mz(_texture(), out) is False
    assert not out.exists()


def test_render_on_failure_renders_and_reraises(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    with (
        pytest.raises(AssertionError, match="oracle broke"),
        render_on_failure(_texture(), "broken_oracle", tmp_path),
    ):
        raise AssertionError("oracle broke")
    assert (tmp_path / "broken_oracle.png").is_file()


def test_render_on_success_writes_nothing(tmp_path: Path) -> None:
    with render_on_failure(_texture(), "fine_oracle", tmp_path):
        pass
    assert list(tmp_path.iterdir()) == []
