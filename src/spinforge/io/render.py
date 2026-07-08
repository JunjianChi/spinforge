"""Visual verification: render the m_z heatmap + cross-section and look; never trust a scalar alone.

Scalars alone can be fabricated by a detection artifact a scalar check would miss, so the render
is the evidence, not decoration. ``render_on_failure`` automates the discipline for tests: a failing
physics-oracle assertion leaves its texture picture behind (CI uploads it as an artifact), a green
run writes nothing. matplotlib is the optional ``viz`` extra -- without it rendering degrades to a
loud warning instead of an ImportError, mirroring the experiments.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def render_mz(m: torch.Tensor, path: Path, title: str = "") -> bool:
    """Render ``m_z`` of a ``[nx, ny, nz, 3]`` texture (mid-z heatmap + mid-y cross-section).

    Returns True if the picture was written, False (with a warning) when matplotlib is missing.
    """
    try:
        import matplotlib
    except ImportError:
        logger.warning("matplotlib missing: NO render for %s -> visual evidence lost", path)
        return False
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mz = m.detach()[..., 2].cpu()
    _, ny, nz = mz.shape
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    im0 = ax[0].imshow(mz[:, :, nz // 2].T.numpy(), origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
    ax[0].set_title(f"m_z, z={nz // 2}")
    im1 = ax[1].imshow(
        # aspect="auto": a thin film (nz << nx) must not collapse to a one-pixel strip
        mz[:, ny // 2, :].T.numpy(),
        origin="lower",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )
    ax[1].set_title(f"m_z cross-section, y={ny // 2}")
    fig.colorbar(im0, ax=ax[0])
    fig.colorbar(im1, ax=ax[1])
    if title:
        fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


@contextmanager
def render_on_failure(m: torch.Tensor, name: str, out_dir: Path) -> Iterator[None]:
    """Render ``m`` to ``out_dir/<name>.png`` if the guarded block raises, then re-raise.

    Wrap the physics-oracle assertions of a test in this so a red run ships its own visual
    evidence; a green run writes nothing.
    """
    try:
        yield
    except BaseException as err:
        render_mz(m, out_dir / f"{name}.png", title=f"{name} (FAILED: {type(err).__name__})")
        raise
