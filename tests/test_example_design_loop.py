"""The committed usage example must actually work -- CI runs it at toy scale.

A README/examples snippet that silently rots is worse than none: this pins the example's public
contract (importable, runs on CPU in seconds, and the design loss genuinely DECREASES, i.e. the
gradient-based design loop does its job).
"""

from __future__ import annotations

from examples.design_loop import run_design_loop


def test_design_loop_reduces_the_miss() -> None:
    result = run_design_loop(n=24, relax_steps=60, design_iters=10, seed=0)
    assert result["final_miss_nm"] < 0.7 * result["baseline_miss_nm"], result
    # the moved object must still be a skyrmion, not an optimizer artifact (the cheap render-free
    # proxy)
    assert abs(result["final_Q"] + 1.0) < 0.3, result
