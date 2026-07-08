"""Autograd-differentiable wrappers around torch.distributed collectives.

The raw collectives are not differentiable. Each wrapper supplies the correct adjoint so the
reverse-mode gradient survives the rank boundary. ``all_to_all`` is the building block of the FFT
transpose; its adjoint is another ``all_to_all`` (the (rank, chunk) index transpose is an involution
for equal chunks, hence self-adjoint).

Two seam styles, one contract:

- **Blocking** (``all_to_all`` / ``all_to_all_complex`` / ``..._mixed``): call = collective done.
- **Split-phase** (``all_to_all_start`` / ``all_to_all_finish`` + the ``A2ABackend`` objects): the
  collective is LAUNCHED at start and COMPLETED at finish, so local compute can run in the window
  between them -- the compute/comm overlap lever. In the backward the roles invert (the DDP
  launch-early/await-last pattern): the finish node's backward launches the adjoint all_to_all and
  the start node's backward completes it, giving the adjoint the same overlap window for free.
  Correctness never depends on asynchrony: on gloo the wait simply blocks.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch
import torch.distributed as dist

# Count every physical all_to_all (forward, adjoint, and checkpoint-recompute alike) so the
# collective traffic the comm-optimization headline targets is measurable (test_traffic).
_A2A_CALLS = 0


def a2a_calls() -> int:
    """All_to_all collectives issued since the last reset (one unit of transpose traffic)."""
    return _A2A_CALLS


def reset_a2a_calls() -> None:
    global _A2A_CALLS
    _A2A_CALLS = 0


# Debug guard: all_to_all permutes (rank, chunk) blocks, so the GLOBAL payload sum and L1 are
# invariants of every physical collective. Checking them catches cross-rank bookkeeping bugs
# (dropped/duplicated/sign-flipped chunks) that per-rank scalar oracles cannot see. Off by default:
# each check costs one tiny extra all_reduce per collective.
_VERIFY_CONSERVATION = False


def set_verify_conservation(on: bool) -> None:
    """Toggle the per-collective conservation assertion (debug mode; all ranks must agree)."""
    global _VERIFY_CONSERVATION
    _VERIFY_CONSERVATION = on


def _assert_conserved(sent: torch.Tensor, recv: torch.Tensor, leg: str) -> None:
    s = sent.detach()
    r = recv.detach()
    stats = torch.stack(
        [
            s.sum(dtype=torch.float64),
            s.abs().sum(dtype=torch.float64),
            r.sum(dtype=torch.float64),
            r.abs().sum(dtype=torch.float64),
        ]
    )
    dist.all_reduce(stats)
    s_sum, s_l1, r_sum, r_l1 = stats.tolist()
    # scale by the global L1 (sums of random payloads cancel toward 0, so a relative check on the
    # sum alone is ill-conditioned); f64 accumulation keeps roundoff far below the 1e-6 gate
    tol = 1e-6 * max(s_l1, r_l1)
    if abs(s_sum - r_sum) > tol or abs(s_l1 - r_l1) > tol:
        raise RuntimeError(
            f"all_to_all {leg}: global payload not conserved "
            f"(sent sum={s_sum:.17g} l1={s_l1:.17g}, recv sum={r_sum:.17g} l1={r_l1:.17g})"
        )


def _check_a2a_shape(x: torch.Tensor) -> None:
    world = dist.get_world_size()
    if x.dim() < 1 or x.size(0) % world:
        raise ValueError(
            f"all_to_all input dim0 must be divisible by the world size ({world}); "
            f"got shape {tuple(x.shape)}"
        )


class _AllToAllSingle(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, x: torch.Tensor) -> torch.Tensor:
        global _A2A_CALLS
        _A2A_CALLS += 1
        # allocate the output FROM the contiguous send buffer: empty_like preserves strides, so a
        # non-contiguous-but-dense input would otherwise scramble the gathered (flat) data.
        _check_a2a_shape(x)
        xc = x.contiguous()
        out = torch.empty_like(xc)
        dist.all_to_all_single(out, xc)
        if _VERIFY_CONSERVATION:
            _assert_conserved(xc, out, "forward")
        return out

    @staticmethod
    def backward(ctx: object, grad: torch.Tensor) -> torch.Tensor:
        # adjoint of all_to_all is all_to_all
        global _A2A_CALLS
        _A2A_CALLS += 1
        gc = grad.contiguous()
        grad_in = torch.empty_like(gc)
        dist.all_to_all_single(grad_in, gc)
        if _VERIFY_CONSERVATION:
            _assert_conserved(gc, grad_in, "adjoint")
        return grad_in


def all_to_all(x: torch.Tensor) -> torch.Tensor:
    """Differentiable all-to-all over dim 0 (split into ``world_size`` equal chunks)."""
    return _AllToAllSingle.apply(x)


def all_to_all_complex(x: torch.Tensor) -> torch.Tensor:
    """Differentiable all-to-all for a complex tensor (gloo has no complex support -> real view)."""
    return torch.view_as_complex(all_to_all(torch.view_as_real(x).contiguous()))


def all_to_all_complex_mixed(
    x: torch.Tensor, wire_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Differentiable all-to-all carrying a REDUCED-PRECISION wire payload (a comm lever).

    Cast to ``wire_dtype`` before the transpose, transpose, cast back -- so the collective moves
    half
    the bytes (f32 wire vs f64) while the caller's compute dtype is preserved for the surrounding
    FFT/spectral arithmetic. The casts are differentiable straight-through ops, so autograd casts
    the
    cotangent down to ``wire_dtype`` before the adjoint all_to_all too: the wire carries the reduced
    payload in BOTH the forward and backward transpose. Drop-in for ``all_to_all_complex`` behind
    the
    ``Collective`` swappable-backend seam.

    Not exact: each pass is a single f64<->wire round-trip, so the output (and gradient) carry a
    relative error bounded by the wire type's unit roundoff (~6e-8 for f32). Demag is linear and
    self-adjoint (kappa~1), so the error does not amplify and the gradient inherits the same bound
    --
    gated by a reduced-tolerance gradcheck + a measured error-vs-exact number, never claimed
    exact. The bandwidth win (half the NVLink bytes) is measured on real multi-GPU hardware.
    """
    real = torch.view_as_real(x)  # [..., 2] in the caller's dtype (e.g. f64)
    wire = real.to(wire_dtype).contiguous()  # downcast the payload -> half the bytes for f32
    out = all_to_all(wire).to(real.dtype)  # transpose on the reduced wire, then upcast back
    return torch.view_as_complex(out.contiguous())


class A2ASlot:
    """In-flight state of ONE split-phase all_to_all and its adjoint.

    The forward pins the send buffer until the wait; the adjoint launch (in the finish node's
    backward) pins the cotangent send buffer until the start node's backward completes -- without
    those refs the collective could read a garbage-collected buffer mid-flight.
    """

    __slots__ = ("fwd_work", "fwd_src", "fwd_waited", "bwd_work", "bwd_src")

    def __init__(self) -> None:
        self.fwd_work: object | None = None
        self.fwd_src: torch.Tensor | None = None
        self.fwd_waited = False
        self.bwd_work: object | None = None
        self.bwd_src: torch.Tensor | None = None


class _A2AStart(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, x: torch.Tensor, slot: A2ASlot) -> torch.Tensor:  # type: ignore[override]
        global _A2A_CALLS
        _A2A_CALLS += 1
        _check_a2a_shape(x)
        xc = x.contiguous()
        out = torch.empty_like(xc)
        slot.fwd_work = dist.all_to_all_single(out, xc, async_op=True)
        slot.fwd_src = xc  # pin the send buffer until the wait
        ctx.slot = slot  # type: ignore[attr-defined]
        return out  # NOT valid until _A2AWait; the start's output must have no other consumer

    @staticmethod
    def backward(ctx: object, grad: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        # _A2AWait.backward launched the adjoint collective; complete it here. The autograd work
        # scheduled between the two nodes is the backward overlap window.
        slot: A2ASlot = ctx.slot  # type: ignore[attr-defined]
        assert slot.bwd_work is not None, "adjoint collective was never launched (graph misuse)"
        slot.bwd_work.wait()  # type: ignore[attr-defined]
        if _VERIFY_CONSERVATION and slot.bwd_src is not None:
            _assert_conserved(slot.bwd_src, grad, "adjoint (split-phase)")
        slot.bwd_work, slot.bwd_src = None, None  # release the cotangent send buffer
        return grad, None

    @classmethod
    def apply_(cls, x: torch.Tensor, slot: A2ASlot) -> torch.Tensor:
        return cls.apply(x, slot)  # typed helper


class _A2AWait(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, out: torch.Tensor, slot: A2ASlot) -> torch.Tensor:  # type: ignore[override]
        if slot.fwd_waited:
            raise RuntimeError("split-phase all_to_all waited twice")
        assert slot.fwd_work is not None, "wait without a matching start"
        slot.fwd_work.wait()  # type: ignore[attr-defined]
        if _VERIFY_CONSERVATION and slot.fwd_src is not None:
            # `out` is the start node's receive buffer (its output has no other consumer)
            _assert_conserved(slot.fwd_src, out, "forward (split-phase)")
        slot.fwd_work, slot.fwd_src, slot.fwd_waited = None, None, True
        ctx.slot = slot  # type: ignore[attr-defined]
        return out.view_as(out)  # alias output (torch.distributed.nn pattern; pinned by tests)

    @staticmethod
    def backward(ctx: object, grad: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        global _A2A_CALLS
        _A2A_CALLS += 1
        slot: A2ASlot = ctx.slot  # type: ignore[attr-defined]
        gc = grad.contiguous()
        gin = torch.empty_like(gc)
        slot.bwd_work = dist.all_to_all_single(gin, gc, async_op=True)
        slot.bwd_src = gc  # pin until _A2AStart.backward completes
        return gin, None  # NOT valid until _A2AStart.backward


def all_to_all_start(x: torch.Tensor) -> tuple[torch.Tensor, A2ASlot]:
    """Launch a differentiable all-to-all over dim 0; complete it with ``all_to_all_finish``.

    The returned placeholder tensor is NOT valid until the finish call and must have no consumer
    other than that call.
    """
    slot = A2ASlot()
    return _A2AStart.apply_(x, slot), slot


def all_to_all_finish(placeholder: torch.Tensor, slot: A2ASlot) -> torch.Tensor:
    """Complete a split-phase all-to-all (exactly once per start)."""
    return _A2AWait.apply(placeholder, slot)


class A2AHandle:
    """One in-flight complex transpose; ``wait()`` exactly once returns the received tensor."""

    __slots__ = ("_ph", "_slot")

    def __init__(self, placeholder: torch.Tensor, slot: A2ASlot) -> None:
        self._ph, self._slot = placeholder, slot

    def wait(self) -> torch.Tensor:
        return torch.view_as_complex(all_to_all_finish(self._ph, self._slot))


class _DoneHandle:
    """Already-completed transpose (from a blocking backend); waits are trivial but still
    guarded."""

    __slots__ = ("_y",)

    def __init__(self, y: torch.Tensor) -> None:
        self._y: torch.Tensor | None = y

    def wait(self) -> torch.Tensor:
        if self._y is None:
            raise RuntimeError("split-phase all_to_all waited twice")
        y, self._y = self._y, None
        return y


class TransposeWaitable(Protocol):
    def wait(self) -> torch.Tensor: ...


class A2ABackend(Protocol):
    """The injectable transpose seam: launch one complex all_to_all, differentiably."""

    def start(self, x: torch.Tensor) -> TransposeWaitable: ...


class TorchAsyncA2A:
    """Default transpose backend: nonblocking torch.distributed all_to_all behind the
    differentiable split-phase pair. ``start(x).wait()`` back-to-back IS the naive schedule;
    scheduling work between them is the overlap row."""

    def start(self, x: torch.Tensor) -> A2AHandle:
        ph, slot = all_to_all_start(torch.view_as_real(x).contiguous())
        return A2AHandle(ph, slot)


class SyncCollectiveA2A:
    """Adapter: any blocking differentiable Collective (counting/timing instruments, the
    mixed-precision wire, the native MPI op) as a backend. The collective runs eagerly at start,
    so schedules degrade to sequential semantics -- correct always, overlap never."""

    __slots__ = ("collective",)

    def __init__(self, collective: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.collective = collective

    def start(self, x: torch.Tensor) -> _DoneHandle:
        return _DoneHandle(self.collective(x))


class _AllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    @staticmethod
    def backward(ctx: object, grad: torch.Tensor) -> torch.Tensor:
        # every rank holds the same summed value; the cotangent passes through unchanged
        return grad


def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """Differentiable SUM all-reduce (each rank ends with the summed value)."""
    return _AllReduceSum.apply(x)
