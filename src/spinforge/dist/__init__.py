"""Distributed layer: differentiable collectives and the slab-decomposed demag (the project's core).

Gradients must flow across rank boundaries. Raw ``torch.distributed`` collectives are NOT
autograd-differentiable (they break the graph at the rank boundary -- the exact gap in
magnum.np.distributed); everything here wraps them in ``autograd.Function`` with the correct
adjoint.
"""
