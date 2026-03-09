import torch
from torch import Tensor

__all__ = [""]

# wrapper for calling convenience
def svdmatmul(a: Tensor, b: Tensor) -> Tensor:
    """Performs a * b with svd kernel"""
    return torch.ops.ppopp.svdmutmul.default(a, b)

# torch.compile support: do we really need this?
@torch.library.register_fake("ppopp::svdmatmul")
def _(a, b):
    torch._check(a.dim() == 2)
    torch._check(b.dim() == 2)
    torch._check(a.shape[1] == b.shape[0])
    output_shape = [a.shape[0], b.shape[1]] 

    return torch.empty(output_shape, dtype=a.dtype, device=a.device)

# training support: backward
# Not sure if middle tensors need to be saved, 
# as they only ought to accelerate the forward, not LoRA-like params
def _setup_context(ctx, inputs, output):
    """Saving tensors for backward"""
    A, B = inputs
    saved_A, saved_B = None, None
    if ctx.needs_input_grad[0]:
        saved_B = B 
    if ctx.needs_input_grad[1]:
        saved_A = A 
    
    ctx.save_for_backward(saved_A, saved_B)

def _backward(ctx, grad):
    """Calculate gradients from saved tensors"""
    A, B = ctx.saved_tensors
    
    grad_A, grad_B = None, None
    
    if ctx.needs_input_grad[0]:
        grad_A = torch.matmul(grad, B.T)
    
    if ctx.needs_input_grad[1]:
        grad_B = torch.matmul(A.T, grad)
    
    return grad_A, grad_B

torch.library.register_autograd(
    "ppopp::svdmatmul", _backward, setup_context=_setup_context)

# mutable operators: not needed for now
# mutable refers to in-place modification
