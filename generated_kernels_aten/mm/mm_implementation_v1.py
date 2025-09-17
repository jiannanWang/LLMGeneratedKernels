import torch
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mm_kernel_impl(*args, **kwargs):
    """
    Custom mm implementation that handles all torch.mm overloads.

    Supported overloads:
    - mm.default(Tensor self, Tensor mat2) -> Tensor
    - mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor
    - mm.dtype(Tensor self, Tensor mat2, *, ScalarType? dtype) -> Tensor
    - mm.dtype_out(Tensor self, Tensor mat2, *, ScalarType? dtype, Tensor(a!) out) -> Tensor

    Matrix multiplication: result[i,j] = sum_k(self[i,k] * mat2[k,j])
    """
    logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Using custom mm implementation")

    # Handle arguments
    out = kwargs.get('out', None)
    dtype = kwargs.get('dtype', None)

    if len(args) == 2:
        # Basic case: mm(tensor1, tensor2)
        self, mat2 = args
    elif len(args) == 3:
        # Case with output: mm(tensor1, tensor2, out)
        self, mat2, out = args
    else:
        raise ValueError(f"Unexpected number of arguments: {len(args)}")

    # Validate input dimensions
    if self.dim() != 2 or mat2.dim() != 2:
        raise RuntimeError("mm: inputs must be 2D tensors")

    if self.size(1) != mat2.size(0):
        raise RuntimeError(f"mm: size mismatch, self: {self.shape}, mat2: {mat2.shape}")

    # Get dimensions
    m, k = self.shape
    k2, n = mat2.shape

    # Implement matrix multiplication using torch.sum and broadcasting
    # We'll use einsum-like logic: result[i,j] = sum_k(self[i,k] * mat2[k,j])

    # Expand dimensions for broadcasting
    # self: (m, k) -> (m, k, 1)
    # mat2: (k, n) -> (1, k, n)
    self_expanded = self.unsqueeze(2)  # (m, k, 1)
    mat2_expanded = mat2.unsqueeze(0)  # (1, k, n)

    # Element-wise multiplication with broadcasting: (m, k, 1) * (1, k, n) -> (m, k, n)
    product = self_expanded.__mul__(mat2_expanded)

    # Sum over the k dimension to get (m, n)
    result = torch.sum(product, dim=1)

    # Handle dtype conversion if requested
    if dtype is not None and result.dtype != dtype:
        result = result.to(dtype)

    if out is not None:
        # Copy result to output tensor
        out.copy_(result)
        return out

    return result