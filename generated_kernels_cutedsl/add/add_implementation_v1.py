import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def add_tensor_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    # Map thread index to logical index of input tensor
    total_elements = gA.shape[0]
    
    # Bounds checking
    if thread_idx < total_elements:

        # Map logical index to physical address via tensor layout
        a_val = gA[thread_idx]
        b_val = gB[thread_idx]

        # Perform element-wise addition
        gC[thread_idx] = a_val + b_val

@cute.kernel
def add_scalar_kernel(
    gA: cute.Tensor,
    gC: cute.Tensor,
    scalar_val,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    # Map thread index to logical index of input tensor
    total_elements = gA.shape[0]
    
    # Bounds checking
    if thread_idx < total_elements:

        # Map logical index to physical address via tensor layout
        a_val = gA[thread_idx]

        # Perform element-wise addition with scalar
        gC[thread_idx] = a_val + scalar_val

@cute.jit
def add_tensor_kernel_launch(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor
):
    num_threads_per_block = 1024

    total_elements = mA.shape[0]
    num_blocks = (total_elements + num_threads_per_block - 1) // num_threads_per_block
    
    kernel = add_tensor_kernel(mA, mB, mC)
    kernel.launch(grid=(num_blocks, 1, 1),
                  block=(num_threads_per_block, 1, 1))

@cute.jit
def add_scalar_kernel_launch(
    mA: cute.Tensor,
    mC: cute.Tensor,
    scalar_val
):
    num_threads_per_block = 1024

    total_elements = mA.shape[0]
    num_blocks = (total_elements + num_threads_per_block - 1) // num_threads_per_block
    
    kernel = add_scalar_kernel(mA, mC, scalar_val)
    kernel.launch(grid=(num_blocks, 1, 1),
                  block=(num_threads_per_block, 1, 1))

@cute.kernel
def add_cutedsl_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    # Map thread index to logical index of input tensor
    total_elements = gA.shape[0]
    
    # Bounds checking
    if thread_idx < total_elements:

        # Map logical index to physical address via tensor layout
        a_val = gA[thread_idx]
        b_val = gB[thread_idx]

        # Perform element-wise addition
        gC[thread_idx] = a_val + b_val

def add_kernel_impl(*args, **kwargs):
    
    # Handle both positional and keyword arguments
    if len(args) >= 2:
        input_tensor = args[0]
        other = args[1]
    elif len(args) == 1 and 'other' in kwargs:
        input_tensor = args[0]
        other = kwargs['other']
    elif 'input' in kwargs and 'other' in kwargs:
        input_tensor = kwargs['input']
        other = kwargs['other']
    else:
        raise ValueError("add requires 'input' and 'other' arguments")
    
    if torch.is_tensor(other):
        input_tensor, other = torch.broadcast_tensors(input_tensor, other)
    
    if 'alpha' in kwargs:
        alpha = kwargs['alpha']
        other = other * alpha
    
    # Remember original device
    original_device = input_tensor.device

    # Flatten all tensors and save their shapes
    original_shape = input_tensor.shape
    input_tensor = input_tensor.flatten()
    if torch.is_tensor(other):
        other = other.flatten()
    
    # Move to GPU if needed
    if not input_tensor.is_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        input_tensor = input_tensor.cuda()
    
    # Check if other is a tensor or scalar
    if torch.is_tensor(other):
        # Tensor + Tensor case
        if not other.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            other = other.cuda()
        
        output = torch.empty_like(input_tensor)
        a_ = from_dlpack(input_tensor)
        b_ = from_dlpack(other)
        c_ = from_dlpack(output)

        add_tensor_kernel_launch_ = cute.compile(add_tensor_kernel_launch, a_, b_, c_)
        add_tensor_kernel_launch_(a_, b_, c_)
    else:
        # Tensor + Scalar case
        # Convert scalar to Python float
        if hasattr(other, 'item'):
            scalar_val = other.item()
        else:
            scalar_val = other
        
        output = torch.empty_like(input_tensor)
        a_ = from_dlpack(input_tensor)
        c_ = from_dlpack(output)

        add_scalar_kernel_launch_ = cute.compile(add_scalar_kernel_launch, a_, c_, scalar_val)
        add_scalar_kernel_launch_(a_, c_, scalar_val)
    
    # Move result back to original device
    if original_device != output.device:
        output = output.to(original_device)
    
    output = output.reshape(original_shape)
    
    return output