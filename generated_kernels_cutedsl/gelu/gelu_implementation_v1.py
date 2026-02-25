import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def gelu_kernel(input_ptr, output_ptr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    thread_idx = bidx * bdim + tidx

    x = input_ptr[thread_idx]
    # Fast GELU approximation
    gelu_x = 0.5 * x * (1.0 + cute.math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    output_ptr[thread_idx] = gelu_x

@cute.jit
def gelu_kernel_launch(
    mA: cute.Tensor,
    mC: cute.Tensor
):
    num_threads_per_block = 1024

    total_elements = mA.shape[0]
    num_blocks = (total_elements + num_threads_per_block - 1) // num_threads_per_block
    
    kernel = gelu_kernel(mA, mC)
    kernel.launch(grid=(num_blocks, 1, 1),
                  block=(num_threads_per_block, 1, 1))



def gelu_kernel_impl(*args, **kwargs):
    
    # Handle both positional and keyword arguments
    if len(args) >= 1:
        input_tensor = args[0]
    elif 'input' in kwargs:
        input_tensor = kwargs['input']
    else:
        raise ValueError("gelu requires 'input' arguments")

    
    # Remember original device
    original_device = input_tensor.device

    # Flatten all tensors and save their shapes
    original_shape = input_tensor.shape
    input_tensor = input_tensor.flatten()

    original_dtype = input_tensor.dtype
    input_tensor = input_tensor.float()
    
    # Move to GPU if needed
    if not input_tensor.is_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        input_tensor = input_tensor.cuda()
    
    output = torch.empty_like(input_tensor)
    a_ = from_dlpack(input_tensor)
    c_ = from_dlpack(output)

    gelu_kernel_launch_ = cute.compile(gelu_kernel_launch, a_, c_)
    gelu_kernel_launch_(a_, c_)
    
    # Move result back to original device
    if original_device != output.device:
        output = output.to(original_device)
    
    if original_dtype != output.dtype:
        output = output.to(original_dtype)
    
    output = output.reshape(original_shape)
    
    return output