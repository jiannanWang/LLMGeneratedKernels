"""
Kernel-Level Overhead Analysis

This script compares custom BackendBench kernels directly against PyTorch's eager mode
to identify the root causes of kernel execution overhead.

Key analysis dimensions:
1. Kernel launch overhead (Triton JIT compilation, grid setup)
2. Memory allocation overhead (torch.empty, .contiguous(), .cuda())
3. Data movement overhead (CPU<->GPU transfers)
4. Python wrapper overhead (argument parsing, validation)
5. Algorithmic overhead (suboptimal kernel implementation)

Usage:
    source ../.venv/bin/activate
    python analyze_kernel_overhead.py
"""

import os
import sys
import time
import functools
from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Any
from contextlib import contextmanager

import torch
import triton
import triton.testing
import numpy as np

# Add the generated kernels to path
sys.path.insert(0, "../generated_kernels_opinfo")


def benchmark_function(fn: Callable, args: tuple, kwargs: dict = None,
                       warmup: int = 25, rep: int = 100,
                       quantiles: Tuple[float, ...] = (0.5, 0.2, 0.8)) -> Dict[str, float]:
    """
    Benchmark a function using Triton's do_bench for accurate GPU kernel timing.

    Uses triton.testing.do_bench which provides:
    - Proper GPU synchronization
    - Statistical analysis with quantiles
    - Automatic warmup handling
    - More accurate timing for GPU operations

    Args:
        fn: Function to benchmark
        args: Positional arguments for fn
        kwargs: Keyword arguments for fn
        warmup: Number of warmup iterations (ms for do_bench)
        rep: Number of repetitions for measurement
        quantiles: Quantiles to compute (default: median, 20th, 80th percentile)

    Returns:
        Dictionary with timing statistics in milliseconds
    """
    if kwargs is None:
        kwargs = {}

    # Create a wrapper that calls the function with args/kwargs
    def bench_fn():
        return fn(*args, **kwargs)

    # Use Triton's do_bench for accurate GPU timing
    # do_bench returns time in milliseconds
    # quantiles returns (median, lower, upper) by default
    ms, min_ms, max_ms = triton.testing.do_bench(
        bench_fn,
        warmup=warmup,
        rep=rep,
        quantiles=quantiles,
        return_mode="median"  # Return median as primary result
    )

    # Also collect individual samples for std calculation
    # Run additional measurements for statistics
    times_ms = []
    for _ in range(rep):
        torch.cuda.synchronize()
        start = time.perf_counter()
        bench_fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)

    return {
        'mean': np.mean(times_ms),
        'std': np.std(times_ms),
        'min': min_ms,
        'max': max_ms,
        'median': ms,  # From do_bench
    }


def benchmark_triton_kernel(kernel_fn, grid, *args, warmup: int = 25, rep: int = 100,
                            **kernel_kwargs) -> Dict[str, float]:
    """
    Benchmark a Triton kernel directly using do_bench.

    This is more accurate for measuring raw kernel performance without
    Python wrapper overhead.

    Args:
        kernel_fn: The Triton kernel function (decorated with @triton.jit)
        grid: Grid configuration for the kernel
        *args: Arguments to pass to the kernel
        warmup: Warmup time in ms
        rep: Number of repetitions
        **kernel_kwargs: Keyword arguments for the kernel (e.g., BLOCK_SIZE)

    Returns:
        Dictionary with timing statistics in milliseconds
    """
    def bench_fn():
        kernel_fn[grid](*args, **kernel_kwargs)

    ms, min_ms, max_ms = triton.testing.do_bench(
        bench_fn,
        warmup=warmup,
        rep=rep,
        quantiles=(0.5, 0.2, 0.8),
        return_mode="median"
    )

    return {
        'mean': ms,  # For kernels, median is a good approximation
        'std': (max_ms - min_ms) / 4,  # Rough estimate
        'min': min_ms,
        'max': max_ms,
        'median': ms,
    }


@contextmanager
def timer(name: str):
    """Context manager for timing code blocks."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"  {name}: {(end - start) * 1000:.4f} ms")


class KernelAnalyzer:
    """Analyzes overhead of individual kernel implementations."""

    def __init__(self, kernel_folder: str = "../generated_kernels_opinfo"):
        self.kernel_folder = kernel_folder
        self.results = {}

    def analyze_view(self):
        """Analyze view operation overhead."""
        print("\n" + "=" * 80)
        print("ANALYZING: aten.view (4.96x slower)")
        print("=" * 80)

        # Import custom kernel
        from view.view_implementation_v1 import view_kernel_impl

        # Test cases with different sizes
        test_cases = [
            ("Small", (12, 64, 128), (12, 64 * 128)),
            ("Medium", (12, 256, 512), (12, 256 * 512)),
            ("Large", (12, 1024, 768), (12, 1024 * 768)),
        ]

        for name, input_shape, output_shape in test_cases:
            print(f"\n--- {name}: {input_shape} -> {output_shape} ---")
            x = torch.randn(input_shape, device='cuda', dtype=torch.bfloat16)

            # Native PyTorch view (essentially free - just metadata change)
            native_stats = benchmark_function(lambda t, s: t.view(s), (x, output_shape))

            # Custom kernel (launches actual GPU kernel)
            custom_stats = benchmark_function(view_kernel_impl, (x, output_shape))

            print(f"  Native PyTorch: {native_stats['mean']:.4f} ± {native_stats['std']:.4f} ms")
            print(f"  Custom Kernel:  {custom_stats['mean']:.4f} ± {custom_stats['std']:.4f} ms")
            print(f"  Slowdown:       {custom_stats['mean'] / native_stats['mean']:.2f}x")

            # Breakdown analysis
            print("\n  Overhead Breakdown:")
            self._analyze_view_overhead(x, output_shape, view_kernel_impl)

    def _analyze_view_overhead(self, x, output_shape, kernel_impl):
        """Break down the overhead sources for view operation using Triton do_bench."""

        # 1. Argument parsing overhead (CPU-bound, use manual timing)
        def arg_parse_fn():
            input_tensor = x
            shape = output_shape
            if isinstance(shape, (tuple, list)) and len(shape) == 1 and isinstance(shape[0], (tuple, list)):
                shape = shape[0]
            return shape

        # For CPU-bound operations, use manual timing
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(1000):
            arg_parse_fn()
        arg_parse_time = (time.perf_counter() - start) / 1000 * 1000
        print(f"    Argument parsing: {arg_parse_time:.4f} ms")

        # 2. Validation overhead (CPU-bound)
        def validation_fn():
            input_numel = x.numel()
            output_numel = 1
            for s in output_shape:
                output_numel *= s
            return output_numel == input_numel

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(1000):
            validation_fn()
        validation_time = (time.perf_counter() - start) / 1000 * 1000
        print(f"    Validation:       {validation_time:.4f} ms")

        # 3. Output tensor allocation (GPU, use do_bench)
        def alloc_fn():
            return torch.empty(output_shape, dtype=x.dtype, device=x.device)

        alloc_time = triton.testing.do_bench(alloc_fn, warmup=25, rep=100)
        print(f"    Memory allocation: {alloc_time:.4f} ms")

        # 4. Triton kernel launch (use do_bench for accurate GPU timing)
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        from view.view_implementation_v1 import view_triton_kernel
        output = torch.empty(output_shape, dtype=x.dtype, device=x.device)

        # Warmup to compile
        view_triton_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        torch.cuda.synchronize()

        # Measure kernel launch with do_bench
        def kernel_fn():
            view_triton_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

        kernel_time = triton.testing.do_bench(kernel_fn, warmup=25, rep=100)
        print(f"    Kernel execution: {kernel_time:.4f} ms")

        # 5. Compare with native view (use do_bench)
        def native_view_fn():
            return x.view(output_shape)

        native_time = triton.testing.do_bench(native_view_fn, warmup=25, rep=100)
        print(f"    Native view:      {native_time:.4f} ms (baseline)")

        print(f"\n  KEY INSIGHT: view() is a metadata-only operation in PyTorch.")
        print(f"  The custom kernel copies data, which is fundamentally wrong for view.")

    def analyze_add(self):
        """Analyze add operation overhead."""
        print("\n" + "=" * 80)
        print("ANALYZING: aten.add.Tensor (2.76x slower)")
        print("=" * 80)

        from add.add_implementation_v1 import add_kernel_impl

        test_cases = [
            ("Small", (12, 64, 128)),
            ("Medium", (12, 256, 512)),
            ("Large", (12, 1024, 768)),
        ]

        for name, shape in test_cases:
            print(f"\n--- {name}: {shape} ---")
            x = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
            y = torch.randn(shape, device='cuda', dtype=torch.bfloat16)

            # Native PyTorch
            native_stats = benchmark_function(torch.add, (x, y))

            # Custom kernel
            custom_stats = benchmark_function(add_kernel_impl, (x, y))

            print(f"  Native PyTorch: {native_stats['mean']:.4f} ± {native_stats['std']:.4f} ms")
            print(f"  Custom Kernel:  {custom_stats['mean']:.4f} ± {custom_stats['std']:.4f} ms")
            print(f"  Slowdown:       {custom_stats['mean'] / native_stats['mean']:.2f}x")

            # Detailed breakdown
            print("\n  Overhead Breakdown:")
            self._analyze_add_overhead(x, y, add_kernel_impl)

    def _analyze_add_overhead(self, x, y, kernel_impl):
        """Break down overhead sources for add operation using Triton do_bench."""

        # 1. Argument parsing (CPU-bound)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(1000):
            input_tensor = x
            other = y
            alpha = 1.0
            out = None
        arg_time = (time.perf_counter() - start) / 1000 * 1000
        print(f"    Argument parsing: {arg_time:.6f} ms")

        # 2. Broadcasting check (CPU-bound)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(1000):
            broadcasted_shape = torch.broadcast_shapes(x.shape, y.shape)
        broadcast_time = (time.perf_counter() - start) / 1000 * 1000
        print(f"    Broadcast check:  {broadcast_time:.6f} ms")

        # 3. Contiguous check (GPU, use do_bench)
        def contiguous_fn():
            x_cont = x.contiguous()
            y_cont = y.contiguous()
            return x_cont, y_cont

        contiguous_time = triton.testing.do_bench(contiguous_fn, warmup=25, rep=100)
        print(f"    Contiguous call:  {contiguous_time:.6f} ms")

        # 4. Output allocation (GPU, use do_bench)
        def alloc_fn():
            return torch.empty_like(x)

        alloc_time = triton.testing.do_bench(alloc_fn, warmup=25, rep=100)
        print(f"    Memory allocation: {alloc_time:.4f} ms")

        # 5. Just the Triton kernel (use do_bench)
        from add.add_implementation_v1 import add_triton_kernel
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        output = torch.empty_like(x)

        # Warmup to compile
        add_triton_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        torch.cuda.synchronize()

        def kernel_fn():
            add_triton_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

        kernel_time = triton.testing.do_bench(kernel_fn, warmup=25, rep=100)
        print(f"    Triton kernel:    {kernel_time:.4f} ms")

        # Compare with native (use do_bench)
        def native_add_fn():
            return torch.add(x, y)

        native_time = triton.testing.do_bench(native_add_fn, warmup=25, rep=100)
        print(f"    Native torch.add: {native_time:.4f} ms")

    def analyze_split(self):
        """Analyze split operation overhead."""
        print("\n" + "=" * 80)
        print("ANALYZING: aten.split.Tensor (5.39x slower)")
        print("=" * 80)

        from split.split_implementation_v1 import split_kernel_impl

        # Test case matching nanoGPT usage: split for q, k, v
        print(f"\n--- nanoGPT QKV split: (12, 64, 384) -> 3x (12, 64, 128) ---")
        x = torch.randn(12, 64, 384, device='cuda', dtype=torch.bfloat16)

        # Native PyTorch
        native_stats = benchmark_function(lambda t: t.split(128, dim=2), (x,))

        # Custom kernel
        custom_stats = benchmark_function(split_kernel_impl, (x, 128, 2))

        print(f"  Native PyTorch: {native_stats['mean']:.4f} ± {native_stats['std']:.4f} ms")
        print(f"  Custom Kernel:  {custom_stats['mean']:.4f} ± {custom_stats['std']:.4f} ms")
        print(f"  Slowdown:       {custom_stats['mean'] / native_stats['mean']:.2f}x")

        print("\n  Overhead Breakdown:")

        # Native split is also just view + slice (metadata operations)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            splits = x.split(128, dim=2)
        torch.cuda.synchronize()
        native_time = (time.perf_counter() - start) / 100 * 1000
        print(f"    Native split (metadata only): {native_time:.4f} ms")

        # Check if the native split creates copies
        splits = x.split(128, dim=2)
        print(f"    Native split storage shared: {splits[0].storage().data_ptr() == x.storage().data_ptr()}")

        print(f"\n  KEY INSIGHT: split() returns views, no data copy.")
        print(f"  The custom kernel performs actual data copies for each split.")

    def analyze_unsafe_view(self):
        """Analyze _unsafe_view operation overhead."""
        print("\n" + "=" * 80)
        print("ANALYZING: aten._unsafe_view (3.69x slower)")
        print("=" * 80)

        from _unsafe_view._unsafe_view_implementation_v1 import _unsafe_view_kernel_impl

        test_cases = [
            ("Small", (12, 64, 128), (12, 64 * 128)),
            ("Medium", (12, 256, 512), (12, 256 * 512)),
        ]

        for name, input_shape, output_shape in test_cases:
            print(f"\n--- {name}: {input_shape} -> {output_shape} ---")
            x = torch.randn(input_shape, device='cuda', dtype=torch.bfloat16)

            # Native PyTorch _unsafe_view (same as view)
            native_stats = benchmark_function(
                lambda t, s: torch.ops.aten._unsafe_view(t, s), (x, output_shape))

            # Custom kernel
            custom_stats = benchmark_function(_unsafe_view_kernel_impl, (x, output_shape))

            print(f"  Native PyTorch: {native_stats['mean']:.4f} ± {native_stats['std']:.4f} ms")
            print(f"  Custom Kernel:  {custom_stats['mean']:.4f} ± {custom_stats['std']:.4f} ms")
            print(f"  Slowdown:       {custom_stats['mean'] / native_stats['mean']:.2f}x")

        print(f"\n  KEY INSIGHT: Same as view() - metadata only in PyTorch,")
        print(f"  but the custom kernel copies all data.")

    def analyze_mm(self):
        """Analyze mm (matrix multiply) operation overhead."""
        print("\n" + "=" * 80)
        print("ANALYZING: aten.mm (1.36x slower)")
        print("=" * 80)

        from mm.mm_implementation_v1 import mm_kernel_impl

        test_cases = [
            ("Small", (768, 128), (128, 512)),
            ("Medium", (768, 512), (512, 768)),
            ("Large", (768, 768), (768, 768)),
        ]

        for name, shape_a, shape_b in test_cases:
            print(f"\n--- {name}: {shape_a} x {shape_b} ---")
            a = torch.randn(shape_a, device='cuda', dtype=torch.bfloat16)
            b = torch.randn(shape_b, device='cuda', dtype=torch.bfloat16)

            # Native PyTorch (uses cuBLAS)
            native_stats = benchmark_function(torch.mm, (a, b))

            # Custom kernel
            custom_stats = benchmark_function(mm_kernel_impl, (a, b))

            print(f"  Native PyTorch (cuBLAS): {native_stats['mean']:.4f} ± {native_stats['std']:.4f} ms")
            print(f"  Custom Triton Kernel:    {custom_stats['mean']:.4f} ± {custom_stats['std']:.4f} ms")
            print(f"  Slowdown:                {custom_stats['mean'] / native_stats['mean']:.2f}x")

        print(f"\n  NOTE: For mm, the Triton kernel is reasonably optimized")
        print(f"  but cuBLAS has years of optimization for specific GPU architectures.")

    def analyze_gelu(self):
        """Analyze GELU operation overhead."""
        print("\n" + "=" * 80)
        print("ANALYZING: aten.gelu (1.76x slower)")
        print("=" * 80)

        from gelu.gelu_implementation_v1 import gelu_kernel_impl

        test_cases = [
            ("Small", (12, 64, 512)),
            ("Medium", (12, 256, 2048)),
        ]

        for name, shape in test_cases:
            print(f"\n--- {name}: {shape} ---")
            x = torch.randn(shape, device='cuda', dtype=torch.bfloat16)

            # Native PyTorch
            native_stats = benchmark_function(torch.nn.functional.gelu, (x,))

            # Custom kernel
            custom_stats = benchmark_function(gelu_kernel_impl, (x,))

            print(f"  Native PyTorch: {native_stats['mean']:.4f} ± {native_stats['std']:.4f} ms")
            print(f"  Custom Kernel:  {custom_stats['mean']:.4f} ± {custom_stats['std']:.4f} ms")
            print(f"  Slowdown:       {custom_stats['mean'] / native_stats['mean']:.2f}x")

    def run_comprehensive_analysis(self):
        """Run analysis for all slow kernels."""
        print("=" * 80)
        print("COMPREHENSIVE KERNEL OVERHEAD ANALYSIS")
        print("=" * 80)
        print("\nThis analysis compares custom BackendBench kernels with PyTorch eager mode")
        print("to identify the root causes of kernel execution overhead.\n")

        self.analyze_view()
        self.analyze_unsafe_view()
        self.analyze_split()
        self.analyze_add()
        self.analyze_mm()
        self.analyze_gelu()

        self.print_summary()

    def print_summary(self):
        """Print summary of findings."""
        print("\n" + "=" * 80)
        print("SUMMARY: ROOT CAUSES OF KERNEL OVERHEAD")
        print("=" * 80)

        print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OVERHEAD CATEGORIES                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. SEMANTIC MISMATCH (view, _unsafe_view, split) - CRITICAL                │
│     ────────────────────────────────────────────────────────────────────    │
│     • PyTorch: These are METADATA-ONLY operations (zero-copy)               │
│     • Custom:  Kernels COPY DATA, fundamentally changing semantics          │
│     • Impact:  ~5x slowdown, memory bandwidth wasted                        │
│     • Fix:     Return views instead of launching copy kernels               │
│                                                                             │
│  2. PYTHON WRAPPER OVERHEAD (all operations) - MODERATE                     │
│     ────────────────────────────────────────────────────────────────────    │
│     • Argument parsing with isinstance(), dict lookups                      │
│     • Device checks and transfers (if not input.is_cuda: ...)               │
│     • Contiguity checks (.contiguous() calls)                               │
│     • Validation logic in Python                                            │
│     • Impact:  ~0.01-0.05 ms per call (significant for small tensors)       │
│     • Fix:     Move to C++/Cython, or use @torch.jit.script                 │
│                                                                             │
│  3. MEMORY ALLOCATION (all operations) - MODERATE                           │
│     ────────────────────────────────────────────────────────────────────    │
│     • torch.empty() allocates new memory each call                          │
│     • PyTorch can reuse memory pools; custom kernels don't                  │
│     • Impact:  ~0.01-0.02 ms per allocation                                 │
│     • Fix:     Use output tensor reuse, work with PyTorch allocator         │
│                                                                             │
│  4. TRITON KERNEL LAUNCH OVERHEAD - LOW TO MODERATE                         │
│     ────────────────────────────────────────────────────────────────────    │
│     • Grid calculation in Python                                            │
│     • Kernel dispatch through Triton runtime                                │
│     • First-call JIT compilation (amortized over many calls)                │
│     • Impact:  ~0.005-0.01 ms per launch                                    │
│     • Fix:     Kernel fusion, persistent kernels                            │
│                                                                             │
│  5. ALGORITHMIC EFFICIENCY (mm, gelu, etc.) - VARIES                        │
│     ────────────────────────────────────────────────────────────────────    │
│     • Triton kernels may not match cuBLAS/cuDNN optimization                │
│     • Block sizes may not be optimal for specific GPU                       │
│     • Missing advanced optimizations (tensor cores, async copy)             │
│     • Impact:  1.3-2x for compute-bound ops                                 │
│     • Fix:     Tune block sizes, use autotune, leverage tensor cores        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         RECOMMENDATIONS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PRIORITY 1 - Fix Semantic Mismatches:                                      │
│  • view, _unsafe_view: Return input.view(shape) directly                    │
│  • split: Return input.split() directly                                     │
│  • These should NOT launch GPU kernels at all                               │
│                                                                             │
│  PRIORITY 2 - Reduce Python Overhead:                                       │
│  • Use @torch.jit.script for wrapper functions                              │
│  • Avoid isinstance() checks in hot path                                    │
│  • Cache device checks                                                      │
│                                                                             │
│  PRIORITY 3 - Optimize Compute Kernels:                                     │
│  • Use Triton autotune for block sizes                                      │
│  • Enable tensor cores for mm operations                                    │
│  • Fuse elementwise operations where possible                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")


def main():
    torch.manual_seed(42)
    analyzer = KernelAnalyzer()
    analyzer.run_comprehensive_analysis()


if __name__ == "__main__":
    main()
