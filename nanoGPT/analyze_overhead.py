"""
Overhead Analysis Script for BackendBench

This script analyzes the overhead introduced by BackendBench.enable() during training.
It breaks down the overhead into:
1. Op registration overhead (enable/disable calls)
2. Custom kernel execution overhead vs native PyTorch kernels
3. Per-operator breakdown to identify slow kernels

Usage:
    source ../.venv/bin/activate
    python analyze_overhead.py config/train_shakespeare_char.py \
        --device=cuda --compile=False --eval_iters=20 \
        --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 \
        --kernel_folder="../generated_kernels_opinfo"
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
from collections import defaultdict
from typing import Dict, List, Tuple
import functools

import numpy as np
import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Config (same defaults as train.py)
out_dir = 'out'
eval_interval = 2000
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2'
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 12
block_size = 64
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False
learning_rate = 6e-4
max_iters = 100
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
kernel_folder = "../generated_kernels_torchbench"
num_warmup_iters = 5
num_benchmark_iters = 20

config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# Operator Timing Tracker using TorchDispatchMode


class OpTimingTracker(TorchDispatchMode):
    """Track execution time for each operator during forward pass."""

    def __init__(self):
        self.op_times: Dict[str, List[float]] = defaultdict(list)
        self.op_counts: Dict[str, int] = defaultdict(int)
        self.total_time = 0.0
        self._enabled = True

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if not self._enabled:
            return func(*args, **kwargs)

        op_name = str(func)

        torch.cuda.synchronize()
        start = time.perf_counter()

        result = func(*args, **kwargs)

        torch.cuda.synchronize()
        end = time.perf_counter()

        elapsed = end - start
        self.op_times[op_name].append(elapsed)
        self.op_counts[op_name] += 1
        self.total_time += elapsed

        return result

    def reset(self):
        self.op_times.clear()
        self.op_counts.clear()
        self.total_time = 0.0

    def get_summary(self) -> Dict[str, Tuple[float, int, float]]:
        """Returns dict of op_name -> (total_time, count, avg_time)"""
        summary = {}
        for op_name, times in self.op_times.items():
            total = sum(times)
            count = len(times)
            avg = total / count if count > 0 else 0
            summary[op_name] = (total, count, avg)
        return summary

    def print_top_ops(self, n=20):
        """Print top N operators by total time."""
        summary = self.get_summary()
        sorted_ops = sorted(summary.items(), key=lambda x: x[1][0], reverse=True)

        print("\n" + "=" * 80)
        print(f"Top {n} Operators by Total Time")
        print("=" * 80)
        print(f"{'Operator':<50} {'Total(ms)':>10} {'Count':>8} {'Avg(ms)':>10}")
        print("-" * 80)

        for op_name, (total, count, avg) in sorted_ops[:n]:
            short_name = op_name[-48:] if len(op_name) > 48 else op_name
            print(f"{short_name:<50} {total*1000:>10.3f} {count:>8} {avg*1000:>10.4f}")

        print("-" * 80)
        print(f"{'Total':<50} {self.total_time*1000:>10.3f}")
        print("=" * 80 + "\n")


# -----------------------------------------------------------------------------
# Data loading

data_dir = os.path.join('data', dataset)


def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# -----------------------------------------------------------------------------
# Benchmark Functions


def benchmark_forward_pass(model, X, Y, ctx, num_iters=20, warmup=5):
    """Benchmark forward pass without any overhead."""
    times = []

    # Warmup
    for _ in range(warmup):
        with ctx:
            logits, loss = model(X, Y)
        torch.cuda.synchronize()

    # Benchmark
    for _ in range(num_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with ctx:
            logits, loss = model(X, Y)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    return times


def benchmark_with_backendbench(model, X, Y, ctx, kernel_folder, num_iters=20, warmup=5):
    """Benchmark forward pass with BackendBench enable/disable overhead."""
    import BackendBench

    forward_times = []
    enable_times = []
    disable_times = []
    total_times = []

    # Warmup
    for _ in range(warmup):
        BackendBench.enable(kernel_folder)
        with ctx:
            logits, loss = model(X, Y)
        BackendBench.disable()
        torch.cuda.synchronize()

    # Benchmark
    for _ in range(num_iters):
        torch.cuda.synchronize()

        total_start = time.perf_counter()

        enable_start = time.perf_counter()
        BackendBench.enable(kernel_folder)
        enable_end = time.perf_counter()

        forward_start = time.perf_counter()
        with ctx:
            logits, loss = model(X, Y)
        torch.cuda.synchronize()
        forward_end = time.perf_counter()

        disable_start = time.perf_counter()
        BackendBench.disable()
        disable_end = time.perf_counter()

        total_end = time.perf_counter()

        enable_times.append(enable_end - enable_start)
        forward_times.append(forward_end - forward_start)
        disable_times.append(disable_end - disable_start)
        total_times.append(total_end - total_start)

    return {
        'forward': forward_times,
        'enable': enable_times,
        'disable': disable_times,
        'total': total_times
    }


def benchmark_with_backendbench_persistent(model, X, Y, ctx, kernel_folder, num_iters=20, warmup=5):
    """Benchmark forward pass with BackendBench enabled once (persistent)."""
    import BackendBench

    forward_times = []

    # Enable once
    enable_start = time.perf_counter()
    BackendBench.enable(kernel_folder)
    enable_end = time.perf_counter()
    enable_times = [enable_end - enable_start]

    # Warmup
    for _ in range(warmup):
        with ctx:
            logits, loss = model(X, Y)
        torch.cuda.synchronize()

    # Benchmark
    for _ in range(num_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with ctx:
            logits, loss = model(X, Y)
        torch.cuda.synchronize()
        end = time.perf_counter()
        forward_times.append(end - start)
    
    disable_start = time.perf_counter()
    BackendBench.disable()
    disable_end = time.perf_counter()
    disable_times = [disable_end - disable_start]


    return {
        'forward': forward_times,
        'enable': enable_times,
        'disable': disable_times
    }


class OpCollectorMode(TorchDispatchMode):
    """Collect operators used during forward pass without timing overhead."""

    def __init__(self):
        self.ops: List[str] = []
        self._enabled = True

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if self._enabled:
            self.ops.append(str(func))

        return func(*args, **kwargs)

    def reset(self):
        self.ops.clear()

    def get_unique_ops(self) -> List[str]:
        """Returns list of unique operators in order of first occurrence."""
        seen = set()
        unique = []
        for op in self.ops:
            if op not in seen:
                seen.add(op)
                unique.append(op)
        return unique


def get_registered_custom_ops(kernel_folder: str) -> List[str]:
    """
    Get the list of custom PyTorch operators registered via BackendBench.
    
    Scans the kernel_folder directory for operator implementations and returns
    the corresponding PyTorch ATen operator names.
    
    Args:
        kernel_folder: Path to the kernel folder containing operator implementations.
        
    Returns:
        List of ATen operator names (e.g., 'aten.add.Tensor', 'aten.gelu.default')
    """
    if not os.path.isdir(kernel_folder):
        raise ValueError(f"Kernel folder does not exist: {kernel_folder}")

    registered_ops = []
    
    for item in os.listdir(kernel_folder):
        item_path = os.path.join(kernel_folder, item)
        if os.path.isdir(item_path) and not item.startswith('_' * 2):
            impl_files = [f for f in os.listdir(item_path) if f.endswith('_implementation_v1.py')]
            if impl_files:
                op_name = f"aten.{item}.default"
                if item == 'add':
                    op_name = "aten.add.Tensor"
                elif item == 'div':
                    op_name = "aten.div.Tensor"
                elif item == 'mul':
                    op_name = "aten.mul.Tensor"
                elif item == 'split':
                    op_name = "aten.split.Tensor"
                elif item == 'arange':
                    op_name = "aten.arange.start"
                registered_ops.append(op_name)
    
    return sorted(registered_ops)


def get_ops_used_in_forward_pass(model, X, Y, ctx) -> List[str]:
    """
    Get the list of PyTorch operators used during the forward pass of a model.
    
    Args:
        model: The PyTorch model (e.g., GPT).
        X: Input tensor.
        Y: Target tensor.
        ctx: Autocast context manager.
        
    Returns:
        List of unique operator names used during forward pass.
    """
    collector = OpCollectorMode()
    
    for _ in range(2):
        with ctx:
            model(X, Y)
        torch.cuda.synchronize()
    
    collector.reset()
    with collector:
        with ctx:
            logits, loss = model(X, Y)
    torch.cuda.synchronize()
    
    return collector.get_unique_ops()


def get_custom_ops_used_in_benchmark_forward_pass(
    model, 
    X, 
    Y, 
    ctx, 
    kernel_folder: str
) -> Dict[str, List[str]]:
    """
    Get the list of custom PyTorch operators that are:
    1. Registered via BackendBench from the kernel_folder
    2. Used during benchmark_forward_pass of GPT
    
    This function identifies which of the registered custom kernels are actually
    invoked during model inference, useful for analyzing which custom implementations
    will impact performance.
    
    Args:
        model: The GPT model to analyze.
        X: Input tensor for the forward pass.
        Y: Target tensor for the forward pass.
        ctx: Autocast context manager (e.g., torch.amp.autocast).
        kernel_folder: Path to the folder containing custom kernel implementations.
        
    Returns:
        Dictionary containing:
            - 'registered_ops': List of all operators registered via BackendBench
            - 'forward_pass_ops': List of all operators used in forward pass
            - 'custom_ops_used': List of registered custom operators that are used
            - 'custom_ops_unused': List of registered custom operators not used
            - 'native_ops_used': List of native ops used (not covered by custom kernels)
            
    Example:
        >>> result = get_custom_ops_used_in_benchmark_forward_pass(
        ...     model, X, Y, ctx, "../generated_kernels_opinfo"
        ... )
        >>> print(result['custom_ops_used'])
        ['aten.add.Tensor', 'aten.gelu.default', 'aten.mm.default', ...]
    """
    registered_ops = get_registered_custom_ops(kernel_folder)
    registered_set = set(registered_ops)
    
    forward_ops = get_ops_used_in_forward_pass(model, X, Y, ctx)
    forward_set = set(forward_ops)
    
    custom_ops_used = [op for op in registered_ops if op in forward_set]
    custom_ops_unused = [op for op in registered_ops if op not in forward_set]
    native_ops_used = [op for op in forward_ops if op not in registered_set]
    
    return {
        'registered_ops': registered_ops,
        'forward_pass_ops': forward_ops,
        'custom_ops_used': custom_ops_used,
        'custom_ops_unused': custom_ops_unused,
        'native_ops_used': native_ops_used,
    }


def print_custom_ops_analysis(result: Dict[str, List[str]]):
    """Pretty print the results from get_custom_ops_used_in_benchmark_forward_pass."""
    print("\n" + "=" * 80)
    print("CUSTOM OPERATORS ANALYSIS")
    print("=" * 80)
    
    print(f"\n--- Registered Custom Operators ({len(result['registered_ops'])}) ---")
    for op in result['registered_ops']:
        print(f"  {op}")
    
    print(f"\n--- Custom Operators Used in Forward Pass ({len(result['custom_ops_used'])}) ---")
    for op in result['custom_ops_used']:
        print(f"  ✓ {op}")
    
    print(f"\n--- Custom Operators NOT Used ({len(result['custom_ops_unused'])}) ---")
    for op in result['custom_ops_unused']:
        print(f"  ✗ {op}")
    
    print(f"\n--- Native Operators (no custom kernel) ({len(result['native_ops_used'])}) ---")
    for op in result['native_ops_used']:
        print(f"  • {op}")
    
    print("\n" + "=" * 80)
    coverage = len(result['custom_ops_used']) / len(result['forward_pass_ops']) * 100 if result['forward_pass_ops'] else 0
    print(f"Custom kernel coverage: {len(result['custom_ops_used'])}/{len(result['forward_pass_ops'])} operators ({coverage:.1f}%)")
    print("=" * 80 + "\n")


def profile_operators(model, X, Y, ctx, use_backendbench=False, kernel_folder=None):
    """Profile individual operator execution times."""
    tracker = OpTimingTracker()

    if use_backendbench:
        import BackendBench
        BackendBench.enable(kernel_folder)

    # Warmup
    for _ in range(3):
        with ctx:
            logits, loss = model(X, Y)
        torch.cuda.synchronize()

    # Profile
    tracker.reset()
    with tracker:
        with ctx:
            logits, loss = model(X, Y)
    torch.cuda.synchronize()

    if use_backendbench:
        import BackendBench
        BackendBench.disable()

    return tracker


# -----------------------------------------------------------------------------
# Main Analysis


def print_stats(times, name):
    """Print statistics for a list of times."""
    times = np.array(times) * 1000  # Convert to ms
    print(f"  {name}:")
    print(f"    Mean:   {np.mean(times):.3f} ms")
    print(f"    Std:    {np.std(times):.3f} ms")
    print(f"    Min:    {np.min(times):.3f} ms")
    print(f"    Max:    {np.max(times):.3f} ms")
    print(f"    Median: {np.median(times):.3f} ms")


def main():
    print("=" * 80)
    print("BackendBench Overhead Analysis")
    print("=" * 80)

    # Setup
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Model setup
    print(f"\nModel config: n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}")
    print(f"Input config: batch_size={batch_size}, block_size={block_size}")

    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      bias=bias, vocab_size=65, dropout=dropout)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(device)
    model.eval()

    # Get batch
    X, Y = get_batch('train')

    print(f"\nRunning benchmarks with {num_warmup_iters} warmup and {num_benchmark_iters} benchmark iterations...")

    # -----------------------------------------------------------------------------
    # Benchmark 1: Baseline (no BackendBench)
    print("\n" + "-" * 80)
    print("1. BASELINE: Native PyTorch (no BackendBench)")
    print("-" * 80)

    baseline_times = benchmark_forward_pass(model, X, Y, ctx, num_iters=num_benchmark_iters, warmup=num_warmup_iters)
    print_stats(baseline_times, "Forward Pass")

    # -----------------------------------------------------------------------------
    # Benchmark 2: BackendBench with enable/disable each iteration (current approach)
    print("\n" + "-" * 80)
    print("2. BACKENDBENCH: Enable/Disable each iteration (current approach)")
    print("-" * 80)

    bb_times = benchmark_with_backendbench(model, X, Y, ctx, kernel_folder,
                                           num_iters=num_benchmark_iters, warmup=num_warmup_iters)
    print_stats(bb_times['forward'], "Forward Pass (with custom kernels)")
    print_stats(bb_times['enable'], "BackendBench.enable()")
    print_stats(bb_times['disable'], "BackendBench.disable()")
    print_stats(bb_times['total'], "Total (enable + forward + disable)")

    # -----------------------------------------------------------------------------
    # Benchmark 3: BackendBench persistent (enable once)
    print("\n" + "-" * 80)
    print("3. BACKENDBENCH PERSISTENT: Enable once, run multiple iterations")
    print("-" * 80)

    bb_persistent_times = benchmark_with_backendbench_persistent(
        model, X, Y, ctx, kernel_folder, num_iters=num_benchmark_iters, warmup=num_warmup_iters)
    print_stats(bb_persistent_times['forward'], "Forward Pass (custom kernels, no re-registration)")
    print_stats(bb_persistent_times['enable'], "BackendBench.enable()")
    print_stats(bb_persistent_times['disable'], "BackendBench.disable()")

    # Note: BackendBench is still enabled at this point
    import BackendBench
    BackendBench.disable()

    # Benchmark 4: Baseline (no BackendBench) with persistent registration
    result = get_custom_ops_used_in_benchmark_forward_pass(model, X, Y, ctx, kernel_folder)

    # Print detailed analysis
    print_custom_ops_analysis(result)

    # -----------------------------------------------------------------------------
    # Analysis Summary
    print("\n" + "=" * 80)
    print("OVERHEAD ANALYSIS SUMMARY")
    print("=" * 80)

    baseline_mean = np.mean(baseline_times) * 1000
    bb_forward_mean = np.mean(bb_times['forward']) * 1000
    bb_enable_mean = np.mean(bb_times['enable']) * 1000
    bb_disable_mean = np.mean(bb_times['disable']) * 1000
    bb_total_mean = np.mean(bb_times['total']) * 1000
    bb_persistent_forward_mean = np.mean(bb_persistent_times['forward']) * 1000
    bb_persistent_enable_mean = np.mean(bb_persistent_times['enable']) * 1000
    bb_persistent_disable_mean = np.mean(bb_persistent_times['disable']) * 1000


    print(f"\n1. Baseline forward pass:              {baseline_mean:.3f} ms")
    print(f"2. BackendBench (per-iter enable):")
    print(f"   - Forward with custom kernels:      {bb_forward_mean:.3f} ms")
    print(f"   - enable() overhead:                {bb_enable_mean:.3f} ms")
    print(f"   - disable() overhead:               {bb_disable_mean:.3f} ms")
    print(f"   - Total per iteration:              {bb_total_mean:.3f} ms")
    print(f"3. BackendBench (persistent enable):")
    print(f"   - Forward with custom kernels:      {bb_persistent_forward_mean:.3f} ms")
    print(f"   - enable() overhead:                {bb_persistent_enable_mean:.3f} ms")
    print(f"   - disable() overhead:               {bb_persistent_disable_mean:.3f} ms")

    print(f"\n--- OVERHEAD BREAKDOWN ---")
    registration_overhead = bb_enable_mean + bb_disable_mean
    kernel_overhead = bb_persistent_forward_mean - baseline_mean
    total_overhead = bb_total_mean - baseline_mean

    print(f"\nRegistration Overhead (enable + disable): {registration_overhead:.3f} ms ({registration_overhead/total_overhead*100:.1f}% of total overhead)")
    print(f"Kernel Execution Overhead:                {kernel_overhead:.3f} ms ({kernel_overhead/total_overhead*100:.1f}% of total overhead)")
    print(f"Total Overhead:                           {total_overhead:.3f} ms")

    print(f"\nSlowdown factors:")
    print(f"  - With per-iter registration:  {bb_total_mean/baseline_mean:.2f}x slower")
    print(f"  - With persistent registration: {bb_persistent_forward_mean/baseline_mean:.2f}x slower")

    # -----------------------------------------------------------------------------
    # Per-Operator Profiling (focused on the 8 custom operators used in forward pass)
    print("\n" + "=" * 80)
    print("PER-OPERATOR PROFILING (8 Custom Operators Used in Forward Pass)")
    print("=" * 80)

    # Get the 8 custom operators used in forward pass
    custom_ops_used = result['custom_ops_used']
    custom_ops_set = set(custom_ops_used)

    print("\n--- Native PyTorch Operators ---")
    baseline_tracker = profile_operators(model, X, Y, ctx, use_backendbench=False)
    baseline_summary = baseline_tracker.get_summary()

    # Filter and print only the 8 custom operators
    filtered_baseline = {op: data for op, data in baseline_summary.items() if op in custom_ops_set}
    sorted_baseline = sorted(filtered_baseline.items(), key=lambda x: x[1][0], reverse=True)

    print("\n" + "=" * 80)
    print(f"Custom Operators Used in Forward Pass ({len(custom_ops_used)} operators)")
    print("=" * 80)
    print(f"{'Operator':<50} {'Total(ms)':>10} {'Count':>8} {'Avg(ms)':>10}")
    print("-" * 80)
    baseline_total = 0.0
    for op_name, (total, count, avg) in sorted_baseline:
        short_name = op_name[-48:] if len(op_name) > 48 else op_name
        print(f"{short_name:<50} {total*1000:>10.3f} {count:>8} {avg*1000:>10.4f}")
        baseline_total += total
    print("-" * 80)
    print(f"{'Total':<50} {baseline_total*1000:>10.3f}")
    print("=" * 80)

    print("\n--- BackendBench Custom Kernels ---")
    bb_tracker = profile_operators(model, X, Y, ctx, use_backendbench=True, kernel_folder=kernel_folder)
    bb_summary = bb_tracker.get_summary()

    # Filter and print only the 8 custom operators
    filtered_bb = {op: data for op, data in bb_summary.items() if op in custom_ops_set}
    sorted_bb = sorted(filtered_bb.items(), key=lambda x: x[1][0], reverse=True)

    print("\n" + "=" * 80)
    print(f"Custom Operators Used in Forward Pass ({len(custom_ops_used)} operators)")
    print("=" * 80)
    print(f"{'Operator':<50} {'Total(ms)':>10} {'Count':>8} {'Avg(ms)':>10}")
    print("-" * 80)
    bb_total = 0.0
    for op_name, (total, count, avg) in sorted_bb:
        short_name = op_name[-48:] if len(op_name) > 48 else op_name
        print(f"{short_name:<50} {total*1000:>10.3f} {count:>8} {avg*1000:>10.4f}")
        bb_total += total
    print("-" * 80)
    print(f"{'Total':<50} {bb_total*1000:>10.3f}")
    print("=" * 80)

    # Compare operators (only the 8 custom operators)
    print("\n" + "=" * 80)
    print("OPERATOR COMPARISON (Custom vs Native) - 8 Custom Operators Only")
    print("=" * 80)

    comparisons = []
    for op_name in custom_ops_used:
        if op_name in bb_summary and op_name in baseline_summary:
            bb_total, bb_count, bb_avg = bb_summary[op_name]
            base_total, base_count, base_avg = baseline_summary[op_name]
            if base_avg > 0:
                slowdown = bb_avg / base_avg
                impact = (bb_avg - base_avg) * bb_count
                comparisons.append((op_name, base_avg, bb_avg, slowdown, bb_count, impact))

    # Sort by impact (biggest slowdowns first)
    comparisons.sort(key=lambda x: x[5], reverse=True)

    print(f"\n{'Operator':<50} {'Native(ms)':>10} {'Custom(ms)':>10} {'Slowdown':>10} {'Impact(ms)':>10}")
    print("-" * 92)
    total_native = 0.0
    total_custom = 0.0
    total_impact = 0.0
    for op_name, base_avg, bb_avg, slowdown, count, impact in comparisons:
        short_name = op_name[-48:] if len(op_name) > 48 else op_name
        print(f"{short_name:<50} {base_avg*1000:>10.4f} {bb_avg*1000:>10.4f} {slowdown:>10.2f}x {impact*1000:>10.3f}")
        total_native += base_avg * count
        total_custom += bb_avg * count
        total_impact += impact
    print("-" * 92)
    overall_slowdown = total_custom / total_native if total_native > 0 else 0
    print(f"{'TOTAL (8 custom ops)':<50} {total_native*1000:>10.4f} {total_custom*1000:>10.4f} {overall_slowdown:>10.2f}x {total_impact*1000:>10.3f}")

    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)

    if registration_overhead > kernel_overhead:
        print("""
The REGISTRATION OVERHEAD (enable/disable) is the PRIMARY cause of slowdown.

RECOMMENDATIONS:
1. Enable BackendBench once at the start of training, not per-iteration
2. Modify train.py to call BackendBench.enable() before the training loop
3. Only call BackendBench.disable() when training completes

Code change suggestion for train.py:
    # Before training loop:
    if use_backendbench:
        BackendBench.enable(kernel_folder)

    # Training loop (remove enable/disable from inside):
    while True:
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)  # No enable/disable here
        ...

    # After training:
    if use_backendbench:
        BackendBench.disable()
""")
    else:
        print("""
The CUSTOM KERNEL EXECUTION is the PRIMARY cause of slowdown.

RECOMMENDATIONS:
1. Review the custom kernel implementations for performance issues
2. Check if kernels are properly optimized (e.g., using Triton efficiently)
3. Profile individual slow kernels and optimize them
4. Consider if certain operators should fall back to native PyTorch
""")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
