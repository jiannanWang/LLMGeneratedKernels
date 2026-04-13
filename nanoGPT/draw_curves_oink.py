# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import os
import re

import matplotlib.pyplot as plt


def read_log_file(log_path):
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found at: {log_path}")
    with open(log_path, "r") as file:
        return file.readlines()


def parse_training_logs(log_lines):
    results = {}

    # New pattern: includes backward_time and iter_time
    new_pattern = (
        r"iter (\d+): loss ([\d\.]+), time ([\d\.]+)ms, "
        r"backward_time ([\d\.]+)ms, iter_time ([\d\.]+)ms, "
        r"backendbench overhead time ([\d\.]+)ms, "
        r"time exclude backendbench overhead ([\d\.]+), "
        r"mfu ([\d\.\-]+)%"
    )
    # Old pattern: forward time only
    old_pattern = (
        r"iter (\d+): loss ([\d\.]+), time ([\d\.]+)ms, "
        r"backendbench overhead time ([\d\.]+)ms, "
        r"time exclude backendbench overhead ([\d\.]+), "
        r"mfu ([\d\.\-]+)%"
    )

    for line in log_lines:
        match = re.search(new_pattern, line)
        if match:
            iteration = int(match.group(1))
            if iteration % 10 != 0:
                continue
            results[iteration] = {
                "loss": float(match.group(2)),
                "time": float(match.group(3)),
                "backward_time": float(match.group(4)),
                "iter_time": float(match.group(5)),
                "overhead_time": float(match.group(6)),
                "time_exclude": float(match.group(7)),
                "mfu": float(match.group(8)),
            }
            continue

        match = re.search(old_pattern, line)
        if match:
            iteration = int(match.group(1))
            if iteration % 10 != 0:
                continue
            fwd_time = float(match.group(3))
            results[iteration] = {
                "loss": float(match.group(2)),
                "time": fwd_time,
                "backward_time": 0.0,
                "iter_time": fwd_time,
                "overhead_time": float(match.group(4)),
                "time_exclude": float(match.group(5)),
                "mfu": float(match.group(6)),
            }

    return results


def parse_total_runtime(log_lines):
    runtime_pattern = r"total runtime exclude: ([\d\.]+) seconds"
    for line in log_lines:
        match = re.search(runtime_pattern, line)
        if match:
            return float(match.group(1))
    return None


def calculate_speedup(log_paths):
    runtimes = []
    for log_path in log_paths:
        log_lines = read_log_file(log_path)
        runtime = parse_total_runtime(log_lines)
        if runtime is None:
            raise ValueError(f"Could not find runtime information in {log_path}")
        runtimes.append(runtime)
        print(f"Runtime from {log_path}: {runtime} seconds")
    speedup = runtimes[0] / runtimes[1]
    print(f"Speedup: {runtimes[0]:.2f} / {runtimes[1]:.2f} = {speedup:.2f}x")
    return speedup


def _plot_curve(results_list, labels, metric, title, y_label, save_path):
    plt.figure(figsize=(10, 6), dpi=100)
    colors = ["blue", "red"]
    for i, results in enumerate(results_list):
        iterations = sorted(results.keys())
        values = [results[it][metric] for it in iterations]
        plt.plot(iterations, values, linestyle="-", color=colors[i],
                 label=labels[i], linewidth=1)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="best")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return plt.gcf()


def process_and_plot_logs(log_paths, labels, save_path_prefix):
    results_list = []
    for log_path in log_paths:
        log_lines = read_log_file(log_path)
        results_list.append(parse_training_logs(log_lines))

    _plot_curve(results_list, labels, "loss",
                "Training Loss vs Iteration", "Loss",
                f"{save_path_prefix}_loss.png")
    _plot_curve(results_list, labels, "time",
                "Forward Time per Iteration", "Forward Time (ms)",
                f"{save_path_prefix}_fwd_time.png")
    _plot_curve(results_list, labels, "iter_time",
                "Iteration Time (Forward + Backward)", "Iter Time (ms)",
                f"{save_path_prefix}_iter_time.png")

    return results_list


if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)

    # --- Flash attention comparison (layernorm only) ---
    print("=== Flash Attention: PyTorch vs Oink ===")
    log_paths = ["logs/log.txt", "logs/oink_log.txt"]
    labels = ["PyTorch Aten", "Oink CuTeDSL"]
    process_and_plot_logs(log_paths, labels, save_path_prefix="figs/oink_comparison")
    calculate_speedup(log_paths)

    # --- No-flash comparison (layernorm + softmax) ---
    print("\n=== No Flash Attention: PyTorch vs Oink ===")
    log_paths_nf = ["logs/noflash_log.txt", "logs/oink_noflash_log.txt"]
    labels_nf = ["PyTorch Aten (no flash)", "Oink CuTeDSL (no flash)"]
    process_and_plot_logs(log_paths_nf, labels_nf, save_path_prefix="figs/oink_noflash_comparison")
    calculate_speedup(log_paths_nf)

    plt.show()
