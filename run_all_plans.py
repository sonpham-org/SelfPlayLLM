#!/usr/bin/env python3
"""Run all experimental plans sequentially.

Plans A-D test different configurations of the evolutionary self-play framework.
Results are saved to separate run directories under runs/.

Total estimated runtime: ~12 hours (RTX 4090, Qwen 3.5 4B/27B).
"""

import subprocess
import sys
import time
import os

PYTHON = sys.executable

PLANS = [
    {
        "name": "Plan A: Validation (4B, sequential, connect4, 5 gens)",
        "args": [
            "--game", "connect4", "-g", "5", "-p", "10",
            "--run-name", "plan_a_validation",
        ],
    },
    {
        "name": "Plan B: Parallel Benchmark (4B, 8 workers, connect4, 10 gens)",
        "args": [
            "--game", "connect4", "-g", "10", "-p", "10", "-j", "8",
            "--run-name", "plan_b_parallel",
        ],
    },
    {
        "name": "Plan C-4B: Model Comparison (4B, 8 workers, connect4, 5 gens)",
        "args": [
            "--game", "connect4", "-g", "5", "-p", "10", "-j", "8",
            "--model", "qwen3.5:4b",
            "--run-name", "plan_c_4b",
        ],
    },
    {
        "name": "Plan C-27B: Model Comparison (27B, 4 workers, connect4, 5 gens)",
        "args": [
            "--game", "connect4", "-g", "5", "-p", "10", "-j", "4",
            "--model", "qwen3.5:27b",
            "--run-name", "plan_c_27b",
        ],
    },
    {
        "name": "Plan D-Othello: Multi-Game (4B, 8 workers, othello, 10 gens)",
        "args": [
            "--game", "othello", "-g", "10", "-p", "10", "-j", "8",
            "--run-name", "plan_d_othello",
        ],
    },
    {
        "name": "Plan D-Checkers: Multi-Game (4B, 8 workers, checkers, 10 gens)",
        "args": [
            "--game", "checkers", "-g", "10", "-p", "10", "-j", "8",
            "--run-name", "plan_d_checkers",
        ],
    },
]


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 70)
    print("  SELFPLAYLLM — ALL PLANS")
    print("=" * 70)
    print(f"\n  Running {len(PLANS)} experiments sequentially.\n")
    for i, plan in enumerate(PLANS):
        print(f"  {i+1}. {plan['name']}")
    print()

    results = []
    total_start = time.time()

    for i, plan in enumerate(PLANS):
        print(f"\n{'#' * 70}")
        print(f"  [{i+1}/{len(PLANS)}] {plan['name']}")
        print(f"{'#' * 70}\n")

        cmd = [PYTHON, "main.py"] + plan["args"]
        print(f"  Command: {' '.join(cmd)}\n")

        t0 = time.time()
        proc = subprocess.run(cmd, text=True)
        elapsed = time.time() - t0

        status = "OK" if proc.returncode == 0 else f"FAIL(exit={proc.returncode})"
        results.append((plan["name"], elapsed, status))

        print(f"\n  >>> {plan['name']}: {status} ({elapsed/60:.1f} min)\n")

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 70}")
    print("  ALL PLANS COMPLETE")
    print(f"{'=' * 70}")
    print(f"\n  Total wall time: {total_elapsed/3600:.1f} hours\n")
    print(f"  {'Status':<12} {'Time':>8}  Plan")
    print(f"  {'-'*50}")
    for name, elapsed, status in results:
        print(f"  {status:<12} {elapsed/60:>6.1f}m  {name}")
    print()

    # Save summary
    summary_path = os.path.join("runs", "all_plans_summary.txt")
    os.makedirs("runs", exist_ok=True)
    with open(summary_path, "w") as f:
        f.write(f"Total wall time: {total_elapsed/3600:.1f} hours\n\n")
        for name, elapsed, status in results:
            f.write(f"{status:<12} {elapsed/60:>6.1f}m  {name}\n")
    print(f"  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
