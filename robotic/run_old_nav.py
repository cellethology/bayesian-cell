#!/usr/bin/env python3
"""
Run bayes_navigation_old for 40 seeds and collect statistics.
"""

import numpy as np
from bayes_navigation_old import run_navigation_simulation
from tqdm import tqdm
from multiprocessing import Pool
import os


def run_single_simulation(seed):
    """Run a single simulation with given seed."""
    np.random.seed(seed)

    example_config = {
        "grid_size": 100,
        "true_motion_sigma": 0.5,
        "min_motion_sigma": 0.1,
        "max_motion_sigma": 0.5,
        "adaptive_rate": 0.8,  # Irrelevant when min == max
        "measurement_noise_factor": 0.06,
        "signal_max": 0.3,
        "signal_decay": 0.2,
        "movement_step_size": 0.2,
        "kernel_size": 5,
        "target_reach_threshold": 5.0,
    }

    trajectory, env, _ = run_navigation_simulation(config=example_config, steps=500000)

    # Check if target was reached
    true_target = env.true_target_pos
    final_pos = trajectory[-1]
    final_distance = np.linalg.norm(final_pos - np.array(true_target))
    target_reached = final_distance < env.config.get("target_reach_threshold", 5.0)

    return {
        "seed": seed,
        "steps_to_target": len(trajectory),
        "target_reached": target_reached,
        "final_distance": final_distance,
    }


def run_multiple_seeds(n_seeds=40, n_processes=None):
    """Run navigation simulation for multiple seeds and collect statistics."""
    print(f"Running bayes_navigation_old for {n_seeds} seeds with parallelization...")

    if n_processes is None:
        n_processes = min(n_seeds, os.cpu_count())

    seeds = list(range(n_seeds))

    # Run simulations in parallel
    with Pool(n_processes) as pool:
        results = list(
            tqdm(
                pool.imap(run_single_simulation, seeds),
                total=len(seeds),
                desc="Running simulations",
            )
        )

    # Calculate statistics
    successful_runs = [r for r in results if r["target_reached"]]
    success_rate = len(successful_runs) / len(results)

    if successful_runs:
        steps_successful = [r["steps_to_target"] for r in successful_runs]
        avg_steps = np.mean(steps_successful)
        std_steps = np.std(steps_successful)
    else:
        avg_steps = float("inf")
        std_steps = 0

    # Print results
    print("\n" + "=" * 50)
    print("BAYES_NAVIGATION_OLD RESULTS")
    print("=" * 50)
    print(
        f"Success Rate: {success_rate:.2%} ({len(successful_runs)}/{len(results)} runs)"
    )

    if successful_runs:
        print(f"Average Steps (successful runs): {avg_steps:.0f} ± {std_steps:.0f}")
    else:
        print("No successful runs completed")

    return results, success_rate, avg_steps, std_steps


if __name__ == "__main__":
    results, success_rate, avg_steps, std_steps = run_multiple_seeds(40)
