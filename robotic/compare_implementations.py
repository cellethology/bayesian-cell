"""
Compare performance of old vs current navigation implementations across multiple seeds.
"""

import numpy as np
import sys
import os

# Add current directory to path to import modules
sys.path.append(os.getcwd())

from bayes_navigation_old import run_navigation_simulation as run_old
from navigation_env import run_navigation_simulation as run_new


def run_comparison(num_seeds=20, max_steps=10000):
    """Run both implementations with multiple seeds and compare results."""

    # Shared configuration
    config = {
        "grid_size": 100,
        "signal_strength_max": 0.2,
        "signal_decay_exp": 0.3,
        "kernel_size": 5,
        "target_reach_threshold": 5.0,
        "motion_decay_rate": 0.8,
    }

    # Old implementation config
    old_config = config.copy()
    old_config.update(
        {
            "true_motion_sigma": 0.5,
            "min_motion_sigma": 0.5,
            "max_motion_sigma": 0.5,
            "measurement_noise_factor": 0.001,
            "movement_step_size": 0.2,
        }
    )

    # New implementation config
    new_config = config.copy()
    new_config.update(
        {
            "motion_noise_type": "angular",
            "angular_noise_sigma": 0.5,
            "magnitude_noise_sigma": 0.0,
            "initial_process_sigma": 0.5,
            "step_size": 0.2,
            "adaptive_filtering": False,
            "adaptive_process_variance": "none",
            "noise_model": "gaussian",
            "noise_std": 0.001,
            "initial_measurement_sigma": 0.001,
        }
    )

    old_steps = []
    new_steps = []
    old_reached = []
    new_reached = []

    print("Running comparison across", num_seeds, "seeds...")
    print("Seed | Old Steps | New Steps | Old Reached | New Reached")
    print("-" * 55)

    for seed in range(num_seeds):
        # Run old implementation
        np.random.seed(seed)
        try:
            old_trajectory, old_env, old_sigmas = run_old(old_config, steps=max_steps)
            old_step_count = len(old_trajectory) - 1  # Subtract initial position
            old_target_reached = old_step_count < max_steps
        except Exception as e:
            print(f"Old implementation failed at seed {seed}: {e}")
            old_step_count = max_steps
            old_target_reached = False

        # Run new implementation
        np.random.seed(seed)
        try:
            (
                new_trajectory,
                new_env,
                new_sigmas,
                new_innovations,
                new_measurement_variances,
            ) = run_new(new_config, steps=max_steps, verbose=False)
            new_step_count = len(new_trajectory) - 1  # Subtract initial position
            new_target_reached = new_step_count < max_steps
        except Exception as e:
            print(f"New implementation failed at seed {seed}: {e}")
            new_step_count = max_steps
            new_target_reached = False

        old_steps.append(old_step_count)
        new_steps.append(new_step_count)
        old_reached.append(old_target_reached)
        new_reached.append(new_target_reached)

        print(
            f"{seed:4d} | {old_step_count:9d} | {new_step_count:9d} | {old_target_reached:11} | {new_target_reached:11}"
        )

    # Calculate statistics
    old_steps = np.array(old_steps)
    new_steps = np.array(new_steps)
    old_reached = np.array(old_reached)
    new_reached = np.array(new_reached)

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    print(f"Success Rate:")
    print(f"  Old: {np.mean(old_reached)*100:.1f}% ({np.sum(old_reached)}/{num_seeds})")
    print(f"  New: {np.mean(new_reached)*100:.1f}% ({np.sum(new_reached)}/{num_seeds})")

    # Only analyze successful runs for step count
    old_successful_steps = old_steps[old_reached]
    new_successful_steps = new_steps[new_reached]

    if len(old_successful_steps) > 0:
        print(f"\nSteps to Target (successful runs only):")
        print(
            f"  Old: {np.mean(old_successful_steps):.1f} ± {np.std(old_successful_steps):.1f}"
        )
        print(
            f"      Min: {np.min(old_successful_steps)}, Max: {np.max(old_successful_steps)}"
        )

    if len(new_successful_steps) > 0:
        print(
            f"  New: {np.mean(new_successful_steps):.1f} ± {np.std(new_successful_steps):.1f}"
        )
        print(
            f"      Min: {np.min(new_successful_steps)}, Max: {np.max(new_successful_steps)}"
        )

    # Statistical test if both have successful runs
    if len(old_successful_steps) > 0 and len(new_successful_steps) > 0:
        from scipy import stats

        t_stat, p_value = stats.ttest_ind(old_successful_steps, new_successful_steps)
        print(f"\nStatistical Test (t-test):")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.3f}")
        if p_value < 0.05:
            print(f"  Significant difference detected (p < 0.05)")
        else:
            print(f"  No significant difference (p >= 0.05)")

    print(f"\nAll Steps (including failures):")
    print(f"  Old: {np.mean(old_steps):.1f} ± {np.std(old_steps):.1f}")
    print(f"  New: {np.mean(new_steps):.1f} ± {np.std(new_steps):.1f}")


if __name__ == "__main__":
    run_comparison(num_seeds=20, max_steps=10000)
