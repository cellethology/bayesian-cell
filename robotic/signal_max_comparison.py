"""
Signal max comparison study for EKF configurations.
Compares Standard EKF, Adaptive Process EKF, and Adaptive Measurement EKF across different signal_max values.
"""

import numpy as np
import pandas as pd
from ekf_comparison import EKFComparison


def bootstrap_median_ci(data, n_bootstrap=1000, confidence=0.95):
    """
    Calculate bootstrap confidence interval for median.

    Args:
        data: Array-like data
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0.95 for 95% CI)

    Returns:
        tuple: (median, ci_lower, ci_upper)
    """
    if len(data) == 0:
        return np.nan, np.nan, np.nan

    data = np.array(data)
    median_val = np.median(data)

    # Bootstrap samples
    bootstrap_medians = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_medians.append(np.median(bootstrap_sample))

    # Calculate confidence interval
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_medians, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_medians, 100 * (1 - alpha / 2))

    return median_val, ci_lower, ci_upper


def run_signal_max_comparison():
    """
    Run comparison of three EKF configurations across different signal_max values.
    Fixed signal_decay = 0.05, varying signal_max from 5 to 50.
    """
    print("=== Signal Max Comparison Study ===")

    # Fixed parameters
    signal_decay = 0.05
    n_runs = 20  # Reasonable number for bootstrap CI
    max_steps = 500000

    # Variable parameter: signal_max values
    signal_max_values = np.linspace(5, 5, 1)

    print(f"Fixed signal_decay: {signal_decay}")
    print(f"Signal max values: {signal_max_values}")
    print(f"Number of signal_max values: {len(signal_max_values)}")
    print(f"Runs per configuration: {n_runs}")
    print(f"Total simulations: {len(signal_max_values) * 3 * n_runs}")

    # Base configuration (shared parameters)
    base_config = {
        "filter_type": "FilterPy_EKF_Corrected",
        "arena_min": -1000.0,
        "arena_max": 1000.0,
        "distance_tolerance": 5.0,
        "signal_decay": signal_decay,  # Fixed
        "robot_start_pos": [-20.0, -20.0],
        "robot_step_size": 0.3,
        "actuator_noise": 0.5,
        "target_true_pos": [20.0, 20.0],
        "initial_belief_mean": [0.0, 0.0],
        "initial_belief_variance": 1000.0,
        "target_motion_sigma": 0.5,
        "baseline_process_noise": 0.5,
        "alpha_R": 0.5,
        "adaptive_measurement_noise": False,
        "eps": 1.0,  # Epsilon parameter for adaptive process noise
        "max_steps": max_steps,
    }

    # Three configurations to compare
    method_configs = {
        "Standard EKF": {
            "adaptive_process_noise": False,
            "adaptive_measurement_noise": False,
            "periodic_boundaries": False,
        },
        "Signal-aware EKF": {
            "adaptive_process_noise": True,
            "adaptive_measurement_noise": False,
            "periodic_boundaries": True,
        },
        "Adaptive EKF": {
            "adaptive_process_noise": False,
            "adaptive_measurement_noise": True,
            "periodic_boundaries": True,
        },
    }

    # Storage for all results
    all_results = []

    # Run experiments for each signal_max value
    for i, signal_max in enumerate(signal_max_values):
        print(f"\n{'='*60}")
        print(f"Running signal_max = {signal_max:.1f} ({i+1}/{len(signal_max_values)})")
        print(f"{'='*60}")

        # Update base config with current signal_max
        current_base_config = base_config.copy()
        current_base_config["signal_max"] = signal_max

        # Create comparison for this signal_max value
        comparison = EKFComparison(current_base_config)

        # Add all three configurations
        for method_name, method_config in method_configs.items():
            comparison.add_config(method_name, method_config)

        # Run comparison
        results = comparison.run_comparison(
            n_runs=n_runs, max_steps=max_steps, verbose=False
        )

        # Add signal_max column to results
        results["signal_max"] = signal_max
        results["method"] = results["config_name"]

        # Store results
        all_results.append(results)

        # Print quick summary for this signal_max value
        print(f"\nSummary for signal_max = {signal_max:.1f}:")
        for method in method_configs.keys():
            method_data = results[results["method"] == method]
            successful_data = method_data[method_data["target_reached"] == True]

            if len(successful_data) > 0:
                median_steps, ci_lower, ci_upper = bootstrap_median_ci(
                    successful_data["steps_to_target"].values
                )
                success_rate = method_data["target_reached"].mean()
                print(
                    f"  {method}: {median_steps:.0f} steps (95% CI: [{ci_lower:.0f}, {ci_upper:.0f}]), {success_rate:.1%} success"
                )
            else:
                print(f"  {method}: No successful runs")

    # Combine all results
    print(f"\n{'='*60}")
    print("COMBINING RESULTS AND CREATING PLOT")
    print(f"{'='*60}")

    combined_results = pd.concat(all_results, ignore_index=True)

    # Save all results to CSV
    results_filename = "signal_max_comparison_results.csv"
    combined_results.to_csv(results_filename, index=False)
    print(f"All results saved to {results_filename}")

    print(f"Data collection complete! Use plot_signal_max_results.py to create plots.")

    return combined_results


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Run the complete comparison study
    results = run_signal_max_comparison()

    print(f"\n{'='*60}")
    print("SIGNAL MAX COMPARISON COMPLETE")
    print(f"{'='*60}")
    print("Files created:")
    print("  - signal_max_comparison_results.csv (all raw results)")
    print("\nTo create plots, run:")
    print("  python plot_signal_max_results.py")
    print("\nOr import and use specific plotting functions:")
    print("  from plot_signal_max_results import quick_plot")
    print("  quick_plot('signal_max_comparison_results.csv', 'median')")
