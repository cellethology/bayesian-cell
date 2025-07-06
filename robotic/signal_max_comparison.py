"""
Signal max comparison study for EKF configurations.
Compares Standard EKF, Adaptive Process EKF, and Adaptive Measurement EKF across different signal_max values.
"""

import numpy as np
import pandas as pd
from ekf_comparison import EKFComparison
from base_config import get_base_config, get_method_configs, get_signal_max_study_config


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

    # Load configurations from base_config.py
    base_config = get_base_config()
    method_configs = get_method_configs()
    study_config = get_signal_max_study_config()

    # Extract study parameters
    signal_decay = study_config["signal_decay"]
    n_runs = study_config["n_runs"]
    max_steps = study_config["max_steps"]
    signal_max_values = np.array(study_config["signal_max_values"])

    # Update base config with study-specific parameters
    base_config.update(
        {
            "signal_decay": signal_decay,
            "max_steps": max_steps,
        }
    )

    print(f"Signal decay: {signal_decay}")
    print(f"Signal max values: {signal_max_values}")
    print(f"Runs per configuration: {n_runs}")
    print(f"Total simulations: {len(signal_max_values) * len(method_configs) * n_runs}")

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

    combined_results = pd.concat(all_results, ignore_index=True)

    # Save all results to CSV
    results_filename = "signal_max_comparison_results.csv"
    combined_results.to_csv(results_filename, index=False)
    print(f"All results saved to {results_filename}")

    return combined_results


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Run the complete comparison study
    results = run_signal_max_comparison()

    print(f"\n{'='*60}")
    print("SIGNAL MAX COMPARISON COMPLETE")
    print(f"{'='*60}")
