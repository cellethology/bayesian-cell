"""
Signal decay comparison study for three Kalman Filter configurations.
Compares Standard KF, Signal-aware KF, and Adaptive KF across different signal decay values.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from histogram_filter.multi_config_comparison import MultiConfigComparison
import seaborn as sns


def run_signal_decay_comparison():
    """
    Run comparison of three KF configurations across different signal decay values.
    Fixed signal_max = 200, varying signal_decay from 0.01 to 1.5.
    """
    print("=== Signal Decay Comparison Study ===")

    # Fixed parameters
    signal_decay = 0.05
    n_runs = 400
    max_steps = 2000000

    # Variable parameter: signal_decay values
    signal_max_values = np.linspace(5, 50, 10)

    print(f"Fixed signal_decay: {signal_decay}")
    print(f"Signal max values: {signal_max_values}")
    print(f"Number of max values: {len(signal_max_values)}")
    print(f"Runs per configuration: {n_runs}")
    print(f"Total simulations: {len(signal_max_values) * 3 * n_runs}")

    # Base configuration (shared parameters)
    base_config = {
        "grid_size": 100,
        "target_motion_sigma": 0.5,
        "motion_noise_type": "isotropic",
        "process_sigma": 0.3,
        "signal_decay": signal_decay,  # Fixed
        "step_size": 0.2,
        "kernel_size": 5,
        "noise_model": "poisson",
    }

    # Three configurations to compare
    method_configs = {
        "Standard KF": {
            "adaptive_process_variance": False,
            "adaptive_filtering": False,
        },
        "Signal-aware KF": {
            "adaptive_process_variance": True,
            "adaptive_decay_type": "power_law",
            "adaptive_rate": 1,
            "power_exponent": 0.5,
            "adaptive_filtering": False,
        },
        "Adaptive KF": {
            "adaptive_process_variance": False,
            "adaptive_filtering": True,
        },
    }

    # Storage for all results
    all_results = []

    # Run experiments for each signal_decay value
    for i, signal_max in enumerate(signal_max_values):
        print(f"\n{'='*60}")
        print(f"Running signal_max = {signal_max:.4f} ({i+1}/{len(signal_max_values)})")
        print(f"{'='*60}")

        # Update base config with current signal_decay
        current_base_config = base_config.copy()
        current_base_config["signal_max"] = signal_max

        # Create comparison for this signal_decay value
        comparison = MultiConfigComparison(current_base_config)

        # Add all three configurations
        for method_name, method_config in method_configs.items():
            comparison.add_config(method_name, method_config)

        # Run comparison
        results = comparison.run_comparison(
            n_runs=n_runs, max_steps=max_steps, verbose=False
        )

        # Add signal_decay column to results
        results["signal_max"] = signal_max
        results["method"] = results["config_name"]

        # Store results
        all_results.append(results)

        # Print quick summary for this decay value
        print(f"\nSummary for signal_max = {signal_max:.4f}:")
        for method in method_configs.keys():
            method_data = results[results["method"] == method]
            mean_steps = method_data["steps_to_target"].mean()
            success_rate = method_data["target_reached"].mean()
            print(
                f"  {method}: {mean_steps:.0f} ± {method_data['steps_to_target'].std():.0f} steps, {success_rate:.1%} success"
            )

    # Combine all results
    print(f"\n{'='*60}")
    print("COMBINING RESULTS AND CREATING PLOTS")
    print(f"{'='*60}")

    combined_results = pd.concat(all_results, ignore_index=True)

    # Save all results to CSV
    results_filename = "signal_decay_comparison_results.csv"
    combined_results.to_csv(results_filename, index=False)
    print(f"All results saved to {results_filename}")

    # Create summary statistics for plotting
    summary_stats = (
        combined_results.groupby(["signal_max", "method"])
        .agg({"steps_to_target": ["mean", "std", "count"], "target_reached": "mean"})
        .reset_index()
    )

    # Flatten column names
    summary_stats.columns = [
        "signal_max",
        "method",
        "mean_steps",
        "std_steps",
        "count",
        "success_rate",
    ]

    # Calculate standard error for error bars
    summary_stats["sem_steps"] = summary_stats["std_steps"] / np.sqrt(
        summary_stats["count"]
    )

    # Save summary statistics
    summary_filename = "signal_decay_summary_stats.csv"
    summary_stats.to_csv(summary_filename, index=False)
    print(f"Summary statistics saved to {summary_filename}")

    print(
        "\nData collection complete! Use plot_signal_decay_results.py to create plots."
    )

    return combined_results, summary_stats


if __name__ == "__main__":
    # Run the complete comparison study
    results, summary = run_signal_decay_comparison()

    print(f"\n{'='*60}")
    print("DATA COLLECTION COMPLETE")
    print(f"{'='*60}")
    print("Files created:")
    print("  - signal_decay_comparison_results.csv (all raw results)")
    print("  - signal_decay_summary_stats.csv (summary statistics)")
    print("\nTo create plots, run:")
    print("  python plot_signal_decay_results.py")
    print("\nOr import and use specific plotting functions:")
    print("  from plot_signal_decay_results import quick_plot")
    print("  quick_plot('signal_decay_comparison_results.csv', 'main')")
