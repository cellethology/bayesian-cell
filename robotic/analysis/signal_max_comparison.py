"""
Signal max comparison study for EKF configurations.
Compares Standard EKF, Adaptive Process EKF, and Adaptive Measurement EKF across different signal_max values.
"""

# Handle imports properly for both module and direct execution
if __name__ == "__main__":
    # When run directly, add parent directory to path
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from analysis.comparison import FilterComparison
from core import get_base_config, get_method_configs, get_signal_max_study_config


def run_signal_max_comparison(inspect_trajectory=False):
    """
    Run comparison of configurations across different signal_max values.
    """

    # Load configurations from base_config.py
    base_config = get_base_config()
    method_configs = get_method_configs()
    study_config = get_signal_max_study_config()

    # Extract study parameters
    signal_decay = study_config["signal_decay"]
    n_runs = study_config["n_runs"]
    signal_max_values = np.array(study_config["signal_max_values"])

    # Update base config with study-specific parameters
    base_config.update(
        {
            "signal_decay": signal_decay,
        }
    )

    print(f"Signal decay: {signal_decay}")
    print(f"Signal max values: {signal_max_values}")

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
        comparison = FilterComparison(current_base_config)

        # Add all three configurations
        for method_name, method_config in method_configs.items():
            comparison.add_config(method_name, method_config)

        # Run comparison
        if not inspect_trajectory:
            results = comparison.run_comparison(n_runs=n_runs, verbose=False)

            # Add signal_max column to results
            results["signal_max"] = signal_max
            results["method"] = results["config_name"]

            # Store results
            all_results.append(results)

            combined_results = pd.concat(all_results, ignore_index=True)

            # Perform paired t-test: Signal-aware EKF vs Standard EKF
            print(f"\nPaired t-test for signal_max = {signal_max}:")
            test_results = comparison.perform_paired_ttest(
                results, "Signal-aware EKF", "Standard EKF"
            )

            # Print the results
            print(f"  Paired observations: {test_results['n_pairs']}")
            print(f"  Mean difference: {test_results['mean_difference']:.1f}")
            print(f"  p-value: {test_results['p_value']:.6f}")

        else:
            print("Generating trajectory comparison...")
            method_names = list(method_configs.keys())
            trajectory_data = comparison.run_comparison(
                n_runs=1,
                collect_trajectories=True,
                seed=3,
            )
            comparison.plot_trajectory_comparison_separate(
                trajectory_data,
                run_index=0,
                step_size=3,
                save_path=f"output/trajectory_comparison.pdf",
                config_names=method_names,
                with_poisson_noise=True,
                spatial_correlation_length=15,
                spatial_correlation_strength=0.5,
            )

    # Save all results to CSV
    if not inspect_trajectory:
        results_filename = "output/signal_max_comparison_results.csv"
        combined_results.to_csv(results_filename, index=False)
        print(f"All results saved to {results_filename}")


if __name__ == "__main__":
    run_signal_max_comparison(inspect_trajectory=False)
