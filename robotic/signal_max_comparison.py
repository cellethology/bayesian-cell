"""
Signal max comparison study for EKF configurations.
Compares Standard EKF, Adaptive Process EKF, and Adaptive Measurement EKF across different signal_max values.
"""

import numpy as np
import pandas as pd
from comparison import EKFComparison
from base_config import get_base_config, get_method_configs, get_signal_max_study_config


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
        if not inspect_trajectory:
            results = comparison.run_comparison(
                n_runs=n_runs, max_steps=max_steps, verbose=False
            )

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

            # Print the results nicely
            print(f"  Paired observations: {test_results['n_pairs']}")
            print(f"  Mean difference: {test_results['mean_difference']:.1f}")
            print(f"  t-statistic: {test_results['t_statistic']:.3f}")
            print(f"  p-value: {test_results['p_value']:.6f}")
            print(f"  Cohen's d: {test_results['cohens_d']:.3f}")

        else:
            print("Generating trajectory comparison for first two methods...")
            method_names = list(method_configs.keys())
            trajectory_data = comparison.run_trajectory_comparison(
                method_names[0], method_names[1], n_runs=1, max_steps=max_steps, seed=48
            )
            comparison.plot_trajectory_comparison_separate(
                trajectory_data,
                run_index=0,
                step_size=1,
                save_path=f"output/trajectory_{method_names[0]}_vs_{method_names[1]}.png",
            )

    # Save all results to CSV
    if not inspect_trajectory:
        results_filename = "output/signal_max_comparison_results.csv"
        combined_results.to_csv(results_filename, index=False)
        print(f"All results saved to {results_filename}")


if __name__ == "__main__":
    run_signal_max_comparison(inspect_trajectory=False)
