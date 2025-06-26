"""
Advanced multi-configuration comparison tool for Bayesian navigation.
Provides easy-to-use interface for comparing arbitrary sets of configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
from navigation_env import run_navigation_simulation
import pandas as pd
from typing import Dict, Any, Optional
import os
from scipy import stats


def run_single_config_simulation(args):
    """Run a single configuration simulation (for multiprocessing)."""
    seed, config_name, config, max_steps, verbose = args

    np.random.seed(seed)
    trajectory, env, _, _, _ = run_navigation_simulation(
        config=config, steps=max_steps, verbose=verbose
    )

    # Final distance to target
    true_target = env.config.get("true_target_pos", (0, 0))
    final_distance = np.linalg.norm(trajectory[-1] - np.array(true_target))
    target_reached = final_distance < env.config.get("target_reach_threshold", 5.0)

    # Calculate mean signal of entire environment
    signal_grid = env.signal_model.compute_all_expected_signal(true_target)
    mean_signal = np.mean(signal_grid)

    results = {
        "config_name": config_name,
        "seed": seed,
        "steps_to_target": len(trajectory),
        "target_reached": target_reached,
        "mean_signal": mean_signal,
    }

    return results


class MultiConfigComparison:
    """
    Easy-to-use multi-configuration comparison tool.

    Usage:
        comparison = MultiConfigComparison()
        comparison.add_config("Method A", config_a)
        comparison.add_config("Method B", config_b)
        results = comparison.run_comparison(n_runs=20)
        comparison.create_comparison_plots(results)
    """

    def __init__(self, base_config: Optional[Dict] = None):
        """
        Initialize comparison tool.

        Args:
            base_config: Base configuration that will be shared across all methods
        """
        self.base_config = base_config or {}
        self.configs = {}
        self.config_order = []  # Track order of added configs

    def add_config(self, name: str, config: Dict[str, Any]):
        """
        Add a configuration to compare.

        Args:
            name: Human-readable name for this configuration
            config: Configuration dictionary (will be merged with base_config)
        """
        full_config = self.base_config.copy()
        full_config.update(config)
        self.configs[name] = full_config
        if name not in self.config_order:
            self.config_order.append(name)

    def run_comparison(
        self,
        n_runs: int = 20,
        max_steps: int = 10000,
        verbose: bool = False,
        n_processes: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Run comparison across all added configurations.

        Args:
            n_runs: Number of runs per configuration (same seeds used for all configs)
            max_steps: Maximum steps per simulation
            verbose: Whether to show simulation progress
            n_processes: Number of processes (None = auto)

        Returns:
            DataFrame with all results
        """
        if not self.configs:
            raise ValueError("No configurations added. Use add_config() first.")

        print(
            f"Running comparison of {len(self.configs)} configurations with {n_runs} runs each..."
        )

        # Create all tasks
        tasks = []
        seeds = list(range(n_runs))  # Same seeds for all configs for fair comparison

        for seed in seeds:
            for config_name, config in self.configs.items():
                tasks.append((seed, config_name, config, max_steps, verbose))

        # Run simulations in parallel
        if n_processes is None:
            n_processes = min(len(tasks), os.cpu_count())  # Reasonable default

        results = []
        with Pool(n_processes) as pool:
            for result in tqdm(
                pool.imap_unordered(run_single_config_simulation, tasks),
                total=len(tasks),
                desc="Running simulations",
            ):
                results.append(result)

        # Convert to DataFrame for easy analysis
        df = pd.DataFrame(results)

        print(f"\\nComparison complete! {len(results)} simulations finished.")
        self._print_summary_statistics(df)

        return df

    def _print_summary_statistics(self, df: pd.DataFrame):
        """Print summary statistics for each configuration."""
        print("=" * 60)
        print("CONFIGURATION COMPARISON SUMMARY")
        print("=" * 60)

        for config_name in df["config_name"].unique():
            config_data = df[df["config_name"] == config_name]

            print(f"\n{config_name}:")
            print(f"  Success Rate: {config_data['target_reached'].mean():.2%}")
            q25 = config_data["steps_to_target"].quantile(0.25)
            q50 = config_data["steps_to_target"].median()
            q75 = config_data["steps_to_target"].quantile(0.75)
            print(f"  Median Steps: {q50:.0f} (Q1: {q25:.0f}, Q3: {q75:.0f})")
            # print mean steps and std to target for successful runs
            successful_data = config_data[config_data["target_reached"] == True]
            print(
                f"  Mean Steps to Target (Successful Runs): {successful_data['steps_to_target'].mean():.0f} ± {successful_data['steps_to_target'].std():.0f}"
            )
            mean_signal_avg = config_data["mean_signal"].mean()
            mean_signal_std = config_data["mean_signal"].std()
            print(f"  Mean Signal: {mean_signal_avg:.4f} ± {mean_signal_std:.4f}")

    def create_comparison_plots(self, df: pd.DataFrame, figsize: tuple = (12, 5)):
        """
        Create comparison plots for success rate and completion time.

        Args:
            df: Results DataFrame from run_comparison()
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Use preserved config order instead of df unique order
        config_names = [
            name for name in self.config_order if name in df["config_name"].values
        ]
        colors = plt.cm.Set2(np.linspace(0, 1, len(config_names)))

        # 1. Success rate comparison
        success_rates = [
            df[df["config_name"] == name]["target_reached"].mean()
            for name in config_names
        ]
        bars = ax1.bar(config_names, success_rates, color=colors, alpha=0.8)
        ax1.set_ylabel("Success Rate")
        ax1.set_title("Target Achievement Rate")
        ax1.set_ylim(0, 1.1)
        for bar, rate in zip(bars, success_rates):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{rate:.1%}",
                ha="center",
                va="bottom",
            )
        ax1.grid(True, alpha=0.3, axis="y")

        # 2. Completion time for successful runs only
        successful_data = []
        labels = []

        for name in config_names:
            config_data = df[df["config_name"] == name]
            successful_runs = config_data[config_data["target_reached"] == True]
            if len(successful_runs) > 0:
                successful_data.append(successful_runs["steps_to_target"].values)
                labels.append(name)

        if successful_data:
            bp = ax2.boxplot(successful_data, tick_labels=labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors[: len(labels)]):
                patch.set_facecolor(color)
            ax2.set_ylabel("Steps to Target")
            ax2.set_title("Completion Time (Successful Runs)")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(
                0.5,
                0.5,
                "No successful runs",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
            ax2.set_title("Completion Time (No Successful Runs)")

        plt.tight_layout()
        return fig

    def export_results(self, df: pd.DataFrame, filename: str):
        """Export results to CSV file."""
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")

    def perform_paired_ttest(
        self,
        df: pd.DataFrame,
        config1: str,
        config2: str,
        metric: str = "steps_to_target",
    ):
        """
        Perform paired t-test between two configurations using the same seeds.

        Args:
            df: Results DataFrame from run_comparison()
            config1: Name of first configuration (baseline)
            config2: Name of second configuration (comparison)
            metric: Metric to compare (default: "steps_to_target")

        Returns:
            dict: Test results including statistic, p-value, and interpretation
        """
        # Get data for both configurations
        data1 = df[df["config_name"] == config1].sort_values("seed")
        data2 = df[df["config_name"] == config2].sort_values("seed")

        # Ensure we have the same seeds for pairing
        common_seeds = set(data1["seed"]) & set(data2["seed"])
        if len(common_seeds) == 0:
            raise ValueError(f"No common seeds found between {config1} and {config2}")

        # Filter to common seeds and sort by seed
        data1_paired = data1[data1["seed"].isin(common_seeds)].sort_values("seed")
        data2_paired = data2[data2["seed"].isin(common_seeds)].sort_values("seed")

        # Extract the metric values
        values1 = data1_paired[metric].values
        values2 = data2_paired[metric].values

        # Perform paired t-test
        # H0: mean difference = 0
        # H1: config2 has lower values than config1 (one-tailed test)
        statistic, p_value_two_tailed = stats.ttest_rel(values1, values2)

        # For one-tailed test (config2 < config1), divide p-value by 2 if statistic > 0
        p_value_one_tailed = (
            p_value_two_tailed / 2 if statistic > 0 else 1 - (p_value_two_tailed / 2)
        )

        # Calculate effect size (Cohen's d for paired samples)
        differences = values1 - values2
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0

        # Summary statistics
        mean1 = np.mean(values1)
        mean2 = np.mean(values2)
        std1 = np.std(values1, ddof=1)
        std2 = np.std(values2, ddof=1)

        results = {
            "config1": config1,
            "config2": config2,
            "metric": metric,
            "n_pairs": len(common_seeds),
            "mean1": mean1,
            "std1": std1,
            "mean2": mean2,
            "std2": std2,
            "mean_difference": mean_diff,
            "std_difference": std_diff,
            "t_statistic": statistic,
            "p_value_two_tailed": p_value_two_tailed,
            "p_value_one_tailed": p_value_one_tailed,
            "cohens_d": cohens_d,
            "significant_two_tailed": p_value_two_tailed < 0.05,
            "significant_one_tailed": p_value_one_tailed < 0.05,
        }

        return results

    def print_ttest_results(self, test_results: dict):
        """Print formatted t-test results."""
        print("\n" + "=" * 60)
        print("PAIRED T-TEST RESULTS")
        print("=" * 60)
        print(
            f"Comparing: {test_results['config2']} vs {test_results['config1']} (baseline)"
        )
        print(f"Metric: {test_results['metric']}")
        print(f"Number of paired observations: {test_results['n_pairs']}")
        print()
        print("Summary Statistics:")
        print(
            f"  {test_results['config1']}: {test_results['mean1']:.2f} ± {test_results['std1']:.2f}"
        )
        print(
            f"  {test_results['config2']}: {test_results['mean2']:.2f} ± {test_results['std2']:.2f}"
        )
        print(
            f"  Mean difference: {test_results['mean_difference']:.2f} ± {test_results['std_difference']:.2f}"
        )
        print()
        print("Test Results:")
        print(f"  t-statistic: {test_results['t_statistic']:.4f}")
        print(f"  p-value (two-tailed): {test_results['p_value_two_tailed']:.6f}")
        print(f"  p-value (one-tailed): {test_results['p_value_one_tailed']:.6f}")
        print(f"  Cohen's d (effect size): {test_results['cohens_d']:.4f}")
        print()
        print("Interpretation:")
        if test_results["significant_one_tailed"]:
            print(
                f"  ✓ {test_results['config2']} has significantly LOWER {test_results['metric']} than {test_results['config1']} (p < 0.05, one-tailed)"
            )
        else:
            print(
                f"  ✗ No significant difference found (p = {test_results['p_value_one_tailed']:.4f} >= 0.05, one-tailed)"
            )

        # Effect size interpretation
        abs_d = abs(test_results["cohens_d"])
        if abs_d < 0.2:
            effect_size = "negligible"
        elif abs_d < 0.5:
            effect_size = "small"
        elif abs_d < 0.8:
            effect_size = "medium"
        else:
            effect_size = "large"
        print(f"  Effect size: {effect_size} (|d| = {abs_d:.3f})")
        print("=" * 60)


# Convenience function for quick comparisons
def quick_compare(
    configs: Dict[str, Dict],
    base_config: Optional[Dict] = None,
    n_runs: int = 10,
    max_steps: int = 5000,
    **kwargs,
):
    """
    Quick comparison function for easy one-liner usage.

    Args:
        configs: Dictionary of {name: config} pairs
        base_config: Base configuration
        n_runs: Number of runs per configuration
        max_steps: Maximum steps per simulation
        **kwargs: Additional arguments for run_comparison()

    Returns:
        tuple: (results_dataframe, comparison_plots_figure)
    """
    comparison = MultiConfigComparison(base_config)

    for name, config in configs.items():
        comparison.add_config(name, config)

    results = comparison.run_comparison(n_runs=n_runs, max_steps=max_steps, **kwargs)
    plots = comparison.create_comparison_plots(results)

    return results, plots


if __name__ == "__main__":
    # Example usage
    print("=== Multi-Configuration Comparison Example ===")

    # Define base configuration (shared parameters)
    base_config = {
        "grid_size": 100,
        "motion_noise_type": "isotropic",
        "process_sigma": 0.4,
        "process_sigma_estimate": 0.4,
        "signal_max": 10,
        "signal_decay": 0.04,
        "step_size": 0.2,
        "kernel_size": 5,
        "adaptive_filtering": True,
        "noise_model": "poisson",
        "noise_estimate": 0.3,  # standard deviation
    }

    # Define configurations to compare
    configs_to_compare = {
        "No Adaptation": {
            "adaptive_process_variance": "none",
            "adaptive_filtering": True,
        },
        "Exponential": {
            "adaptive_process_variance": "exponential",
            "adaptive_rate": 0.8,
        },
    }

    # Run comparison
    comparison = MultiConfigComparison(base_config)
    for name, config in configs_to_compare.items():
        comparison.add_config(name, config)

    results = comparison.run_comparison(n_runs=500, max_steps=500000)
    plots = comparison.create_comparison_plots(results)

    # Perform paired t-test
    test_results = comparison.perform_paired_ttest(
        results,
        config1="No Adaptation",
        config2="Exponential",
        metric="steps_to_target",
    )
    comparison.print_ttest_results(test_results)

    plt.show()
