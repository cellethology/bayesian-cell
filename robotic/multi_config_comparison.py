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
        "process_sigma": 0.2,
        # "angular_noise_sigma": 0.3,
        # "magnitude_noise_sigma": 0.3,
        "process_sigma_estimate": 0.2,
        "signal_max": 10,
        "signal_decay": 0.02,
        "step_size": 0.2,
        "kernel_size": 5,
        "adaptive_filtering": False,
        "noise_model": "poisson",
        "noise_std": 0.06,
        "measurement_sigma_estimate": 0.01,
    }

    # Define configurations to compare
    configs_to_compare = {
        "No Adaptation": {
            "adaptive_process_variance": "none",
        },
        "Exponential": {
            "adaptive_process_variance": "exponential",
            "adaptive_rate": 0.8,
        },
    }

    # Quick comparison
    results, plots = quick_compare(
        configs_to_compare,
        base_config=base_config,
        n_runs=100,  # Small number for demo
        max_steps=500000,
    )

    plt.show()
