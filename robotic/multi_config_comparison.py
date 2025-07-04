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
from visualization import NavigationVisualizer


def run_single_config_simulation(args):
    """Run a single configuration simulation (for multiprocessing)."""
    seed, config_name, config, max_steps, verbose = args

    np.random.seed(seed)
    trajectory, target_trajectory, env, _, _, _, success_target_pos = (
        run_navigation_simulation(config=config, steps=max_steps, verbose=verbose)
    )

    # Final distance to target
    true_target = env.config.get("true_target_pos", (0, 0))
    final_distance = np.linalg.norm(trajectory[-1] - np.array(true_target))
    target_reached = final_distance < env.config.get("target_reach_threshold", 5.0)

    # Calculate mean signal of entire environment
    signal_grid = env.signal_model.compute_all_expected_signal(true_target)
    mean_signal = np.mean(signal_grid)

    # Calculate signal at starting position
    starting_position = trajectory[0]
    signal_at_start = env.signal_model.get_expected_signal(
        starting_position[0], starting_position[1], true_target
    )

    results = {
        "config_name": config_name,
        "seed": seed,
        "steps_to_target": len(trajectory),
        "target_reached": target_reached,
        "mean_signal": mean_signal,
        "signal_at_start": signal_at_start,
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

    def create_comparison_plots(
        self, df: pd.DataFrame, figsize: tuple = (12, 5), log_scale: bool = False
    ):
        """
        Create comparison plots for success rate and completion time.

        Args:
            df: Results DataFrame from run_comparison()
            figsize: Figure size
            log_scale: Whether to use log scale for the completion time plot
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

            # Apply log scale if requested
            if log_scale:
                ax2.set_yscale("log")
                ax2.set_ylabel("Steps to Target (log scale)")
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
        statistic, p_value_two_tailed = stats.ttest_rel(values1, values2)

        # Calculate one-tailed p-values for both directions
        # H1a: config2 < config1 (config2 is better - lower values)
        p_value_config2_better = (
            p_value_two_tailed / 2 if statistic > 0 else 1 - (p_value_two_tailed / 2)
        )

        # H1b: config1 < config2 (config1 is better - lower values)
        p_value_config1_better = (
            p_value_two_tailed / 2 if statistic < 0 else 1 - (p_value_two_tailed / 2)
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
            "p_value_config2_better": p_value_config2_better,
            "p_value_config1_better": p_value_config1_better,
            "cohens_d": cohens_d,
            "significant_two_tailed": p_value_two_tailed < 0.05,
            "significant_config2_better": p_value_config2_better < 0.05,
            "significant_config1_better": p_value_config1_better < 0.05,
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
        print(
            f"  p-value ({test_results['config2']} better): {test_results['p_value_config2_better']:.6f}"
        )
        print(
            f"  p-value ({test_results['config1']} better): {test_results['p_value_config1_better']:.6f}"
        )
        print(f"  Cohen's d (effect size): {test_results['cohens_d']:.4f}")
        print()
        print("Interpretation:")
        if test_results["significant_config2_better"]:
            print(
                f"  ✓ {test_results['config2']} has significantly LOWER {test_results['metric']} than {test_results['config1']} (p = {test_results['p_value_config2_better']:.4f} < 0.05, one-tailed)"
            )
        elif test_results["significant_config1_better"]:
            print(
                f"  ✓ {test_results['config1']} has significantly LOWER {test_results['metric']} than {test_results['config2']} (p = {test_results['p_value_config1_better']:.4f} < 0.05, one-tailed)"
            )
        else:
            print(f"  ✗ No significant difference found between configurations")
            print(
                f"    {test_results['config2']} better: p = {test_results['p_value_config2_better']:.4f}"
            )
            print(
                f"    {test_results['config1']} better: p = {test_results['p_value_config1_better']:.4f}"
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

    def run_trajectory_comparison(
        self,
        config1: str,
        config2: str,
        n_runs: int = 5,
        max_steps: int = 10000,
        seed: int = None,
    ):
        """
        Run trajectory comparison between two configurations.

        Args:
            config1: Name of first configuration
            config2: Name of second configuration
            n_runs: Number of runs to collect trajectories for
            max_steps: Maximum steps per simulation
            seed: Starting seed (optional)

        Returns:
            dict: Trajectories and environments for both configurations
        """
        if config1 not in self.configs or config2 not in self.configs:
            raise ValueError(f"Both {config1} and {config2} must be added configs")

        trajectories_data = {config1: [], config2: []}

        seeds = list(range(seed or 0, (seed or 0) + n_runs))

        for run_seed in seeds:
            # Generate target trajectory once for both configs to ensure consistency
            np.random.seed(run_seed)

            # Pre-generate target motion for max_steps (will be truncated as needed)
            target_motion_sigma = self.configs[config1].get("target_motion_sigma", 0.1)
            grid_size = self.configs[config1].get("grid_size", 100)

            # Generate enough random numbers for target motion
            target_random_x = np.random.normal(0, target_motion_sigma, max_steps)
            target_random_y = np.random.normal(0, target_motion_sigma, max_steps)

            # Store results for both configs with same seed
            seed_results = {}

            for config_name in [config1, config2]:
                # Use a different seed offset for robot/measurement randomness
                # This ensures target motion is same but other randomness can differ
                # Use a deterministic hash that's consistent across sessions
                import hashlib

                deterministic_hash = int(
                    hashlib.md5(config_name.encode()).hexdigest()[:8], 16
                )
                robot_seed = run_seed + deterministic_hash % 1000000
                np.random.seed(robot_seed)

                # Temporarily store target random arrays in config
                temp_config = self.configs[config_name].copy()
                temp_config["_target_random_x"] = target_random_x
                temp_config["_target_random_y"] = target_random_y
                temp_config["_target_random_index"] = 0  # Track current index

                trajectory, target_trajectory, env, _, _, _, success_target_pos = (
                    run_navigation_simulation(
                        config=temp_config, steps=max_steps, verbose=False
                    )
                )
                seed_results[config_name] = {
                    "trajectory": trajectory,
                    "target_trajectory": target_trajectory,
                    "env": env,
                    "seed": run_seed,
                    "success_target_pos": success_target_pos,
                }

            # Add both results to trajectories_data
            for config_name in [config1, config2]:
                trajectories_data[config_name].append(seed_results[config_name])

        return trajectories_data

    def plot_trajectory_comparison(
        self,
        trajectories_data: dict,
        n_plots: int = 4,
        figsize: tuple = (16, 4),
        save_path: str = None,
    ):
        """
        Plot trajectory comparisons with both methods on same subplot (same seed).

        Args:
            trajectories_data: Output from run_trajectory_comparison
            n_plots: Number of subplots to show (one per seed)
            figsize: Figure size
            save_path: Optional path to save figure

        Returns:
            matplotlib figure
        """
        config_names = list(trajectories_data.keys())
        if len(config_names) != 2:
            raise ValueError("Exactly two configurations required")

        config1, config2 = config_names
        n_runs = min(len(trajectories_data[config1]), len(trajectories_data[config2]))
        n_plots = min(n_plots, n_runs)

        # Calculate grid layout with max 4 plots per row
        max_cols = 4
        n_rows = (n_plots + max_cols - 1) // max_cols  # Ceiling division
        n_cols = min(n_plots, max_cols)

        # Adjust figure size based on grid layout
        plot_width = 4
        plot_height = 4
        figsize = (n_cols * plot_width, n_rows * plot_height)

        # Create subplots grid
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # Handle single plot case
        if n_plots == 1:
            axes = [axes]
        # Handle single row case
        elif n_rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        # Handle multi-row case
        else:
            axes = axes.flatten()

        for i in range(n_plots):
            ax = axes[i]

            # Get data for both configurations for the same seed
            traj_data1 = trajectories_data[config1][i]
            traj_data2 = trajectories_data[config2][i]

            trajectory1 = np.array(traj_data1["trajectory"])
            trajectory2 = np.array(traj_data2["trajectory"])
            target_trajectory1 = np.array(traj_data1["target_trajectory"])
            target_trajectory2 = np.array(traj_data2["target_trajectory"])
            success_pos1 = traj_data1["success_target_pos"]
            success_pos2 = traj_data2["success_target_pos"]
            env1 = traj_data1["env"]
            seed = traj_data1["seed"]

            # Use the first environment for the signal map
            signal_map = env1.signal_model.compute_all_expected_signal(
                env1.true_target_pos
            )

            # Plot signal map
            im = ax.imshow(signal_map, cmap="Greens", interpolation="nearest")

            # Plot both trajectories with different colors
            ax.plot(
                trajectory1[:, 1],
                trajectory1[:, 0],
                "b-",
                linewidth=2,
                label=f"{config1} ({len(trajectory1)} steps)",
                alpha=0.8,
            )
            ax.plot(
                trajectory2[:, 1],
                trajectory2[:, 0],
                "r-",
                linewidth=2,
                label=f"{config2} ({len(trajectory2)} steps)",
                alpha=0.8,
            )

            # Plot start position (should be the same for both)
            ax.plot(
                trajectory1[0, 1], trajectory1[0, 0], "bs", markersize=8, label="Start"
            )

            # Plot end positions with different markers
            ax.plot(
                trajectory1[-1, 1], trajectory1[-1, 0], "bo", markersize=6, alpha=0.8
            )
            ax.plot(
                trajectory2[-1, 1], trajectory2[-1, 0], "ro", markersize=6, alpha=0.8
            )

            # Verify that target trajectories are identical up to shorter length
            min_len = min(len(target_trajectory1), len(target_trajectory2))
            traj_match = np.allclose(
                target_trajectory1[:min_len], target_trajectory2[:min_len], atol=1e-10
            )

            # Additional debugging
            if min_len > 0:
                first_diff_idx = None
                for j in range(min_len):
                    if not np.allclose(
                        target_trajectory1[j], target_trajectory2[j], atol=1e-10
                    ):
                        first_diff_idx = j
                        break

                if first_diff_idx is not None:
                    print(
                        f"Seed {seed}: Target trajectories diverge at step {first_diff_idx}"
                    )
                    print(
                        f"  Traj1[{first_diff_idx}]: {target_trajectory1[first_diff_idx]}"
                    )
                    print(
                        f"  Traj2[{first_diff_idx}]: {target_trajectory2[first_diff_idx]}"
                    )

            # Add verification indicator to title if trajectories don't match as expected
            if not traj_match and min_len > 0:
                print(
                    f"Warning: Target trajectories diverge before expected! Seed {seed}, match up to {min_len}: {traj_match}"
                )

            # Plot success positions where each robot reached the target
            if success_pos1 is not None:
                ax.plot(
                    success_pos1[1],
                    success_pos1[0],
                    "go",
                    markersize=10,
                    label=f"{config1} Success",
                    markeredgecolor="darkgreen",
                    markeredgewidth=2,
                )

            if success_pos2 is not None:
                ax.plot(
                    success_pos2[1],
                    success_pos2[0],
                    "g^",
                    markersize=10,
                    label=f"{config2} Success",
                    markeredgecolor="darkgreen",
                    markeredgewidth=2,
                )

            # Plot final target position
            ax.plot(
                env1.true_target_pos[1],
                env1.true_target_pos[0],
                "g*",
                markersize=15,
                label="Final Target",
            )

            # Formatting
            ax.set_title(f"Seed {seed}")
            ax.legend(fontsize=12)

            # Add colorbar only to the last subplot
            if i == n_plots - 1:
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label("Signal level")

        # Hide any unused subplots
        total_subplots = n_rows * n_cols
        for i in range(n_plots, total_subplots):
            axes[i].set_visible(False)

        plt.suptitle(f"Trajectory Comparison: {config1} vs {config2}", fontsize=16)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
            print(f"Trajectory plot saved to {save_path}")

        return fig

    def plot_trajectory_comparison_separate(
        self,
        trajectories_data: dict,
        run_index: int = 0,
        figsize: tuple = (7, 4),
        save_path: str = None,
    ):
        """
        Plot trajectory comparison with separate side-by-side plots for one pair.

        Args:
            trajectories_data: Output from run_trajectory_comparison
            run_index: Which run to plot (default: 0)
            figsize: Figure size
            save_path: Optional path to save figure

        Returns:
            matplotlib figure
        """
        config_names = list(trajectories_data.keys())
        if len(config_names) != 2:
            raise ValueError("Exactly two configurations required")

        config1, config2 = config_names

        # Get data for the specified run
        traj_data1 = trajectories_data[config1][run_index]
        traj_data2 = trajectories_data[config2][run_index]

        trajectory1 = np.array(traj_data1["trajectory"])
        trajectory2 = np.array(traj_data2["trajectory"])
        env1 = traj_data1["env"]
        env2 = traj_data2["env"]

        # Create side-by-side plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Get signal maps for both environments (use same colorbar range)
        signal_map1 = env1.signal_model.compute_all_expected_signal(
            env1.true_target_pos
        )
        signal_map2 = env2.signal_model.compute_all_expected_signal(
            env2.true_target_pos
        )

        # Get common color range for both plots
        vmin = min(signal_map1.min(), signal_map2.min())
        vmax = max(signal_map1.max(), signal_map2.max())

        # Determine colors based on configuration names
        if "Standard KF" in config1:
            color1 = "tab:blue"
        else:
            color1 = "tab:orange"

        if "Signal-aware" in config2 or "signal-aware" in config2.lower():
            color2 = "tab:orange"
        else:
            color2 = "tab:blue"

        # Plot configuration 1
        im1 = ax1.imshow(
            signal_map1, cmap="Greens", interpolation="nearest", vmin=vmin, vmax=vmax
        )
        ax1.plot(
            trajectory1[:, 1],
            trajectory1[:, 0],
            color=color1,
            linewidth=2.5,
            alpha=0.8,
        )

        # Plot start position
        ax1.plot(trajectory1[0, 1], trajectory1[0, 0], "ks", markersize=8)

        # Plot final target position
        ax1.plot(
            env1.true_target_pos[1],
            env1.true_target_pos[0],
            "r*",
            markersize=15,
        )

        # Plot configuration 2
        im2 = ax2.imshow(
            signal_map2, cmap="Greens", interpolation="nearest", vmin=vmin, vmax=vmax
        )
        ax2.plot(
            trajectory2[:, 1],
            trajectory2[:, 0],
            color=color2,
            linewidth=2.5,
            alpha=0.8,
        )

        # Plot start position
        ax2.plot(trajectory2[0, 1], trajectory2[0, 0], "ks", markersize=8)

        # Plot final target position
        ax2.plot(
            env2.true_target_pos[1],
            env2.true_target_pos[0],
            "r*",
            markersize=15,
        )

        # Remove y-axis ticks and labels from right plot
        ax2.set_yticklabels([])
        ax2.tick_params(left=False)

        # Add horizontal legend below both plots
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="tab:blue", lw=2, label="Standard KF"),
            Line2D([0], [0], color="tab:orange", lw=2, label="Signal-aware KF"),
            Line2D([0], [0], marker="s", color="k", lw=0, markersize=8, label="Start"),
            Line2D(
                [0], [0], marker="*", color="r", lw=0, markersize=15, label="Target"
            ),
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=4,
            fontsize=11,
            frameon=False,
        )

        # Add colorbar on the right side (thinner and closer)
        fig.subplots_adjust(right=0.9, wspace=0.05)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(im1, cax=cbar_ax)
        cbar.set_label("Signal Strength")

        plt.subplots_adjust(bottom=0.15)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Separate trajectory plot saved to {save_path}")

        return fig


# Convenience function for quick comparisons
def quick_compare(
    configs: Dict[str, Dict],
    base_config: Optional[Dict] = None,
    n_runs: int = 10,
    max_steps: int = 5000,
    log_scale: bool = False,
    **kwargs,
):
    """
    Quick comparison function for easy one-liner usage.

    Args:
        configs: Dictionary of {name: config} pairs
        base_config: Base configuration
        n_runs: Number of runs per configuration
        max_steps: Maximum steps per simulation
        log_scale: Whether to use log scale for completion time plot
        **kwargs: Additional arguments for run_comparison()

    Returns:
        tuple: (results_dataframe, comparison_plots_figure)
    """
    comparison = MultiConfigComparison(base_config)

    for name, config in configs.items():
        comparison.add_config(name, config)

    results = comparison.run_comparison(n_runs=n_runs, max_steps=max_steps, **kwargs)
    plots = comparison.create_comparison_plots(results, log_scale=log_scale)

    return results, plots


if __name__ == "__main__":
    # Example usage
    print("=== Multi-Configuration Comparison Example ===")

    # Define base configuration (shared parameters)
    base_config = {
        "grid_size": 100,
        "motion_noise_type": "isotropic",
        "process_sigma": 0.3,
        "target_motion_sigma": 0.5,
        "signal_max": 10,
        "signal_decay": 0.05,
        "step_size": 0.2,
        "kernel_size": 5,
        "adaptive_filtering": False,
        "noise_model": "poisson",
        # process_sigma_estimate and noise_estimate will be set intelligently by default
    }

    # Define configurations to compare
    configs_to_compare = {
        "Standard KF": {
            "adaptive_process_variance": False,
        },
        "Signal-aware KF": {
            "adaptive_process_variance": True,
            "adaptive_decay_type": "power_law",
            "adaptive_rate": 1,
            "power_exponent": 0.5,
        },
    }

    # Run comparison
    comparison = MultiConfigComparison(base_config)
    for name, config in configs_to_compare.items():
        comparison.add_config(name, config)

    results = comparison.run_comparison(n_runs=200, max_steps=2000000)
    plots = comparison.create_comparison_plots(results, log_scale=True)

    # Perform paired t-test
    test_results = comparison.perform_paired_ttest(
        results,
        config1="Standard KF",
        config2="Signal-aware KF",
        metric="steps_to_target",
    )
    comparison.print_ttest_results(test_results)

    # Plot trajectory comparisons
    print("\n=== Generating Trajectory Comparison Plots ===")
    trajectory_data = comparison.run_trajectory_comparison(
        "Standard KF", "Signal-aware KF", n_runs=1, max_steps=50000, seed=2
    )
    trajectory_fig = comparison.plot_trajectory_comparison_separate(
        trajectory_data, run_index=0, save_path="trajectory_comparison_separate.pdf"
    )

    plt.show()
