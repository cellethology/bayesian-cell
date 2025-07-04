"""
Multi-configuration comparison tool for EKF target tracking.
Compare different EKF configurations and analyze performance differences.
"""

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
from typing import Dict, Any, Optional
import os
from scipy import stats
from ekf_environment import EKFEnvironment
from ekf_visualization import EKFVisualizer


def run_single_ekf_simulation(args):
    """Run a single EKF simulation (for multiprocessing)."""
    seed, config_name, config, max_steps, verbose = args

    # Set seed for this run
    np.random.seed(seed)

    # Remove random_seed from config to avoid conflicts
    config_copy = config.copy()
    config_copy.pop("random_seed", None)

    # Create environment and run simulation
    env = EKFEnvironment(config_copy, verbose=verbose)
    results = env.run_simulation(max_steps)

    # Sigma statistics
    sigma_mean = (
        np.mean(results["sigma_history"])
        if len(results["sigma_history"]) > 0
        else config.get("baseline_process_noise", 0.5)
    )
    sigma_std = (
        np.std(results["sigma_history"]) if len(results["sigma_history"]) > 0 else 0.0
    )
    sigma_final = (
        results["sigma_history"][-1]
        if len(results["sigma_history"]) > 0
        else config.get("baseline_process_noise", 0.5)
    )

    performance_metrics = {
        "config_name": config_name,
        "seed": seed,
        "steps_to_target": results["steps_completed"],
        "target_reached": results["target_reached"],
        "sigma_mean": sigma_mean,
        "sigma_std": sigma_std,
        "adaptive_process_noise": config.get("adaptive_process_noise", False),
        "adaptive_measurement_noise": config.get("adaptive_measurement_noise", False),
    }

    return performance_metrics


class EKFComparison:
    """
    Multi-configuration comparison tool for EKF simulations.

    Usage:
        comparison = EKFComparison()
        comparison.add_config("Standard EKF", standard_config)
        comparison.add_config("Adaptive EKF", adaptive_config)
        results = comparison.run_comparison(n_runs=20)
        comparison.create_comparison_plots(results)
    """

    def __init__(self, base_config: Optional[Dict] = None):
        """
        Initialize comparison tool.

        Args:
            base_config: Base configuration shared across all methods
        """
        self.base_config = base_config or {}
        self.configs = {}
        self.config_order = []

    def add_config(self, name: str, config: Dict[str, Any]):
        """
        Add a configuration to compare.

        Args:
            name: Human-readable name for this configuration
            config: Configuration dictionary (merged with base_config)
        """
        full_config = self.base_config.copy()
        full_config.update(config)
        self.configs[name] = full_config
        if name not in self.config_order:
            self.config_order.append(name)

    def run_comparison(
        self,
        n_runs: int = 20,
        max_steps: int = 1000000,
        verbose: bool = False,
        n_processes: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Run comparison across all configurations.

        Args:
            n_runs: Number of runs per configuration
            max_steps: Maximum steps per simulation
            verbose: Whether to show simulation progress
            n_processes: Number of processes (None = auto)

        Returns:
            DataFrame with all results
        """
        if not self.configs:
            raise ValueError("No configurations added. Use add_config() first.")

        print(
            f"Running EKF comparison of {len(self.configs)} configurations with {n_runs} runs each..."
        )

        # Create all tasks
        tasks = []
        seeds = list(range(n_runs))  # Same seeds for fair comparison

        for seed in seeds:
            for config_name, config in self.configs.items():
                tasks.append((seed, config_name, config, max_steps, verbose))

        # Run simulations in parallel
        if n_processes is None:
            n_processes = min(len(tasks), os.cpu_count())

        results = []
        with Pool(n_processes) as pool:
            for result in tqdm(
                pool.imap_unordered(run_single_ekf_simulation, tasks),
                total=len(tasks),
                desc="Running EKF simulations",
            ):
                results.append(result)

        # Convert to DataFrame
        df = pd.DataFrame(results)

        print(f"\nComparison complete! {len(results)} simulations finished.")
        self._print_summary_statistics(df)

        return df

    def _print_summary_statistics(self, df: pd.DataFrame):
        """Print summary statistics for each configuration."""
        print("=" * 70)
        print("EKF CONFIGURATION COMPARISON SUMMARY")
        print("=" * 70)

        for config_name in self.config_order:
            if config_name not in df["config_name"].values:
                continue

            config_data = df[df["config_name"] == config_name]

            print(f"\n{config_name}:")
            print(f"  Success Rate: {config_data['target_reached'].mean():.2%}")

            # Steps to target statistics
            successful_data = config_data[config_data["target_reached"] == True]
            if len(successful_data) > 0:
                median_steps = successful_data["steps_to_target"].median()
                q1_steps = successful_data["steps_to_target"].quantile(0.25)
                q3_steps = successful_data["steps_to_target"].quantile(0.75)

                # Bootstrap 95% confidence interval for median
                n_bootstrap = 1000
                bootstrap_medians = []
                steps_values = successful_data["steps_to_target"].values

                for _ in range(n_bootstrap):
                    bootstrap_sample = np.random.choice(
                        steps_values, size=len(steps_values), replace=True
                    )
                    bootstrap_medians.append(np.median(bootstrap_sample))

                ci_lower = np.percentile(bootstrap_medians, 2.5)
                ci_upper = np.percentile(bootstrap_medians, 97.5)

                print(
                    f"  Median steps to Target (successful): {median_steps:.0f} (95% CI: [{ci_lower:.0f}, {ci_upper:.0f}])"
                )
                print(
                    f"  Q1, Q3 steps to Target (successful): {q1_steps:.0f}, {q3_steps:.0f}"
                )
                sem = successful_data["steps_to_target"].std() / np.sqrt(
                    len(successful_data)
                )
                print(
                    f"  Mean steps to Target (successful): {successful_data['steps_to_target'].mean():.0f} ± {sem:.0f}"
                )

    def create_comparison_plots(self, df: pd.DataFrame, figsize: tuple = (15, 10)):
        """
        Create comprehensive comparison plots.

        Args:
            df: Results DataFrame from run_comparison()
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        axes = axes.flatten()

        config_names = [
            name for name in self.config_order if name in df["config_name"].values
        ]
        colors = plt.cm.Set2(np.linspace(0, 1, len(config_names)))

        # 1. Success Rate
        success_rates = [
            df[df["config_name"] == name]["target_reached"].mean()
            for name in config_names
        ]
        bars = axes[0].bar(config_names, success_rates, color=colors, alpha=0.8)
        axes[0].set_ylabel("Success Rate")
        axes[0].set_title("Target Achievement Rate")
        axes[0].set_ylim(0, 1.1)
        for bar, rate in zip(bars, success_rates):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{rate:.1%}",
                ha="center",
                va="bottom",
            )
        axes[0].grid(True, alpha=0.3, axis="y")

        # 2. Steps to Target (successful runs only)
        successful_data = []
        labels = []
        for name in config_names:
            config_data = df[df["config_name"] == name]
            successful_runs = config_data[config_data["target_reached"] == True]
            if len(successful_runs) > 0:
                successful_data.append(successful_runs["steps_to_target"].values)
                labels.append(name)

        if successful_data:
            bp = axes[1].boxplot(successful_data, tick_labels=labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors[: len(labels)]):
                patch.set_facecolor(color)
            axes[1].set_ylabel("Steps to Target")
            axes[1].set_title("Completion Time (Successful Runs)")
            axes[1].grid(True, alpha=0.3)

        # 3. Sigma Mean
        sigma_means = []
        for name in config_names:
            config_data = df[df["config_name"] == name]
            sigma_means.append(config_data["sigma_mean"].values)

        bp = axes[2].boxplot(sigma_means, tick_labels=config_names, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        axes[2].set_ylabel("Mean σ_Q")
        axes[2].set_title("Average Process Noise")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_mean_plot_with_ci(
        self,
        df: pd.DataFrame,
        save_path: str = "ekf_mean_comparison_with_ci.png",
        figsize: tuple = (10, 6),
        confidence: float = 0.95,
        show_plot: bool = True,
    ):
        """
        Create line plot showing mean steps to target with confidence intervals.

        Args:
            df: Results DataFrame from run_comparison()
            save_path: Path to save the plot
            figsize: Figure size tuple
            confidence: Confidence level (0.95 for 95% CI)
            show_plot: Whether to display the plot

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Get config names in order
        config_names = [
            name for name in self.config_order if name in df["config_name"].values
        ]
        colors = plt.cm.Set2(np.linspace(0, 1, len(config_names)))

        # Calculate mean and CI for each configuration
        x_pos = range(len(config_names))
        means = []
        ci_lowers = []
        ci_uppers = []

        for config_name in config_names:
            config_data = df[df["config_name"] == config_name]
            successful_data = config_data[config_data["target_reached"] == True]

            if len(successful_data) > 0:
                steps_data = successful_data["steps_to_target"].values
                mean_val = np.mean(steps_data)

                # Bootstrap confidence interval for mean
                n_bootstrap = 1000
                bootstrap_means = []
                for _ in range(n_bootstrap):
                    bootstrap_sample = np.random.choice(
                        steps_data, size=len(steps_data), replace=True
                    )
                    bootstrap_means.append(np.mean(bootstrap_sample))

                # Calculate confidence interval
                alpha = 1 - confidence
                ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
                ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

                means.append(mean_val)
                ci_lowers.append(ci_lower)
                ci_uppers.append(ci_upper)
            else:
                means.append(np.nan)
                ci_lowers.append(np.nan)
                ci_uppers.append(np.nan)

        # Create error bars (CI intervals)
        ci_errors = [
            [means[i] - ci_lowers[i] for i in range(len(means))],
            [ci_uppers[i] - means[i] for i in range(len(means))],
        ]

        # Plot with confidence intervals
        bars = ax.bar(
            x_pos,
            means,
            yerr=ci_errors,
            color=colors,
            alpha=0.8,
            capsize=5,
            error_kw={"linewidth": 2, "capthick": 2},
        )

        # Add value labels on bars
        for i, (bar, mean_val, ci_lower, ci_upper) in enumerate(
            zip(bars, means, ci_lowers, ci_uppers)
        ):
            if not np.isnan(mean_val):
                # Add mean value on top of bar
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (ci_upper - mean_val) + max(means) * 0.02,
                    f"{mean_val:.0f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        # Formatting
        ax.set_xlabel("Configuration", fontsize=12)
        ax.set_ylabel("Mean Steps to Target", fontsize=12)
        ax.set_title(
            f"Mean Completion Time with {confidence*100:.0f}% Confidence Intervals",
            fontsize=14,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(config_names, rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        # Remove top and right spines for cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        # Save plot
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Mean plot with CI saved to {save_path}")

        # Show plot
        if show_plot:
            plt.show()

        return fig

    def create_line_plot_with_ci(
        self,
        df: pd.DataFrame,
        x_column: str,
        save_path: str = "ekf_line_comparison_with_ci.png",
        figsize: tuple = (10, 6),
        confidence: float = 0.95,
        show_plot: bool = True,
    ):
        """
        Create line plot showing mean steps to target with confidence intervals across a parameter.

        Args:
            df: Results DataFrame from run_comparison()
            x_column: Column name to use for x-axis (e.g., 'signal_max', 'target_motion_sigma')
            save_path: Path to save the plot
            figsize: Figure size tuple
            confidence: Confidence level (0.95 for 95% CI)
            show_plot: Whether to display the plot

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Get config names in order
        config_names = [
            name for name in self.config_order if name in df["config_name"].values
        ]
        colors = plt.cm.Set2(np.linspace(0, 1, len(config_names)))

        # Color and style for each method
        method_styles = {
            config_names[i]: {
                "color": colors[i],
                "marker": ["o", "s", "^", "D", "v"][i % 5],
            }
            for i in range(len(config_names))
        }

        # Get unique x values
        x_values = sorted(df[x_column].unique())

        print(f"Creating line plot for methods: {config_names}")
        print(f"{x_column} values: {x_values}")

        # Plot each method
        for config_name in config_names:
            method_means = []
            ci_lowers = []
            ci_uppers = []
            valid_x_values = []

            for x_val in x_values:
                method_data = df[
                    (df["config_name"] == config_name) & (df[x_column] == x_val)
                ]

                # Get successful runs only
                successful_data = method_data[method_data["target_reached"] == True]

                if len(successful_data) > 0:
                    steps_data = successful_data["steps_to_target"].values
                    mean_val = np.mean(steps_data)

                    # Bootstrap confidence interval for mean
                    n_bootstrap = 1000
                    bootstrap_means = []
                    for _ in range(n_bootstrap):
                        bootstrap_sample = np.random.choice(
                            steps_data, size=len(steps_data), replace=True
                        )
                        bootstrap_means.append(np.mean(bootstrap_sample))

                    # Calculate confidence interval
                    alpha = 1 - confidence
                    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
                    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

                    if not np.isnan(mean_val):
                        method_means.append(mean_val)
                        ci_lowers.append(ci_lower)
                        ci_uppers.append(ci_upper)
                        valid_x_values.append(x_val)

            if len(method_means) > 0:
                style = method_styles.get(
                    config_name, {"color": "black", "marker": "o"}
                )

                print(f"Plotting {config_name}: {len(method_means)} points")

                # Plot mean line with CI error bars
                ax.errorbar(
                    valid_x_values,
                    method_means,
                    yerr=[
                        [
                            method_means[i] - ci_lowers[i]
                            for i in range(len(method_means))
                        ],
                        [
                            ci_uppers[i] - method_means[i]
                            for i in range(len(method_means))
                        ],
                    ],
                    label=config_name,
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=2,
                    markersize=6,
                    capsize=3,
                    capthick=1,
                )
            else:
                print(f"Warning: No valid data points for {config_name}")

        # Formatting
        ax.set_xlabel(x_column.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel("Mean Steps to Target", fontsize=12)
        ax.set_title(
            f"Mean Completion Time with {confidence*100:.0f}% Confidence Intervals",
            fontsize=14,
        )
        ax.legend(fontsize=11, frameon=False)
        ax.grid(True, alpha=0.3)

        # Remove top and right spines for cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        # Save plot
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Line plot with CI saved to {save_path}")

        # Show plot
        if show_plot:
            plt.show()

        return fig

    def perform_paired_ttest(
        self,
        df: pd.DataFrame,
        config1: str,
        config2: str,
        metric: str = "steps_to_target",
    ):
        """
        Perform paired t-test between two configurations.

        Args:
            df: Results DataFrame
            config1: First configuration name
            config2: Second configuration name
            metric: Metric to compare

        Returns:
            dict: Test results
        """
        data1 = df[df["config_name"] == config1].sort_values("seed")
        data2 = df[df["config_name"] == config2].sort_values("seed")

        common_seeds = set(data1["seed"]) & set(data2["seed"])
        if len(common_seeds) == 0:
            raise ValueError(f"No common seeds found between {config1} and {config2}")

        data1_paired = data1[data1["seed"].isin(common_seeds)].sort_values("seed")
        data2_paired = data2[data2["seed"].isin(common_seeds)].sort_values("seed")

        values1 = data1_paired[metric].values
        values2 = data2_paired[metric].values

        statistic, p_value = stats.ttest_rel(values1, values2)

        # Effect size (Cohen's d for paired samples)
        differences = values1 - values2
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0

        results = {
            "config1": config1,
            "config2": config2,
            "metric": metric,
            "n_pairs": len(common_seeds),
            "mean1": np.mean(values1),
            "mean2": np.mean(values2),
            "mean_difference": mean_diff,
            "t_statistic": statistic,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "significant": p_value < 0.05,
        }

        return results

    def run_trajectory_comparison(
        self,
        config1: str,
        config2: str,
        n_runs: int = 3,
        max_steps: int = 100000,
        seed: int = None,
    ):
        """
        Run trajectory comparison between two configurations.

        Args:
            config1: First configuration name
            config2: Second configuration name
            n_runs: Number of trajectory runs
            max_steps: Maximum steps per run
            seed: Starting seed

        Returns:
            dict: Trajectory data for both configurations
        """
        if config1 not in self.configs or config2 not in self.configs:
            raise ValueError(f"Both {config1} and {config2} must be added configs")

        trajectories_data = {config1: [], config2: []}
        seeds = list(range(seed or 0, (seed or 0) + n_runs))

        for run_seed in seeds:
            # Generate target trajectory that's consistent between configs
            # We'll use the same random seed for both configs to ensure consistent targets
            seed_results = {}

            for config_name in [config1, config2]:
                # Set seed and run simulation
                np.random.seed(run_seed)
                env = EKFEnvironment(self.configs[config_name], verbose=False)
                results = env.run_simulation(max_steps)

                seed_results[config_name] = {
                    "results": results,
                    "seed": run_seed,
                    "robot_trajectory": results.get("robot_trajectory", []),
                    "target_trajectory": results.get("target_trajectory", []),
                    "env": env,
                }

            # Add both results to trajectories_data
            for config_name in [config1, config2]:
                trajectories_data[config_name].append(seed_results[config_name])

        return trajectories_data

    def plot_trajectory_comparison(
        self, trajectories_data: dict, figsize: tuple = (12, 8), save_path: str = None
    ):
        """
        Plot trajectory comparison between configurations.

        Args:
            trajectories_data: Output from run_trajectory_comparison()
            figsize: Figure size
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        config_names = list(trajectories_data.keys())
        n_runs = len(trajectories_data[config_names[0]])

        fig, axes = plt.subplots(2, n_runs, figsize=figsize)
        if n_runs == 1:
            axes = axes.reshape(2, 1)

        for run_idx in range(n_runs):
            for config_idx, config_name in enumerate(config_names):
                ax = axes[config_idx, run_idx]

                results = trajectories_data[config_name][run_idx]["results"]
                seed = trajectories_data[config_name][run_idx]["seed"]

                # Create visualizer and plot trajectory
                visualizer = EKFVisualizer(results)
                visualizer._plot_signal_field_with_trajectories(ax)

                ax.set_title(f"{config_name} (Seed {seed})")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Trajectory comparison saved to {save_path}")

        return fig

    def plot_trajectory_comparison_separate(
        self,
        trajectories_data: dict,
        run_index: int = 0,
        figsize: tuple = (12, 5),
        save_path: str = None,
        step_size: int = 20,
    ):
        """
        Plot trajectory comparison with separate side-by-side plots for one pair,
        including target trajectory and time-based color intensity.

        Args:
            trajectories_data: Output from run_trajectory_comparison
            run_index: Which run to plot (default: 0)
            figsize: Figure size
            save_path: Optional path to save figure
            step_size: Plot every Nth point to reduce clutter (default: 20)

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

        # Extract trajectories
        robot_trajectory1 = np.array(traj_data1["robot_trajectory"])
        robot_trajectory2 = np.array(traj_data2["robot_trajectory"])
        target_trajectory1 = np.array(traj_data1["target_trajectory"])
        target_trajectory2 = np.array(traj_data2["target_trajectory"])

        # Create side-by-side plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Get signal field from environment (assumed to be same for both configs)
        env1 = traj_data1["env"]
        signal_field = self._compute_signal_field(env1)

        # Plot signal field for both subplots
        arena_min = env1.config["arena_min"]
        arena_max = env1.config["arena_max"]
        im1 = ax1.imshow(
            signal_field,
            extent=[arena_min, arena_max, arena_min, arena_max],
            origin="lower",
            cmap="Greens",
            alpha=0.7,
        )
        im2 = ax2.imshow(
            signal_field,
            extent=[arena_min, arena_max, arena_min, arena_max],
            origin="lower",
            cmap="Greens",
            alpha=0.7,
        )

        # Plot robot trajectories with time-based color intensity
        self._plot_trajectory_with_time_colors(
            ax1, robot_trajectory1, "Robot", "blue", env1, step_size
        )
        self._plot_trajectory_with_time_colors(
            ax2, robot_trajectory2, "Robot", "blue", env1, step_size
        )

        # Plot target trajectories with time-based color intensity
        self._plot_trajectory_with_time_colors(
            ax1, target_trajectory1, "Target", "red", env1, step_size
        )
        self._plot_trajectory_with_time_colors(
            ax2, target_trajectory2, "Target", "red", env1, step_size
        )

        # Set titles
        ax1.set_title(f"{config1}")
        ax2.set_title(f"{config2}")

        # Set axis labels
        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position")
        ax2.set_xlabel("X Position")
        ax2.set_ylabel("Y Position")

        # Add shared colorbar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(im1, cax=cbar_ax)
        cbar.set_label("Signal Strength")

        # Add legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="blue", lw=2, label="Robot Trajectory"),
            Line2D([0], [0], color="red", lw=2, label="Target Trajectory"),
            Line2D(
                [0], [0], marker="o", color="black", lw=0, markersize=8, label="Start"
            ),
            Line2D(
                [0], [0], marker="*", color="black", lw=0, markersize=10, label="End"
            ),
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=4,
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Separate trajectory plot saved to {save_path}")

        return fig

    def plot_trajectory_comparison_all_runs(
        self,
        trajectories_data: dict,
        figsize: tuple = (16, 10),
        save_path: str = None,
        step_size: int = 20,
        show_plot: bool = True,
    ):
        """
        Plot trajectory comparison showing all runs with separate plots for each configuration.
        Similar to plot_trajectory_comparison_separate but for all runs.

        Args:
            trajectories_data: Output from run_trajectory_comparison
            figsize: Figure size tuple
            save_path: Optional path to save figure
            step_size: Plot every Nth point to reduce clutter (default: 20)
            show_plot: Whether to display the plot

        Returns:
            matplotlib figure
        """
        config_names = list(trajectories_data.keys())
        if len(config_names) != 2:
            raise ValueError("Exactly two configurations required")

        config1, config2 = config_names
        n_runs = len(trajectories_data[config1])

        # Calculate grid layout - max 4 columns, 2 rows per config
        max_cols = 4
        n_cols = min(n_runs, max_cols)
        n_rows_per_config = (n_runs + max_cols - 1) // max_cols  # Ceiling division
        total_rows = 2 * n_rows_per_config  # 2 configs

        # Create figure with subplots
        fig, axes = plt.subplots(total_rows, n_cols, figsize=figsize)

        # Handle different subplot arrangements
        if total_rows == 1:
            axes = axes.reshape(1, -1) if n_cols > 1 else axes.reshape(1, 1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        print(
            f"Creating trajectory plot for {n_runs} runs with {n_cols} columns and {total_rows} rows (2 configs)"
        )

        # Plot each configuration in separate rows
        for config_idx, config_name in enumerate([config1, config2]):
            row_offset = config_idx * n_rows_per_config

            for run_idx in range(n_runs):
                # Calculate subplot position
                subplot_row = row_offset + (run_idx // max_cols)
                subplot_col = run_idx % max_cols

                ax = axes[subplot_row, subplot_col]

                # Get data for this configuration and run
                traj_data = trajectories_data[config_name][run_idx]

                # Extract trajectories
                robot_trajectory = np.array(traj_data["robot_trajectory"])
                target_trajectory = np.array(traj_data["target_trajectory"])

                env = traj_data["env"]
                seed = traj_data["seed"]

                # Get signal field
                signal_field = self._compute_signal_field(env)
                arena_min = env.config["arena_min"]
                arena_max = env.config["arena_max"]

                # Plot signal field background
                im = ax.imshow(
                    signal_field,
                    extent=[arena_min, arena_max, arena_min, arena_max],
                    origin="lower",
                    cmap="Greens",
                    alpha=0.7,
                )

                # Plot robot trajectory
                self._plot_trajectory_with_time_colors(
                    ax, robot_trajectory, "Robot", "blue", env, step_size
                )

                # Plot target trajectory
                self._plot_trajectory_with_time_colors(
                    ax, target_trajectory, "Target", "red", env, step_size
                )

                # Add results info to title
                steps = traj_data["results"]["steps_completed"]
                reached = traj_data["results"]["target_reached"]

                title = f"Run {run_idx+1} (Seed {seed})\n"
                title += f"{steps} steps ({'✓' if reached else '✗'})"

                ax.set_title(title, fontsize=10)
                ax.set_xlabel("X Position", fontsize=9)
                ax.set_ylabel("Y Position", fontsize=9)

        # Hide any unused subplots
        total_subplots = total_rows * n_cols
        used_subplots = 2 * n_runs  # 2 configs × n_runs

        for row in range(total_rows):
            for col in range(n_cols):
                subplot_idx = row * n_cols + col
                if subplot_idx >= used_subplots:
                    axes[row, col].set_visible(False)

        # Add configuration labels on the left
        for config_idx, config_name in enumerate([config1, config2]):
            row_center = config_idx * n_rows_per_config + (n_rows_per_config - 1) / 2
            fig.text(
                0.02,
                0.85 - config_idx * 0.4,
                config_name,
                rotation=90,
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )

        # Add overall title
        fig.suptitle(
            f"All Trajectory Runs: {config1} vs {config2}", fontsize=16, y=0.98
        )

        # Create legend at the bottom
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="blue", lw=2, label="Robot Trajectory"),
            Line2D([0], [0], color="red", lw=2, label="Target Trajectory"),
            Line2D(
                [0], [0], marker="o", color="black", lw=0, markersize=6, label="Start"
            ),
            Line2D(
                [0], [0], marker="*", color="black", lw=0, markersize=8, label="End"
            ),
        ]

        fig.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=4,
            fontsize=11,
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.93, bottom=0.12, left=0.08)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"All runs trajectory plot saved to {save_path}")

        if show_plot:
            plt.show()

        return fig

    def _compute_signal_field(self, env):
        """Compute signal field for visualization."""
        arena_min = env.config["arena_min"]
        arena_max = env.config["arena_max"]
        signal_max = env.config["signal_max"]
        signal_decay = env.config["signal_decay"]
        target_pos = env.config["target_true_pos"]

        # Create coordinate grid
        x = np.linspace(arena_min, arena_max, 100)
        y = np.linspace(arena_min, arena_max, 100)
        X, Y = np.meshgrid(x, y)

        # Compute signal field using the environment's method which handles periodic boundaries
        _, _, signal_field = env.compute_signal_field(target_pos)

        # Ensure signal field has the right size for visualization
        if signal_field.shape != (100, 100):
            # Interpolate to desired grid size using RegularGridInterpolator
            from scipy.interpolate import RegularGridInterpolator

            arena_range = arena_max - arena_min
            original_x = np.linspace(arena_min, arena_max, signal_field.shape[1])
            original_y = np.linspace(arena_min, arena_max, signal_field.shape[0])

            # Create interpolator
            interp = RegularGridInterpolator(
                (original_y, original_x),
                signal_field,
                method="linear",
                bounds_error=False,
                fill_value=0,
            )

            # Create new grid points
            new_y, new_x = np.meshgrid(y, x, indexing="ij")
            points = np.stack([new_y.ravel(), new_x.ravel()], axis=-1)

            # Interpolate
            signal_field = interp(points).reshape(100, 100)

        return signal_field

    def _plot_trajectory_with_time_colors(
        self, ax, trajectory, label, base_color, env=None, step_size=20
    ):
        """Plot trajectory with time-based color intensity, handling periodic boundaries."""
        if len(trajectory) < 2:
            return

        # Subsample trajectory to reduce clutter
        if len(trajectory) > step_size:
            # Always include start and end points
            indices = list(range(0, len(trajectory), step_size))
            if indices[-1] != len(trajectory) - 1:
                indices.append(len(trajectory) - 1)
            trajectory = trajectory[indices]

        # Create color map based on time
        n_points = len(trajectory)
        # Map color names to proper colormap names
        color_map = {
            "blue": "Blues",
            "red": "Reds",
            "green": "Greens",
            "orange": "Oranges",
            "purple": "Purples",
        }
        cmap_name = color_map.get(base_color, "viridis")
        colors = plt.cm.get_cmap(cmap_name)(np.linspace(0.3, 1.0, n_points))

        # Check if we need to handle periodic boundaries
        periodic = env is not None and env.config.get("periodic_boundaries", False)

        if periodic:
            arena_min = env.config["arena_min"]
            arena_max = env.config["arena_max"]
            arena_size = arena_max - arena_min

        # Plot trajectory segments
        for i in range(len(trajectory) - 1):
            current_pos = trajectory[i]
            next_pos = trajectory[i + 1]

            # Check for wrapping discontinuities in periodic boundaries
            if periodic and self._is_wrapping_discontinuity(
                current_pos, next_pos, arena_size
            ):
                # Don't draw line across the discontinuity
                continue

            ax.plot(
                [current_pos[0], next_pos[0]],
                [current_pos[1], next_pos[1]],
                color=colors[i],
                linewidth=2,
                alpha=0.8,
            )

        # Plot start and end points
        ax.plot(
            trajectory[0, 0],
            trajectory[0, 1],
            "o",
            color=base_color,
            markersize=8,
            label=f"{label} Start",
        )
        ax.plot(
            trajectory[-1, 0],
            trajectory[-1, 1],
            "*",
            color=base_color,
            markersize=12,
            label=f"{label} End",
        )

    def _is_wrapping_discontinuity(self, pos1, pos2, arena_size):
        """Check if there's a wrapping discontinuity between two positions."""
        diff = np.abs(pos2 - pos1)
        # If any dimension has a jump larger than half the arena size, it's likely a wrap
        return np.any(diff > arena_size * 0.4)


def quick_ekf_compare(
    configs: Dict[str, Dict],
    base_config: Optional[Dict] = None,
    n_runs: int = 10,
    max_steps: int = 1000000,
):
    """
    Quick comparison function for EKF configurations.

    Args:
        configs: Dictionary of {name: config} pairs
        base_config: Base configuration
        n_runs: Number of runs per configuration
        max_steps: Maximum steps per simulation

    Returns:
        tuple: (results_dataframe, comparison_plots_figure)
    """
    comparison = EKFComparison(base_config)

    for name, config in configs.items():
        comparison.add_config(name, config)

    results = comparison.run_comparison(n_runs=n_runs, max_steps=max_steps)
    plots = comparison.create_comparison_plots(results)

    return results, plots


if __name__ == "__main__":
    # Example usage
    print("=== EKF Multi-Configuration Comparison Example ===")

    # Base configuration
    base_config = {
        "arena_min": 0.0,
        "arena_max": 200.0,
        "distance_tolerance": 5.0,
        "signal_max": 10.0,
        "signal_decay": 0.05,
        "robot_start_pos": [60.0, 60.0],
        "robot_step_size": 0.3,
        "actuator_noise": 0.5,
        "target_true_pos": [140.0, 140.0],
        "initial_belief_mean": [100.0, 100.0],
        "initial_belief_variance": 10000.0,
        "target_motion_sigma": 0.5,
        "baseline_process_noise": 0.5,
        "alpha_R": 0.5,
        "adaptive_measurement_noise": False,
        "max_steps": 1000000,
    }

    # Configurations to compare
    configs_to_compare = {
        "Standard EKF": {
            "adaptive_process_noise": False,
            "adaptive_measurement_noise": False,
            "periodic_boundaries": True,
        },
        "Adaptive Process EKF": {
            "adaptive_process_noise": True,
            "adaptive_measurement_noise": False,
            "periodic_boundaries": True,
        },
        # "Adaptive Measurement EKF": {
        #     "adaptive_process_noise": False,
        #     "adaptive_measurement_noise": True,
        #     "periodic_boundaries": True,
        # },
    }

    # Run comparison
    comparison = EKFComparison(base_config)
    for name, config in configs_to_compare.items():
        comparison.add_config(name, config)

    # results = comparison.run_comparison(n_runs=500, max_steps=1000000)
    # plots = comparison.create_comparison_plots(results)

    # # Create mean plot with confidence intervals
    # ci_plot = comparison.create_mean_plot_with_ci(results, show_plot=False)

    # # Perform statistical tests
    # comparisons = [
    #     ("Standard EKF", "Adaptive Process EKF"),
    #     ("Standard EKF", "Adaptive Measurement EKF"),
    # ]

    # print(f"\nPaired t-test results:")
    # for config1, config2 in comparisons:
    #     test_results = comparison.perform_paired_ttest(
    #         results, config1, config2, "steps_to_target"
    #     )
    #     print(f"\n{config1} vs {config2}:")
    #     print(f"  t-statistic: {test_results['t_statistic']:.4f}")
    #     print(f"  p-value: {test_results['p_value']:.6f}")
    #     print(f"  Significant: {test_results['significant']}")
    #     print(f"  Cohen's d: {test_results['cohens_d']:.4f}")

    # # Plot trajectory comparisons - compare periodic vs non-periodic
    print("\n=== Generating EKF Trajectory Comparison Plots ===")
    trajectory_data = comparison.run_trajectory_comparison(
        "Standard EKF", "Adaptive Process EKF", n_runs=1, max_steps=10000, seed=195
    )

    # Plot single run comparison
    trajectory_fig = comparison.plot_trajectory_comparison_separate(
        trajectory_data,
        run_index=0,
        step_size=1,
        save_path="ekf_trajectory_comparison_separate.png",
    )

    # Plot all runs comparison
    # all_runs_fig = comparison.plot_trajectory_comparison_all_runs(
    #     trajectory_data,
    #     step_size=5,
    #     save_path="ekf_trajectory_comparison_all_runs.png",
    #     show_plot=False,
    # )

    plt.show()
