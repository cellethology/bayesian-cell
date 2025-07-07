"""
Visualization and plotting utilities for EKF target tracking.
Handles all plotting and visualization tasks for EKF simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class EKFVisualizer:
    """Handles visualization and plotting for EKF simulation results."""

    def __init__(self, results):
        """
        Initialize visualizer with simulation results.

        Args:
            results: Dictionary containing simulation results from EKFEnvironment.run_simulation()
        """
        self.results = results
        self.config = results["config"]

        # Extract key data
        self.robot_trajectory = results["robot_trajectory"]
        self.target_trajectory = results["target_trajectory"]
        self.sigma_history = results["sigma_history"]
        self.R_est_history = results.get("R_est_history", [])
        self.final_belief = results["final_belief"]
        self.final_target_pos = results["final_target_pos"]

    def create_three_plot_figure(self, figsize=(15, 5)):
        """
        Create the three required plots:
        1. Environment signal field with trajectories
        2. Sigma evolution over time
        3. Final belief state

        Args:
            figsize: Figure size (width, height)

        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

        # Plot 1: Signal field with trajectories
        self._plot_signal_field_with_trajectories(ax1)

        # Plot 2: Sigma evolution
        self._plot_sigma_evolution(ax2)

        # Plot 3: Final belief state
        self._plot_final_belief_state(ax3)

        plt.tight_layout()
        return fig

    def _plot_signal_field_with_trajectories(self, ax):
        """Plot signal field with robot and target trajectories."""
        # Compute signal field for visualization
        arena_size = int(self.config["arena_max"] - self.config["arena_min"])
        x = np.linspace(self.config["arena_min"], self.config["arena_max"], arena_size)
        y = np.linspace(self.config["arena_min"], self.config["arena_max"], arena_size)
        X, Y = np.meshgrid(x, y)

        # Use final target position for signal field
        signal_grid = np.zeros_like(X)
        target_pos = self.final_target_pos

        for i in range(arena_size):
            for j in range(arena_size):
                pos = np.array([X[i, j], Y[i, j]])
                distance = np.linalg.norm(pos - target_pos)
                signal_grid[i, j] = self.config["signal_max"] * np.exp(
                    -self.config["signal_decay"] * distance
                )

        # Plot signal field as background
        im = ax.imshow(
            signal_grid,
            extent=[
                self.config["arena_min"],
                self.config["arena_max"],
                self.config["arena_min"],
                self.config["arena_max"],
            ],
            origin="lower",
            cmap="Greens",
            alpha=0.7,
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Signal Strength")

        # Plot robot trajectory (blue)
        ax.plot(
            self.robot_trajectory[:, 0],
            self.robot_trajectory[:, 1],
            "b-",
            linewidth=2,
            label="Robot Path",
            alpha=0.8,
        )
        ax.plot(
            self.robot_trajectory[0, 0],
            self.robot_trajectory[0, 1],
            "bs",
            markersize=8,
            label="Robot Start",
        )
        ax.plot(
            self.robot_trajectory[-1, 0],
            self.robot_trajectory[-1, 1],
            "bo",
            markersize=6,
            label="Robot End",
        )

        # Plot target trajectory (red)
        # ax.plot(self.target_trajectory[:, 0], self.target_trajectory[:, 1],
        #         'r-', linewidth=2, label='Target Path', alpha=0.8)
        ax.plot(
            self.target_trajectory[0, 0],
            self.target_trajectory[0, 1],
            "ro",
            markersize=8,
            label="Target Start",
        )
        ax.plot(
            self.target_trajectory[-1, 0],
            self.target_trajectory[-1, 1],
            "r*",
            markersize=15,
            label="Target End",
        )

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title("Signal Field & Trajectories")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_sigma_evolution(self, ax):
        """Plot evolution of sigma_Q and R_est over time."""
        steps = np.arange(len(self.sigma_history))
        
        # Plot sigma_Q
        ax.plot(steps, self.sigma_history, "g-", linewidth=2, label="σ_Q (Process Noise)")
        
        # Add horizontal line for baseline if adaptive
        if self.config["adaptive_process_noise"]:
            baseline = self.config["baseline_process_noise"]
            ax.axhline(
                y=baseline,
                color="r",
                linestyle="--",
                alpha=0.7,
                label=f"Baseline σ_Q = {baseline}",
            )
        
        # Plot R_est if available and adaptive measurement noise is enabled
        if len(self.R_est_history) > 0 and self.config.get("adaptive_measurement_noise", False):
            # Create second y-axis for R_est
            ax2 = ax.twinx()
            R_steps = np.arange(len(self.R_est_history))
            ax2.plot(R_steps, self.R_est_history, "b-", linewidth=2, label="R_est (Measurement Noise)")
            ax2.set_ylabel("Measurement Noise R_est", color="b")
            ax2.tick_params(axis='y', labelcolor='b')
            
            # Add legend for both axes
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax.legend()
        
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Process Noise σ_Q", color="g")
        ax.tick_params(axis='y', labelcolor='g')
        ax.set_title("Noise Evolution Over Time")
        ax.grid(True, alpha=0.3)

        # Log scale if values vary significantly
        if len(self.sigma_history) > 0:
            sigma_range = np.max(self.sigma_history) / np.min(self.sigma_history)
            if sigma_range > 10:
                ax.set_yscale("log")
                ax.set_ylabel("Process Noise σ_Q (log scale)", color="g")

    def _plot_final_belief_state(self, ax):
        """Plot final belief state as probability distribution."""
        mu, Sigma = self.final_belief

        # Create grid for belief visualization
        arena_size = 100  # Finer grid for belief
        x = np.linspace(self.config["arena_min"], self.config["arena_max"], arena_size)
        y = np.linspace(self.config["arena_min"], self.config["arena_max"], arena_size)
        X, Y = np.meshgrid(x, y)

        # Compute belief probability at each point
        pos = np.dstack((X, Y))
        # Ensure positive definiteness for visualization
        Sigma_vis = Sigma + 1e-8 * np.eye(2)
        rv = multivariate_normal(mu, Sigma_vis)
        belief_grid = rv.pdf(pos)

        # Plot belief as heatmap
        im = ax.imshow(
            belief_grid,
            extent=[
                self.config["arena_min"],
                self.config["arena_max"],
                self.config["arena_min"],
                self.config["arena_max"],
            ],
            origin="lower",
            cmap="viridis",
            alpha=0.8,
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Belief Probability Density")

        # Plot true target position
        ax.plot(
            self.final_target_pos[0],
            self.final_target_pos[1],
            "r*",
            markersize=15,
            label="True Target",
        )

        # Plot estimated target position (belief mean)
        ax.plot(
            mu[0],
            mu[1],
            "bx",
            markersize=12,
            markeredgewidth=3,
            label="Estimated Target",
        )

        # Plot confidence ellipse
        self._plot_confidence_ellipse(
            ax, mu, Sigma, confidence=0.95, color="white", linewidth=2, linestyle="--"
        )

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title("Final Belief State")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_confidence_ellipse(self, ax, mu, Sigma, confidence=0.95, **kwargs):
        """Plot confidence ellipse for 2D Gaussian distribution."""
        from scipy.stats import chi2

        # Get chi-squared value for confidence level
        chi2_val = chi2.ppf(confidence, df=2)

        # Eigendecomposition of covariance matrix
        eigenvals, eigenvecs = np.linalg.eigh(Sigma)

        # Calculate ellipse parameters
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width = 2 * np.sqrt(chi2_val * eigenvals[0])
        height = 2 * np.sqrt(chi2_val * eigenvals[1])

        # Create and plot ellipse
        from matplotlib.patches import Ellipse

        ellipse = Ellipse(mu, width, height, angle=angle, fill=False, **kwargs)
        ax.add_patch(ellipse)

    def plot_trajectory_only(self, figsize=(8, 8)):
        """Plot just the trajectories on signal field for quick visualization."""
        fig, ax = plt.subplots(figsize=figsize)
        self._plot_signal_field_with_trajectories(ax)
        plt.tight_layout()
        return fig

    def plot_sigma_only(self, figsize=(8, 6)):
        """Plot just the sigma evolution for detailed analysis."""
        fig, ax = plt.subplots(figsize=figsize)
        self._plot_sigma_evolution(ax)
        plt.tight_layout()
        return fig
    
    def plot_R_est_only(self, figsize=(8, 6)):
        """Plot just the R_est evolution for detailed analysis."""
        if len(self.R_est_history) == 0:
            print("No R_est history available. Make sure adaptive_measurement_noise is enabled.")
            return None
            
        fig, ax = plt.subplots(figsize=figsize)
        steps = np.arange(len(self.R_est_history))
        ax.plot(steps, self.R_est_history, "b-", linewidth=2, label="R_est")
        
        # Add horizontal line for initial value
        if len(self.R_est_history) > 0:
            initial_R = self.R_est_history[0]
            ax.axhline(
                y=initial_R,
                color="r",
                linestyle="--",
                alpha=0.7,
                label=f"Initial R_est = {initial_R:.3f}",
            )
        
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Measurement Noise R_est")
        ax.set_title("R_est Evolution Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Log scale if values vary significantly
        if len(self.R_est_history) > 0:
            R_range = np.max(self.R_est_history) / np.min(self.R_est_history)
            if R_range > 10:
                ax.set_yscale("log")
                ax.set_ylabel("Measurement Noise R_est (log scale)")
        
        plt.tight_layout()
        return fig

    def plot_belief_only(self, figsize=(8, 8)):
        """Plot just the final belief state for detailed analysis."""
        fig, ax = plt.subplots(figsize=figsize)
        self._plot_final_belief_state(ax)
        plt.tight_layout()
        return fig

    def save_plots(self, filename_prefix, format="png", dpi=300):
        """
        Save all plots to files.

        Args:
            filename_prefix: Prefix for filenames
            format: File format ('png', 'pdf', 'svg')
            dpi: Resolution for raster formats
        """
        # Main three-plot figure
        fig_main = self.create_three_plot_figure()
        fig_main.savefig(
            f"{filename_prefix}_three_plots.{format}", dpi=dpi, bbox_inches="tight"
        )

        # Individual plots
        fig_traj = self.plot_trajectory_only()
        fig_traj.savefig(
            f"{filename_prefix}_trajectories.{format}", dpi=dpi, bbox_inches="tight"
        )

        fig_sigma = self.plot_sigma_only()
        fig_sigma.savefig(
            f"{filename_prefix}_sigma_evolution.{format}", dpi=dpi, bbox_inches="tight"
        )

        fig_belief = self.plot_belief_only()
        fig_belief.savefig(
            f"{filename_prefix}_belief_state.{format}", dpi=dpi, bbox_inches="tight"
        )

        print(f"All plots saved with prefix '{filename_prefix}' in {format} format")

        return fig_main, fig_traj, fig_sigma, fig_belief


def visualize_ekf_results(results, plot_type="three_plots", figsize=None, show=True):
    """
    Convenience function for creating EKF visualizations.

    Args:
        results: Simulation results from EKFEnvironment.run_simulation()
        plot_type: "three_plots", "trajectory", "sigma", "R_est", or "belief"
        figsize: Figure size tuple
        show: Whether to show the plot

    Returns:
        matplotlib Figure object
    """
    visualizer = EKFVisualizer(results)

    if plot_type == "three_plots":
        figsize = figsize or (15, 5)
        fig = visualizer.create_three_plot_figure(figsize)
    elif plot_type == "trajectory":
        figsize = figsize or (8, 8)
        fig = visualizer.plot_trajectory_only(figsize)
    elif plot_type == "sigma":
        figsize = figsize or (8, 6)
        fig = visualizer.plot_sigma_only(figsize)
    elif plot_type == "R_est":
        figsize = figsize or (8, 6)
        fig = visualizer.plot_R_est_only(figsize)
    elif plot_type == "belief":
        figsize = figsize or (8, 8)
        fig = visualizer.plot_belief_only(figsize)
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")

    if show and fig is not None:
        plt.show()

    return fig
