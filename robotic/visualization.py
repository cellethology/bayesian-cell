"""
Visualization and plotting utilities for Bayesian navigation.
Handles all plotting and visualization tasks.
"""
import numpy as np
import matplotlib.pyplot as plt


class NavigationVisualizer:
    """Handles visualization and plotting for navigation results."""
    
    def __init__(self, env, trajectory, sigmas, innovations, measurement_variances):
        self.env = env
        self.trajectory = np.array(trajectory)
        self.sigmas = np.array(sigmas)
        self.innovations = np.array(innovations)
        self.measurement_variances = np.array(measurement_variances)
    
    def create_comprehensive_plot(self, figsize=(12, 9)):
        """Create a comprehensive visualization with all relevant plots in a balanced 3x3 grid."""
        fig = plt.figure(figsize=figsize)
        
        # Create a 3x3 grid for better balance (using 7 of 9 positions)
        # Row 1: Navigation overview
        self._plot_signal_map_and_trajectory(fig.add_subplot(331))
        self._plot_belief_state(fig.add_subplot(332))
        self._plot_sigma_evolution(fig.add_subplot(333))
        
        # Row 2: Measurement and innovation analysis
        self._plot_measurement_variance(fig.add_subplot(334))
        self._plot_innovation_timeline(fig.add_subplot(335))
        self._plot_innovation_histogram(fig.add_subplot(336))
        
        # Row 3: Sigma distribution (centered in bottom row)
        self._plot_sigma_histogram(fig.add_subplot(338))  # Skip position 337 for centering
        
        plt.tight_layout()
        return fig
    
    def _plot_signal_map_and_trajectory(self, ax):
        """Plot signal strength map with robot trajectory."""
        signal_map = self.env.signal_model.compute_all_expected_signal(self.env.true_target_pos)
        avg_signal_strength = np.mean(signal_map)
        print(f"Average signal strength over entire space: {avg_signal_strength:.4f}")
        
        im = ax.imshow(signal_map, cmap="Greens", interpolation="nearest")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Signal Strength")
        
        ax.plot(self.trajectory[:, 1], self.trajectory[:, 0], "b-", label="Robot Path")
        ax.plot(
            self.env.true_target_pos[1],
            self.env.true_target_pos[0],
            "g*",
            markersize=15,
            label="Target",
        )
        ax.plot(self.trajectory[0, 1], self.trajectory[0, 0], "bs", label="Start")
        ax.set_title("Signal Map & Trajectory")
        ax.legend()
    
    def _plot_belief_state(self, ax):
        """Plot final belief state with target positions."""
        ax.imshow(self.env.belief, cmap="viridis", interpolation="nearest")
        ax.plot(
            self.env.true_target_pos[1],
            self.env.true_target_pos[0],
            "r*",
            markersize=15,
            label="True Target",
        )
        
        estimated_pos = np.unravel_index(np.argmax(self.env.belief), self.env.belief.shape)
        ax.plot(
            estimated_pos[1],
            estimated_pos[0],
            "bx",
            markersize=15,
            label="Estimated Target",
        )
        ax.set_title("Final Belief State")
        ax.legend()
    
    def _plot_innovation_histogram(self, ax):
        """Plot histogram of innovations."""
        ax.hist(
            self.innovations,
            bins=50,
            density=True,
            alpha=0.7,
            color="skyblue",
            label="Innovation Histogram",
        )
        ax.axvline(
            np.median(self.innovations),
            color="red",
            linestyle="--",
            label=f"Median = {np.median(self.innovations):.4f}",
        )
        ax.axvline(
            np.std(self.innovations),
            color="green",
            linestyle="--",
            label=f"Std = {np.std(self.innovations):.4f}",
        )
        ax.set_xlabel("Innovation")
        ax.set_ylabel("Density")
        ax.set_title("Innovation Distribution")
        ax.legend()
    
    def _plot_sigma_evolution(self, ax):
        """Plot motion sigma evolution over time."""
        ax.plot(self.sigmas, "r-", label="Motion Sigma")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Sigma")
        ax.set_title("Motion Sigma Evolution")
        ax.legend()
    
    def _plot_measurement_variance(self, ax):
        """Plot measurement variance evolution."""
        ax.plot(self.measurement_variances, "g-", label="Measurement Variance")
        
        if "noise_std" in self.env.config:
            true_variance = self.env.config["noise_std"] ** 2
            ax.plot(
                np.ones_like(self.measurement_variances) * true_variance,
                "r--",
                label="True Noise Variance",
            )
        
        ax.set_yscale("log")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Measurement Variance")
        ax.set_title("Measurement Variance Evolution")
        ax.legend()
    
    def _plot_innovation_timeline(self, ax):
        """Plot innovation over time."""
        ax.plot(self.innovations, "b-", alpha=0.7, label="Innovation")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Innovation")
        ax.set_title("Innovation Timeline")
        ax.legend()
    
    def _plot_sigma_histogram(self, ax):
        """Plot histogram of motion sigma values."""
        ax.hist(
            self.sigmas,
            bins=50,
            density=True,
            color="salmon",
            alpha=0.7,
            label="Motion Sigma Histogram",
        )
        ax.axvline(
            np.median(self.sigmas),
            color="blue",
            linestyle="--",
            label=f"Median = {np.median(self.sigmas):.4f}",
        )
        ax.set_xlabel("Motion Sigma")
        ax.set_ylabel("Density")
        ax.set_title("Motion Sigma Distribution")
        ax.legend()
    
    def plot_trajectory_only(self, figsize=(8, 8)):
        """Plot just the trajectory on signal map for quick visualization."""
        fig, ax = plt.subplots(figsize=figsize)
        self._plot_signal_map_and_trajectory(ax)
        plt.tight_layout()
        return fig
    
    def plot_adaptation_metrics(self, figsize=(12, 8)):
        """Plot adaptation-related metrics (sigma, variance, innovations)."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        self._plot_sigma_evolution(ax1)
        self._plot_measurement_variance(ax2)
        self._plot_innovation_timeline(ax3)
        self._plot_innovation_histogram(ax4)
        
        plt.tight_layout()
        return fig


def visualize_simulation_results(env, trajectory, sigmas, innovations, measurement_variances, 
                               plot_type="comprehensive", figsize=None, show=True):
    """
    Convenience function for creating visualizations.
    
    Args:
        env: NavigationEnvironment instance
        trajectory: Robot trajectory
        sigmas: Motion sigma evolution
        innovations: Innovation sequence
        measurement_variances: Measurement variance evolution
        plot_type: "comprehensive", "trajectory", or "adaptation"
        figsize: Figure size tuple
        show: Whether to show the plot
    
    Returns:
        matplotlib Figure object
    """
    visualizer = NavigationVisualizer(env, trajectory, sigmas, innovations, measurement_variances)
    
    if plot_type == "comprehensive":
        figsize = figsize or (12, 9)
        fig = visualizer.create_comprehensive_plot(figsize)
    elif plot_type == "trajectory":
        figsize = figsize or (6, 6)
        fig = visualizer.plot_trajectory_only(figsize)
    elif plot_type == "adaptation":
        figsize = figsize or (10, 6)
        fig = visualizer.plot_adaptation_metrics(figsize)
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")
    
    if show:
        plt.show()
    
    return fig