"""
Main entry point for EKF target tracking simulation.
Run this file to execute a single EKF simulation with visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from ekf_environment import EKFEnvironment
from ekf_visualization import visualize_ekf_results


def create_default_config():
    """Create default configuration for EKF simulation."""
    return {
        # Arena parameters
        "arena_min": 0.0,
        "arena_max": 200.0,
        "distance_tolerance": 5.0,
        # Signal parameters
        "signal_max": 5.0,  # c0 - maximum signal strength
        "signal_decay": 0.05,  # lambda - signal decay rate
        # Robot parameters
        "robot_start_pos": [75.0, 75.0],
        "robot_step_size": 0.1,
        "actuator_noise": 1.0,  # sigma_u - robot actuator noise
        # Target parameters
        "target_true_pos": [125.0, 125.0],
        "target_motion_sigma": 0.5,  # target random walk noise
        # Filter parameters
        "filter_type": "FilterPy_EKF_Corrected",  # "EKF" or "UKF"
        "initial_belief_mean": [100.0, 100.0],  # broad prior mean
        "initial_belief_variance": 600.0,  # broad prior variance
        "baseline_process_noise": 1.0,  # sigma_Q baseline
        "adaptive_process_noise": True,  # enable/disable adaptive noise
        "alpha_R": 0.001,  # innovation-based measurement noise update rate
        "adaptive_measurement_noise": False,  # enable/disable adaptive measurement noise
        # UKF-specific parameters (ignored if using EKF)
        "ukf_alpha": 0.001,  # UKF spread parameter
        "ukf_beta": 2.0,  # UKF distribution parameter
        "ukf_kappa": 0.0,  # UKF secondary scaling parameter
        "eps": 1,
        # Simulation parameters
        "max_steps": 300000,
        "random_seed": 12,
    }


def create_ukf_config():
    """Create configuration optimized for UKF simulation."""
    config = create_default_config()
    config.update(
        {
            "filter_type": "UKF",
            # UKF works better with slightly different parameters
            "ukf_alpha": 0.001,  # Small spread for stable performance
            "ukf_beta": 2.0,  # Optimal for Gaussian distributions
            "ukf_kappa": 0.0,  # Default secondary scaling
            # You can adjust these for better UKF performance
            "baseline_process_noise": 0.2,  # UKF often works well with lower process noise
        }
    )
    return config


def run_single_simulation(config=None, config_name="Default", verbose=True):
    """
    Run a single EKF simulation with the given configuration.

    Args:
        config: Configuration dictionary (uses default if None)
        config_name: Name for the configuration (for display)
        verbose: Whether to print progress information

    Returns:
        dict: Simulation results
    """
    if config is None:
        config = create_default_config()

    print(f"\n{'='*60}")
    print(f"Running EKF Simulation: {config_name}")
    print(f"{'='*60}")

    # Print key configuration parameters
    print(f"Configuration:")
    print(f"  Arena: [{config['arena_min']:.1f}, {config['arena_max']:.1f}]")
    print(f"  Robot start: {config['robot_start_pos']}")
    print(f"  Target start: {config['target_true_pos']}")
    print(f"  Signal max: {config['signal_max']:.1f}")
    print(f"  Signal decay: {config['signal_decay']:.3f}")
    print(f"  Adaptive process noise: {config['adaptive_process_noise']}")
    print(f"  Baseline process noise: {config['baseline_process_noise']:.2f}")
    print(f"  Adaptive measurement noise: {config['adaptive_measurement_noise']}")
    print(f"  Alpha R: {config['alpha_R']:.3f}")
    print(f"  Random seed: {config['random_seed']}")

    # Set random seed for single simulation reproducibility
    if config.get("random_seed") is not None:
        np.random.seed(config["random_seed"])

    # Create environment and run simulation
    env = EKFEnvironment(config, verbose=verbose)
    results = env.run_simulation()

    # Print results summary
    print(f"\nSimulation Results:")
    print(f"  Steps completed: {results['steps_completed']:,}")
    print(f"  Target reached: {results['target_reached']}")
    print(
        f"  Final distance: {np.linalg.norm(results['final_robot_pos'] - results['final_target_pos']):.2f}"
    )

    # Belief accuracy
    final_mu, final_Sigma = results["final_belief"]
    belief_error = np.linalg.norm(final_mu - results["final_target_pos"])
    belief_uncertainty = np.sqrt(np.trace(final_Sigma))
    print(f"  Belief error: {belief_error:.2f}")
    print(f"  Belief uncertainty: {belief_uncertainty:.2f}")

    # Sigma statistics
    if len(results["sigma_history"]) > 0:
        sigma_mean = np.mean(results["sigma_history"])
        sigma_std = np.std(results["sigma_history"])
        sigma_final = results["sigma_history"][-1]
        print(
            f"  Sigma statistics: mean={sigma_mean:.3f}, std={sigma_std:.3f}, final={sigma_final:.3f}"
        )

    # R_est statistics
    if len(results["R_est_history"]) > 0:
        R_est_mean = np.mean(results["R_est_history"])
        R_est_std = np.std(results["R_est_history"])
        R_est_final = results["R_est_history"][-1]
        R_est_initial = results["R_est_history"][0]
        print(
            f"  R_est statistics: mean={R_est_mean:.3f}, std={R_est_std:.3f}, initial={R_est_initial:.3f}, final={R_est_final:.3f}"
        )

    return results


def main():
    """Main function to run tracking simulation and create visualizations."""
    print("Target Tracking Simulation (EKF/UKF)")
    print("=====================================")

    # You can choose which configuration to run by uncommenting one of these:

    # Option 1: Default EKF configuration
    config = create_default_config()
    config_name = "Default EKF"

    # Option 2: UKF configuration
    # config = create_ukf_config()
    # config_name = "Default UKF"

    # Option 3: Custom EKF configuration
    # config = create_default_config()
    # config["adaptive_process_noise"] = False  # Turn off adaptive noise
    # config_name = "Non-Adaptive EKF"

    # Option 4: Custom UKF configuration
    # config = create_ukf_config()
    # config["ukf_alpha"] = 0.01  # Larger spread for more exploration
    # config_name = "High-Alpha UKF"

    # Run simulation
    results = run_single_simulation(config, config_name, verbose=True)

    # Create visualizations
    print(f"\nCreating visualizations...")

    # Create the main three-plot figure
    fig_main = visualize_ekf_results(results, plot_type="three_plots", show=False)

    # Show all plots
    plt.show()

    print(f"\nSimulation completed successfully!")
    print(f"Close the plot windows to exit.")


if __name__ == "__main__":
    main()
