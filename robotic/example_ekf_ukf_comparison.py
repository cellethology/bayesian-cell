#!/usr/bin/env python3
"""
Example script demonstrating EKF vs UKF comparison.
Shows how to easily switch between filter types and compare performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from filter_factory import FilterFactory, get_default_config
from ekf_environment import EKFEnvironment
from ekf_comparison import EKFComparison


def run_simple_comparison():
    """Run a simple comparison between EKF and UKF."""
    print("=" * 60)
    print("SIMPLE EKF vs UKF COMPARISON")
    print("=" * 60)

    # Base configuration for both filters
    base_config = {
        "arena_min": 0.0,
        "arena_max": 100.0,
        "distance_tolerance": 3.0,
        "signal_max": 40.0,
        "signal_decay": 0.05,
        "robot_start_pos": [20.0, 20.0],
        "robot_step_size": 0.5,
        "actuator_noise": 0.3,
        "target_true_pos": [80.0, 80.0],
        "initial_belief_mean": [50.0, 50.0],
        "initial_belief_variance": 200.0,
        "target_motion_sigma": 0.2,
        "baseline_process_noise": 0.2,
        "adaptive_process_noise": False,
        "max_steps": 400000,
    }

    # Configurations to compare
    configs = {
        "EKF": {
            "filter_type": "EKF",
        },
        "UKF": {
            "filter_type": "UKF",
            "ukf_alpha": 0.001,
            "ukf_beta": 0.02,
            "ukf_kappa": 1.0,
        },
    }

    # Run comparison
    comparison = EKFComparison(base_config)
    for name, config in configs.items():
        comparison.add_config(name, config)

    print("Running comparison (this may take a moment)...")
    results = comparison.run_comparison(n_runs=10, max_steps=10000, verbose=False)

    # Create plots
    fig = comparison.create_comparison_plots(results, figsize=(12, 4))
    plt.suptitle("EKF vs UKF Performance Comparison", fontsize=14)
    plt.tight_layout()
    plt.show()

    return results


def run_single_filter_demo(filter_type="EKF"):
    """Demonstrate running a single filter."""
    print(f"\n{filter_type} SINGLE RUN DEMONSTRATION")
    print("=" * 40)

    # Get default configuration for the specified filter type
    config = get_default_config(filter_type)

    # Adjust for quick demo
    config.update(
        {
            "arena_min": 0.0,
            "arena_max": 100.0,
            "robot_start_pos": [30.0, 30.0],
            "target_true_pos": [70.0, 70.0],
            "max_steps": 5000,
            "random_seed": 42,
        }
    )

    print(f"Running {filter_type} simulation...")

    # Set random seed for reproducible results
    np.random.seed(config["random_seed"])

    # Create environment and run simulation
    env = EKFEnvironment(config, verbose=True)
    results = env.run_simulation()

    print(f"\n{filter_type} Results:")
    print(f"  Steps completed: {results['steps_completed']}")
    print(f"  Target reached: {results['target_reached']}")

    final_distance = np.linalg.norm(
        results["final_robot_pos"] - results["final_target_pos"]
    )
    print(f"  Final distance: {final_distance:.2f}")

    # Belief accuracy
    final_mu, final_Sigma = results["final_belief"]
    belief_error = np.linalg.norm(final_mu - results["final_target_pos"])
    belief_uncertainty = np.sqrt(np.trace(final_Sigma))
    print(f"  Belief error: {belief_error:.2f}")
    print(f"  Belief uncertainty: {belief_uncertainty:.2f}")

    return results


def demonstrate_filter_switching():
    """Demonstrate how easy it is to switch between filters."""
    print("\n" + "=" * 60)
    print("FILTER SWITCHING DEMONSTRATION")
    print("=" * 60)

    # Same base configuration
    base_config = {
        "arena_min": 0.0,
        "arena_max": 50.0,
        "robot_start_pos": [10.0, 10.0],
        "target_true_pos": [40.0, 40.0],
        "signal_max": 15.0,
        "max_steps": 1000,
        "random_seed": 123,
    }

    # Demonstrate switching with just one parameter
    print("Creating EKF...")
    ekf_config = base_config.copy()
    ekf_config["filter_type"] = "EKF"
    ekf_filter = FilterFactory.create_filter(ekf_config)
    print(f"Created: {type(ekf_filter).__name__}")

    print("\nCreating UKF...")
    ukf_config = base_config.copy()
    ukf_config["filter_type"] = "UKF"
    ukf_filter = FilterFactory.create_filter(ukf_config)
    print(f"Created: {type(ukf_filter).__name__}")

    # Show UKF-specific features
    if hasattr(ukf_filter, "get_ukf_parameters"):
        ukf_params = ukf_filter.get_ukf_parameters()
        print(f"UKF parameters: {ukf_params}")

    print("\nFilter switching is that easy!")


def main():
    """Main demonstration function."""
    print("EKF/UKF Integration Demonstration")
    print("=" * 50)

    # Run comparison
    comparison_results = run_simple_comparison()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("Key takeaways:")
    print("1. Switch between EKF and UKF by changing 'filter_type' in config")
    print("2. UKF typically provides better accuracy for nonlinear systems")
    print("3. EKF is computationally faster")
    print("4. Both filters share the same interface and can be compared easily")


if __name__ == "__main__":
    main()
