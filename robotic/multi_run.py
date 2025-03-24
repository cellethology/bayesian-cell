from robot_sim.bayes_homing import run_simulation
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def run_multiple_simulations(base_config, n_runs=100, max_steps=20000):
    """
    Run multiple simulations with given configuration
    Returns number of steps to reach target for each run
    """
    steps_to_target = []
    failed_runs = 0

    for i in range(n_runs):
        if i % 10 == 0:  # Progress indicator
            print(f"Run {i}/{n_runs}")

        trajectory, _, _ = run_simulation(config=base_config, steps=max_steps)
        steps_taken = len(trajectory)

        # If steps_taken equals max_steps, the run didn't reach target
        if steps_taken == max_steps + 1:
            failed_runs += 1
        else:
            steps_to_target.append(steps_taken)

    return np.array(steps_to_target), failed_runs


if __name__ == "__main__":
    # Base configuration
    base_config = {
        "grid_size": 100,
        "target_pos": (75, 75),
        "true_motion_sigma": 0.5,
        "max_motion_sigma": 0.5,
        "motion_decay_rate": 2,
        "measurement_noise_factor": 1e-4,
        "signal_strength_max": 1,
        "signal_decay_exp": 0.3,
        "movement_step_size": 0.02,
        "kernel_size": 5,
        "target_reach_threshold": 5.0,
    }

    # Create two configurations
    adaptive_config = base_config.copy()
    adaptive_config["min_motion_sigma"] = 0.1

    nonadaptive_config = base_config.copy()
    nonadaptive_config["min_motion_sigma"] = nonadaptive_config["max_motion_sigma"]

    # Run simulations for both configurations
    n_runs = 1
    print("Running adaptive strategy simulations...")
    np.random.seed(82)
    adaptive_steps, adaptive_fails = run_multiple_simulations(adaptive_config, n_runs)

    print("\nRunning non-adaptive strategy simulations...")
    np.random.seed(82)
    nonadaptive_steps, nonadaptive_fails = run_multiple_simulations(
        nonadaptive_config, n_runs
    )

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot histograms
    bins = np.linspace(
        0, max(np.max(adaptive_steps), np.max(nonadaptive_steps)) + 5, 30
    )

    ax1.hist(
        adaptive_steps,
        bins=bins,
        alpha=0.5,
        label=f"Adaptive (fails={adaptive_fails})",
        density=True,
    )
    ax1.hist(
        nonadaptive_steps,
        bins=bins,
        alpha=0.5,
        label=f"Non-adaptive (fails={nonadaptive_fails})",
        density=True,
    )
    ax1.set_title("Distribution of Steps to Target")
    ax1.set_xlabel("Number of Steps")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.grid(True)

    # Box plot
    box_data = [adaptive_steps, nonadaptive_steps]
    ax2.boxplot(box_data, labels=["Adaptive", "Non-adaptive"])
    ax2.set_title("Steps to Target Comparison")
    ax2.set_ylabel("Number of Steps")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Print statistics
    print("\nStatistics:")
    print("\nAdaptive Strategy:")
    print(f"Mean steps: {np.mean(adaptive_steps):.2f}")
    print(f"Median steps: {np.median(adaptive_steps):.2f}")
    print(f"Std steps: {np.std(adaptive_steps):.2f}")
    print(f"Failed runs: {adaptive_fails}/{n_runs}")

    print("\nNon-adaptive Strategy:")
    print(f"Mean steps: {np.mean(nonadaptive_steps):.2f}")
    print(f"Median steps: {np.median(nonadaptive_steps):.2f}")
    print(f"Std steps: {np.std(nonadaptive_steps):.2f}")
    print(f"Failed runs: {nonadaptive_fails}/{n_runs}")

    # Perform paired t test
    _, p_value = stats.ttest_rel(adaptive_steps, nonadaptive_steps)

    print(f"\nt-test p-value: {p_value:.4f}")
