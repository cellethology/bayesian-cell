from bayes_homing import run_simulation
import numpy as np
import json
from multiprocessing import Pool, cpu_count
from functools import partial
import concurrent.futures


def run_single_simulation(config, max_steps=20000):
    """
    Run a single simulation and return its results
    """
    trajectory, _, _ = run_simulation(config=config, steps=max_steps)
    steps_taken = len(trajectory)
    return steps_taken


def run_single_simulation_wrapper(args):
    config, max_steps, _ = args
    return run_single_simulation(config=config, max_steps=max_steps)


def run_simulations_for_step_size(args):
    """
    Run all simulations for a single step size
    """
    base_config, step_size, n_runs, max_steps = args

    print(f"\nTesting movement_step_size: {step_size}")
    config = base_config.copy()
    config["movement_step_size"] = step_size

    # Prepare arguments for the wrapper function
    sim_args = [(config, max_steps, i) for i in range(n_runs)]

    # Run simulations in parallel for this step size
    with Pool() as pool:
        steps_list = pool.map(run_single_simulation_wrapper, sim_args)

    # Process results
    steps_to_target = [s for s in steps_list if s != max_steps + 1]
    failed_runs = sum(1 for s in steps_list if s == max_steps + 1)

    return {
        str(step_size): {
            "mean_steps": float(np.mean(steps_to_target)) if steps_to_target else None,
            "median_steps": (
                float(np.median(steps_to_target)) if steps_to_target else None
            ),
            "std_steps": float(np.std(steps_to_target)) if steps_to_target else None,
            "failed_runs": failed_runs,
            "successful_runs": len(steps_to_target),
            "all_steps": steps_to_target,
        }
    }


if __name__ == "__main__":
    # Base configuration
    base_config = {
        "grid_size": 100,
        "target_pos": (75, 75),
        "true_motion_sigma": 0.5,
        "max_motion_sigma": 0.5,
        "motion_decay_rate": 2,
        "measurement_noise_factor": 1e-1,
        "signal_strength_max": 1,
        "signal_decay_exp": 0.3,
        "movement_step_size": 0.02,  # This will be overridden
        "kernel_size": 5,
        "target_reach_threshold": 5.0,
        "min_motion_sigma": 1e-5,
    }

    # Define step sizes to test
    step_sizes = [0.1]
    n_runs = 24
    max_steps = 3000000

    # Define min_motion_sigma values to test
    min_motion_sigma_values = [1e-5, 0.5]

    for min_motion_sigma in min_motion_sigma_values:
        base_config["max_motion_sigma"] = min_motion_sigma

        # Prepare arguments for parallel processing
        args_list = [
            (base_config, step_size, n_runs, max_steps) for step_size in step_sizes
        ]

        # Run parallel simulations for each step size
        print(
            f"Running simulations for max_motion_sigma={min_motion_sigma} using {cpu_count()} CPU cores..."
        )
        np.random.seed(82)  # For reproducibility

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results_list = list(executor.map(run_simulations_for_step_size, args_list))

        # Combine results
        results = {}
        for r in results_list:
            results.update(r)

        # Save results and config to file
        output_data = {
            "config": base_config,
            "results": results,
        }
        output_filename = f"results_max_motion_sigma_{min_motion_sigma}.json"
        with open(output_filename, "w") as f:
            json.dump(output_data, f, indent=4)

        print(f"Results saved to {output_filename}")
