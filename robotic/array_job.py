#!/usr/bin/env python
import json
import numpy as np
import submitit
from bayes_navigation import run_navigation_simulation


class SimulationRunner:
    def __init__(self, fixed_config, adaptive_config, seed, steps=1000):
        """
        Args:
            fixed_config (dict): Configuration for fixed-sigma simulation.
            adaptive_config (dict): Configuration for adaptive-sigma simulation.
            seed (int): Random seed for reproducibility.
            steps (int): Maximum simulation steps.
        """
        self.fixed_config = fixed_config
        self.adaptive_config = adaptive_config
        self.seed = seed
        self.steps = steps

    def __call__(self):
        np.random.seed(self.seed)
        # Run simulation with the fixed-sigma configuration.
        fixed_traj, fixed_env, _ = run_navigation_simulation(
            config=self.fixed_config, steps=self.steps
        )
        fixed_steps_to_target = len(fixed_traj)

        # Reset the seed to ensure the adaptive simulation experiences the same random numbers.
        np.random.seed(self.seed)
        adaptive_traj, adaptive_env, _ = run_navigation_simulation(
            config=self.adaptive_config, steps=self.steps
        )
        adaptive_steps_to_target = len(adaptive_traj)

        # Collect the results
        results = {
            "seed": self.seed,
            "fixed_config": fixed_env.config,
            "adaptive_config": adaptive_env.config,
            "fixed_steps_to_target": fixed_steps_to_target,
            "adaptive_steps_to_target": adaptive_steps_to_target,
        }

        # Save the results to a file named using the seed.
        output_filename = f"tmp/results_seed_{self.seed}.json"
        with open(output_filename, "w") as f:
            json.dump(results, f, indent=2)

        # Return results (Submitit will capture this in the job logs)
        return results


if __name__ == "__main__":
    # Number of runs for each strategy (each run produces one fixed-sigma and one adaptive-sigma result)
    n_runs = 50  # Number of repeated runs
    max_steps = 100000000  # Maximum simulation steps

    # Define the range of measurement_noise_factor values to test
    measurement_noise_factors = np.logspace(-3, -1, num=5)

    # Generate a list of seeds so that each run gets a unique seed.
    seeds = list(range(1, n_runs + 1))

    # Create a list to hold all tasks
    tasks = []

    for measurement_noise_factor in measurement_noise_factors:
        # Define the configuration for the fixed-sigma simulation.
        fixed_config = {
            "grid_size": 100,
            "true_motion_sigma": 0.5,
            "min_motion_sigma": 0.5,
            "max_motion_sigma": 0.5,
            "motion_decay_rate": 4.0,  # Irrelevant when min == max
            "measurement_noise_factor": measurement_noise_factor,
            "signal_strength_max": 0.2,
            "signal_decay_exp": 0.3,
            "movement_step_size": 0.02,
            "kernel_size": 5,
            "target_reach_threshold": 5.0,
        }

        # Define the configuration for the adaptive-sigma simulation
        adaptive_config = fixed_config.copy()
        adaptive_config["min_motion_sigma"] = 1e-5

        # Create a list of SimulationRunner tasks for each seed and measurement_noise_factor
        tasks.extend(
            [
                SimulationRunner(fixed_config, adaptive_config, seed, steps=max_steps)
                for seed in seeds
            ]
        )

    # Set up the Submitit executor.
    executor = submitit.AutoExecutor(folder="tmp/submitit_logs")
    executor.update_parameters(
        timeout_min=240,  # maximum runtime in minutes
        gpus_per_node=0,  # number of GPUs (0 if not needed)
        cpus_per_task=1,  # number of CPUs per task
        nodes=1,
    )

    # Submit the tasks as an array job.
    job = executor.map_array(lambda task: task(), tasks)

    # Print a summary.
    results = [future.results() for future in job]
    for res in results:
        print(
            f"Seed {res[0]['seed']}: Fixed steps = {res[0]['fixed_steps_to_target']}, "
            f"Adaptive steps = {res[0]['adaptive_steps_to_target']}, "
            f"Measurement Noise Factor = {res[0]["fixed_config"]['measurement_noise_factor']}"
        )

    # Organize the results into two dictionaries, one for fixed-sigma and one for adaptive-sigma. For each dictionary, the keys are the measurement noise factors and the values are lists of the number of steps to reach the target for each seed.
    fixed_results = {mnf: [] for mnf in measurement_noise_factors}
    adaptive_results = {mnf: [] for mnf in measurement_noise_factors}
    for res in results:
        fixed_results[res[0]["fixed_config"]["measurement_noise_factor"]].append(
            res[0]["fixed_steps_to_target"]
        )
        adaptive_results[res[0]["adaptive_config"]["measurement_noise_factor"]].append(
            res[0]["adaptive_steps_to_target"]
        )

    # Save the results to a file.
    with open("tmp/fixed_results.json", "w") as f:
        json.dump(fixed_results, f, indent=2)
    with open("tmp/adaptive_results.json", "w") as f:
        json.dump(adaptive_results, f, indent=2)

    print("All done!")
