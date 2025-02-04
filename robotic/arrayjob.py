#!/usr/bin/env python
import os
import json
import numpy as np
import submitit
from bayes_homing import run_simulation


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
        fixed_traj, _, fixed_env = run_simulation(
            config=self.fixed_config, steps=self.steps
        )
        fixed_steps_to_target = len(fixed_traj)

        # Reset the seed to ensure the adaptive simulation experiences the same random numbers.
        np.random.seed(self.seed)
        adaptive_traj, _, adaptive_env = run_simulation(
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
        output_filename = f"results_seed_{self.seed}.json"
        with open(output_filename, "w") as f:
            json.dump(results, f, indent=2)

        # Return results (Submitit will capture this in the job logs)
        return results


if __name__ == "__main__":
    # Number of runs for each strategy (each run produces one fixed-sigma and one adaptive-sigma result)
    n_runs = 50  # Change this value as needed.
    max_steps = 1000000  # Maximum simulation steps

    # Define the configuration for the fixed-sigma simulation.
    # Here, we force the adaptive sigma to be fixed by setting min_motion_sigma and max_motion_sigma equal.
    fixed_config = {
        "grid_size": 4000,
        "true_motion_sigma": 0.5,
        "min_motion_sigma": 0.5,
        "max_motion_sigma": 0.5,
        "motion_decay_rate": 2.0,  # Irrelevant when min == max
        "measurement_noise_factor": 1e-4,
        "signal_strength_max": 0.5,
        "signal_decay_exp": 0.2,
        "movement_step_size": 1,
        "kernel_size": 5,
        "target_reach_threshold": 1.5,
    }

    # Define the configuration for the adaptive-sigma simulation, same as fixed-sigma but with min_motion_sigma < max_motion_sigma.
    adaptive_config = fixed_config.copy()
    adaptive_config["min_motion_sigma"] = 1e-5

    # Generate a list of seeds so that each run gets a unique seed.
    seeds = list(range(1, n_runs + 1))

    # Create a list of SimulationRunner tasks, one for each seed.
    tasks = [
        SimulationRunner(fixed_config, adaptive_config, seed, steps=max_steps)
        for seed in seeds
    ]

    # Set up the Submitit executor.
    executor = submitit.AutoExecutor(folder="submitit_logs")
    executor.update_parameters(
        timeout_min=1200,  # maximum runtime in minutes
        gpus_per_node=0,  # number of GPUs (0 if not needed)
        cpus_per_task=1,  # number of CPUs per task
        nodes=1,
    )

    # Submit the tasks as an array job.
    job = executor.map_array(lambda task: task(), tasks)

    # Optionally, wait for all results to complete and print a summary.
    results = [future.results() for future in job]
    for res in results:
        print(
            f"Seed {res[0]['seed']}: Fixed steps = {res[0]['fixed_steps_to_target']}, "
            f"Adaptive steps = {res[0]['adaptive_steps_to_target']}"
        )
