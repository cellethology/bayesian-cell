"""
Navigation environment that orchestrates all components.
Main coordinator for Bayesian navigation simulation.
"""

import numpy as np
from utils import GridCache
from signal_model import SignalModel
from motion_model import MotionModel
from bayesian_filter import BayesianFilter


class NavigationEnvironment:
    """Main navigation environment that coordinates all subsystems."""

    def __init__(self, config=None):
        # Set up configuration with defaults
        self.config = {
            "grid_size": 100,
            "initial_belief": None,
            "true_target_pos": None,
            "true_process_sigma": 0.5,
            "min_motion_sigma": 0.1,
            "motion_decay_rate": 8,
            "signal_strength_max": 1,
            "signal_decay_exp": 0.3,
            "step_size": 0.1,
            "kernel_size": 5,
            "target_reach_threshold": 2.0,
            "innovation_window_size": 20,
            "adaptation_rate": 0.4,
            "initial_measurement_sigma": 0.5,
            "initial_process_sigma": 0.1,
            "min_allowed_variance": 1e-6,
            "adaptive_filtering": False,
            "adaptive_process_variance": "none",
            "noise_model": "poisson",
            "noise_std": 1,
        }

        if config is not None:
            self.config.update(config)

        # Set up target position
        if self.config["true_target_pos"] is None:
            grid_size = self.config["grid_size"]
            self.true_target_pos = (grid_size * 4 // 5, grid_size * 4 // 5)
            self.config["true_target_pos"] = self.true_target_pos
        else:
            self.true_target_pos = self.config["true_target_pos"]

        # Initialize subsystems
        self.grid_cache = GridCache(self.config["grid_size"])
        self.signal_model = SignalModel(self.config, self.grid_cache)
        self.motion_model = MotionModel(self.config)
        self.bayesian_filter = BayesianFilter(
            self.config, self.grid_cache, self.signal_model, self.motion_model
        )

        # Set adaptive process variance flag
        self.adaptive_process_variance = self.config["adaptive_process_variance"]

    @property
    def belief(self):
        """Access to current belief state."""
        return self.bayesian_filter.belief

    @property
    def _cache_hit_count(self):
        """Motion kernel cache statistics."""
        return self.motion_model.cache_hit_count

    @property
    def _cache_miss_count(self):
        """Motion kernel cache statistics."""
        return self.motion_model.cache_miss_count

    def get_noisy_measurement(self, pos, target_pos):
        """Get noisy measurement at given position."""
        return self.signal_model.get_noisy_measurement(pos, target_pos)

    def update_belief(self, measurement, robot_pos):
        """Update belief based on measurement."""
        return self.bayesian_filter.update_belief(measurement, robot_pos)

    def motion_update(self, current_signal, current_pos=None):
        """Apply motion update to belief."""
        return self.bayesian_filter.motion_update(
            current_signal, current_pos, self.adaptive_process_variance
        )

    def get_next_intended_action(self, robot_pos):
        """Get next intended action."""
        return self.motion_model.get_next_intended_action(robot_pos, self.belief)

    def update_position(self, robot_pos, action):
        """Update robot position based on action."""
        return self.motion_model.update_position(robot_pos, action)

    def update_process_variance_from_motion_error(self, robot_pos):
        """Update process variance based on motion error (for error_based adaptation)."""
        self.motion_model.update_process_variance_from_motion_error(robot_pos)


def run_navigation_simulation(config=None, steps=100, verbose=False):
    """Run a complete navigation simulation."""
    env = NavigationEnvironment(config)

    robot_pos = (env.config["grid_size"] // 5, env.config["grid_size"] // 5)
    trajectory = [robot_pos]
    sigmas = [env.config["initial_process_sigma"]]
    innovations = []
    measurement_variances = []

    for step in range(steps):
        measurement = env.get_noisy_measurement(robot_pos, env.true_target_pos)
        innovation, measure_var = env.update_belief(measurement, robot_pos)
        sigma = env.motion_update(measurement, robot_pos)
        action = env.get_next_intended_action(robot_pos)
        robot_pos = env.update_position(robot_pos, action)

        # Update process variance based on actual motion error (for error_based adaptation)
        if env.adaptive_process_variance == "error_based":
            env.update_process_variance_from_motion_error(robot_pos)

        trajectory.append(robot_pos)
        sigmas.append(sigma)
        innovations.append(innovation)
        measurement_variances.append(measure_var)

        distance_to_target = np.linalg.norm(
            np.array(robot_pos) - np.array(env.true_target_pos)
        )
        if distance_to_target < env.config["target_reach_threshold"]:
            if verbose:
                print(f"Target reached in {len(trajectory)} steps!")
            break
        if step % 2000 == 0 and verbose:
            print(f"Distance to target at step {step}: {distance_to_target:.2f}")

    return trajectory, env, sigmas, innovations, measurement_variances


if __name__ == "__main__":
    # Example usage with visualization
    from visualization import visualize_simulation_results

    np.random.seed(2)
    example_config = {
        "grid_size": 100,
        "motion_noise_type": "angular",
        "angular_noise_sigma": 0.5,
        "magnitude_noise_sigma": 0.0,
        "initial_process_sigma": 0.5,
        "motion_decay_rate": 0.8,
        "signal_strength_max": 0.2,
        "signal_decay_exp": 0.3,
        "step_size": 0.2,
        "kernel_size": 5,
        "adaptive_filtering": False,
        "adaptive_process_variance": "none",
        "noise_model": "gaussian",
        "noise_std": 0.06,
        "initial_measurement_sigma": 0.06,
    }

    trajectory, env, sigmas, innovations, measurement_variances = (
        run_navigation_simulation(config=example_config, steps=50000, verbose=True)
    )

    # Create comprehensive visualization
    visualize_simulation_results(
        env,
        trajectory,
        sigmas,
        innovations,
        measurement_variances,
        plot_type="comprehensive",
    )
