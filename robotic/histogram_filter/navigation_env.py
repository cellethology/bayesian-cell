"""
Navigation environment that orchestrates all components.
Main coordinator for Bayesian navigation simulation.
"""

import numpy as np
from signal_model import SignalModel
from motion_model import MotionModel
from bayesian_filter import BayesianFilter


class NavigationEnvironment:
    """Main navigation environment that coordinates all subsystems."""

    def __init__(self, config=None, verbose=False):
        # Set up configuration with defaults
        self.config = {
            "grid_size": 100,
            "initial_belief": None,
            "true_target_pos": None,
            "process_sigma": 0.5,  # Robot actuator noise sigma
            "target_motion_sigma": 0.1,  # Target random walk sigma
            "min_motion_sigma": 0.01,  # Minimum target motion uncertainty
            "adaptive_rate": 1,  # Target motion adaptation rate
            "adaptive_decay_type": "power_law",
            "power_exponent": 1.0,
            "signal_max": 1,
            "signal_decay": 0.3,
            "step_size": 0.1,
            "kernel_size": 5,
            "target_reach_threshold": 2.0,
            "innovation_window_size": 20,
            "adaptation_rate": 0.4,
            "min_allowed_variance": 1e-6,
            "adaptive_filtering": False,  # Adaptive measurement noise
            "adaptive_process_variance": False,  # Adaptive target motion uncertainty
            "noise_model": "poisson",
            "noise_std": 0.06,
        }

        if config is not None:
            self.config.update(config)

        if self.config["noise_model"] == "poisson":
            self.config.pop("noise_std", None)

        # Set up target position
        if self.config["true_target_pos"] is None:
            grid_size = self.config["grid_size"]
            self.true_target_pos = (grid_size * 3 // 4, grid_size * 3 // 4)
            self.config["true_target_pos"] = self.true_target_pos
        else:
            self.true_target_pos = self.config["true_target_pos"]

        # Set default starting position if not specified
        if "start_pos" not in self.config:
            grid_size = self.config["grid_size"]
            self.config["start_pos"] = (grid_size // 4, grid_size // 4)

        # Set intelligent defaults for estimation parameters
        # Note: process_sigma_estimate will be set after signal_model is initialized

        # Default: noise_estimate = sqrt(expected signal at starting position)
        if "noise_estimate" not in self.config:
            start_pos = self.config["start_pos"]
            target_pos = self.true_target_pos

            # Calculate distance from start to target
            distance = np.sqrt(
                (start_pos[0] - target_pos[0]) ** 2
                + (start_pos[1] - target_pos[1]) ** 2
            )

            # Calculate expected signal using signal model formula
            expected_signal = self.config["signal_max"] * np.exp(
                -self.config["signal_decay"] * distance
            )

            # Set noise_estimate to sqrt of expected signal (reasonable default)
            self.config["noise_estimate"] = np.sqrt(expected_signal)
            if verbose:
                print(
                    f"Setting noise_estimate = {self.config['noise_estimate']:.6f} (sqrt of expected signal {expected_signal:.6f} at start pos {start_pos})"
                )

        # Initialize subsystems (removed GridCache for simplicity)
        self.signal_model = SignalModel(self.config)
        self.motion_model = MotionModel(self.config)

        # Set process_sigma_estimate now that signal_model is available
        if "process_sigma_estimate" not in self.config:
            # Apply signal-based scaling to target motion uncertainty
            if self.config["adaptive_process_variance"]:
                scaling_factor = self.config["noise_estimate"]
                self.config["process_sigma_estimate"] = (
                    self.config["target_motion_sigma"] * self.config["noise_estimate"]
                )
            else:
                self.config["process_sigma_estimate"] = self.config[
                    "target_motion_sigma"
                ]

            if verbose:
                print(
                    f"Setting process_sigma_estimate = {self.config['process_sigma_estimate']:.4f}"
                )
                print(
                    f"  target_motion_sigma: {self.config['target_motion_sigma']:.4f}"
                )
                print(f"  mean_signal: {mean_signal:.4f}")
                print(f"  scaling_factor: {scaling_factor:.4f}")

        self.bayesian_filter = BayesianFilter(
            self.config, self.signal_model, self.motion_model
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

    def update_target_position(self):
        """Update target position with random Gaussian motion."""
        # Check if we have pre-generated target motion (for deterministic comparisons)
        if "_target_random_x" in self.config and "_target_random_y" in self.config:
            idx = self.config["_target_random_index"]
            if idx < len(self.config["_target_random_x"]):
                dx = self.config["_target_random_x"][idx]
                dy = self.config["_target_random_y"][idx]
                self.config["_target_random_index"] = idx + 1
            else:
                # Fallback if we run out of pre-generated values
                sigma = self.config["target_motion_sigma"]
                dx = np.random.normal(0, sigma)
                dy = np.random.normal(0, sigma)
        else:
            # Normal random generation
            sigma = self.config["target_motion_sigma"]
            dx = np.random.normal(0, sigma)
            dy = np.random.normal(0, sigma)

        # Update target position with boundary constraints
        new_x = np.clip(self.true_target_pos[0] + dx, 0, self.config["grid_size"] - 1)
        new_y = np.clip(self.true_target_pos[1] + dy, 0, self.config["grid_size"] - 1)

        self.true_target_pos = (new_x, new_y)
        self.config["true_target_pos"] = self.true_target_pos


def run_navigation_simulation(config=None, steps=100, verbose=False):
    """Run a complete navigation simulation."""
    env = NavigationEnvironment(config, verbose=verbose)

    robot_pos = env.config["start_pos"]
    trajectory = [robot_pos]
    target_trajectory = [env.true_target_pos]
    sigmas = [env.config["process_sigma_estimate"]]
    innovations = []
    measurement_variances = []
    success_target_pos = None  # Target position when robot first reaches it

    for step in range(steps):
        # Update target position with random motion
        env.update_target_position()

        measurement = env.get_noisy_measurement(robot_pos, env.true_target_pos)
        innovation, measure_var = env.update_belief(measurement, robot_pos)
        sigma = env.motion_update(measurement, robot_pos)
        action = env.get_next_intended_action(robot_pos)
        if np.any(np.isnan(action)):
            print(f"Step {step}: NaN action detected: {action}")
            print(f"Belief has NaN: {np.any(np.isnan(env.belief))}")
            print(f"Belief sum: {np.sum(env.belief)}")
        robot_pos = env.update_position(robot_pos, action)

        # No need for error-based adaptation tracking with simplified boolean approach

        trajectory.append(robot_pos)
        target_trajectory.append(env.true_target_pos)
        sigmas.append(sigma)
        innovations.append(innovation)
        measurement_variances.append(measure_var)

        distance_to_target = np.linalg.norm(
            np.array(robot_pos) - np.array(env.true_target_pos)
        )
        if distance_to_target < env.config["target_reach_threshold"]:
            if success_target_pos is None:  # Record target position on first success
                success_target_pos = env.true_target_pos
            if verbose:
                print(f"Target reached in {len(trajectory)} steps!")
            break
        if step % 2000 == 0 and verbose:
            print(f"Distance to target at step {step}: {distance_to_target:.2f}")

    return (
        trajectory,
        target_trajectory,
        env,
        sigmas,
        innovations,
        measurement_variances,
        success_target_pos,
    )


if __name__ == "__main__":
    # Example usage with visualization
    from visualization import visualize_simulation_results

    np.random.seed(4)
    example_config = {
        "grid_size": 100,
        "process_sigma": 0.3,
        "target_motion_sigma": 0.5,
        "signal_max": 5,
        "signal_decay": 0.05,
        "step_size": 0.2,
        "kernel_size": 5,
        "adaptive_filtering": False,
        "adaptive_process_variance": True,
        "adaptive_decay_type": "power_law",
        "adaptive_rate": 1,
        "power_exponent": 0.5,
        "noise_model": "poisson",
        # process_sigma_estimate and noise_estimate will be set intelligently by default
    }

    (
        trajectory,
        target_trajectory,
        env,
        sigmas,
        innovations,
        measurement_variances,
        success_target_pos,
    ) = run_navigation_simulation(config=example_config, steps=10000, verbose=True)

    # Create comprehensive visualization
    visualize_simulation_results(
        env,
        trajectory,
        target_trajectory,
        sigmas,
        innovations,
        measurement_variances,
        plot_type="comprehensive",
    )
