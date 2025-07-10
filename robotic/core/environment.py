"""
EKF Environment that orchestrates target tracking simulation.
Main coordinator for Extended Kalman Filter navigation simulation.
Now supports both EKF and UKF through unified interface.
"""

import numpy as np
from filters import FilterPyExtendedKalmanFilter, FilterPyUnscentedKalmanFilter


class EKFEnvironment:
    """Main tracking environment that coordinates target tracking simulation.

    Supports both Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF)
    through unified interface.
    """

    def __init__(self, config, verbose=False, random_seed=None):
        """
        Initialize EKF environment with configuration.

        Args:
            config: Configuration dictionary with simulation parameters
            verbose: Whether to print debug information
            target_rng: Optional random state for target movement (defaults to np.random)
            robot_rng: Optional random state for robot movement and measurements (defaults to np.random)
        """

        self.config = config

        self.verbose = verbose

        # Store random number generators for consistent randomness
        self.target_rng = np.random.default_rng(seed=random_seed)
        self.robot_rng = np.random.default_rng(seed=random_seed + 1000)

        # Initialize positions
        self.robot_pos = np.array(self.config["robot_start_pos"], dtype=float)
        self.target_pos = np.array(self.config["target_true_pos"], dtype=float)

        # Initialize filter (EKF or UKF based on config)
        # Set default filter type if not specified
        if "filter_type" not in self.config:
            self.config["filter_type"] = "EKF"

        # Direct filter instantiation based on filter type
        filter_type = self.config["filter_type"]
        if filter_type == "EKF":
            self.filter = FilterPyExtendedKalmanFilter(self.config)
        elif filter_type == "UKF":
            self.filter = FilterPyUnscentedKalmanFilter(self.config)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}. Use 'EKF' or 'UKF'.")

        # Keep backward compatibility
        self.ekf = self.filter

        # Pre-allocate trajectory storage for better performance
        max_steps = self.config["max_steps"]
        self.robot_trajectory = np.zeros((max_steps + 1, 2))
        self.target_trajectory = np.zeros((max_steps + 1, 2))
        self.robot_trajectory[0] = self.robot_pos
        self.target_trajectory[0] = self.target_pos

        self.belief_history = []  # Keep as list since it stores tuples
        self.measurements = np.zeros(max_steps)
        self.trajectory_length = 1  # Track current length

        if verbose:
            filter_type = self.config["filter_type"]
            print(f"Tracking Environment initialized with {filter_type}:")
            print(f"  Robot start: {self.robot_pos}")
            print(f"  Target start: {self.target_pos}")

            # Show UKF-specific parameters if using UKF
            if "UKF" in filter_type:
                print(f"  UKF alpha: {self.config['ukf_alpha']}")
                print(f"  UKF beta: {self.config['ukf_beta']}")
                print(f"  UKF kappa: {self.config['ukf_kappa']}")

    def _clip_position(self, pos):
        """Clip position to arena boundaries."""
        return np.clip(pos, self.config["arena_min"], self.config["arena_max"])

    def _distance(self, pos1, pos2):
        """Calculate Euclidean distance."""
        diff = pos1 - pos2
        return np.sqrt(diff[0] ** 2 + diff[1] ** 2)

    def get_signal_measurement(self, step):
        """Generate noisy signal measurement at current robot position."""
        # True expected signal using Euclidean distance
        distance = self._distance(self.target_pos, self.robot_pos)
        lambda_true = self.config["signal_max"] * np.exp(
            -self.config["signal_decay"] * distance
        )

        # Poisson noise
        measurement = self.robot_rng.poisson(lambda_true)
        self.measurements[step] = measurement

        return measurement

    def update_belief(self, measurement):
        """Update target belief using EKF."""
        mu, Sigma, sigma_Q = self.ekf.predict_and_update(measurement, self.robot_pos)
        # Only copy if needed for history - EKF already returns copies
        self.belief_history.append((mu, Sigma))
        return mu, Sigma, sigma_Q

    def move_robot(self):
        """Move robot towards current target estimate."""
        # Get current target estimate
        mu, _ = self.ekf.get_belief_state()

        # Calculate direction vector
        direction = mu - self.robot_pos
        if np.linalg.norm(direction) > 1e-6:
            # Move towards estimate
            self.robot_pos += (
                self.config["robot_step_size"] * direction / np.linalg.norm(direction)
            )

        # Add actuator noise
        self.robot_pos += self.robot_rng.normal(0, self.config["actuator_noise"], 2)

        # Apply boundary conditions (clip to arena)
        self.robot_pos = self._clip_position(self.robot_pos)

        # Store in pre-allocated array
        self.robot_trajectory[self.trajectory_length] = self.robot_pos

    def move_target(self):
        """Move target with random walk."""
        self.target_pos += self.target_rng.normal(
            0, self.config["target_motion_sigma"], 2
        )

        # Apply boundary conditions (clip to arena)
        self.target_pos = self._clip_position(self.target_pos)

        # Store in pre-allocated array
        self.target_trajectory[self.trajectory_length] = self.target_pos

    def check_target_reached(self):
        """Check if robot is close enough to target."""
        distance = self._distance(self.robot_pos, self.target_pos)
        return distance < self.config["distance_tolerance"]

    def run_simulation(self):
        """
        Run complete EKF simulation.

        Returns:
            dict: Simulation results including trajectories and final state
        """
        max_steps = self.config["max_steps"]

        if self.verbose:
            print(f"Starting EKF simulation for up to {max_steps} steps...")

        for step in range(max_steps):
            # 1. Robot senses environment
            measurement = self.get_signal_measurement(step)

            # 2. Update belief with EKF
            mu, Sigma, sigma_Q = self.update_belief(measurement)

            # 3. Move robot towards estimated target
            self.move_robot()

            # 4. Move target randomly
            self.move_target()

            # 5. Update trajectory length
            self.trajectory_length += 1

            # 6. Check termination condition
            if self.check_target_reached():
                if self.verbose:
                    print(f"Target reached at step {step}!")
                    print(f"Final target position: {self.target_pos}")
                    print(f"Final robot position: {self.robot_pos}")
                break

            # Progress reporting
            if self.verbose and step % 10000 == 0:
                distance = self._distance(self.robot_pos, self.target_pos)
                print(f"Step {step}: Distance to target = {distance:.2f}")

        # Prepare results - slice arrays to actual length used
        results = {
            "steps_completed": step + 1,
            "target_reached": self.check_target_reached(),
            "robot_trajectory": self.robot_trajectory[: self.trajectory_length],
            "target_trajectory": self.target_trajectory[: self.trajectory_length],
            "belief_history": self.belief_history,
            "measurements": self.measurements[: step + 1],
            "sigma_history": self.ekf.get_sigma_history(),
            "R_est_history": self.ekf.get_R_est_history(),
            "final_belief": self.ekf.get_belief_state(),
            "final_target_pos": self.target_pos.copy(),
            "final_robot_pos": self.robot_pos.copy(),
            "config": self.config,
        }

        if self.verbose:
            print(f"Simulation completed in {results['steps_completed']} steps")
            print(f"Target reached: {results['target_reached']}")

        return results

    def reset(self):
        """Reset environment to initial state."""
        # Don't reset random seed here - let comparison script control it

        # Reset positions
        self.robot_pos = np.array(self.config["robot_start_pos"], dtype=float)
        self.target_pos = np.array(self.config["target_true_pos"], dtype=float)

        # Reset EKF
        self.ekf.reset()

        # Reset pre-allocated arrays
        max_steps = self.config["max_steps"]
        self.robot_trajectory = np.zeros((max_steps + 1, 2))
        self.target_trajectory = np.zeros((max_steps + 1, 2))
        self.robot_trajectory[0] = self.robot_pos
        self.target_trajectory[0] = self.target_pos

        self.belief_history = []
        self.measurements = np.zeros(max_steps)
        self.trajectory_length = 1

    def compute_signal_field(self, target_pos=None):
        """
        Compute signal field over the entire arena for visualization.

        Args:
            target_pos: Target position for signal computation (uses current if None)

        Returns:
            tuple: (x_grid, y_grid, signal_grid)
        """
        if target_pos is None:
            target_pos = self.target_pos

        # Create grid
        arena_range = self.config["arena_max"] - self.config["arena_min"]
        grid_size = int(arena_range)  # 1 unit resolution
        x = np.linspace(self.config["arena_min"], self.config["arena_max"], grid_size)
        y = np.linspace(self.config["arena_min"], self.config["arena_max"], grid_size)
        X, Y = np.meshgrid(x, y)

        # Vectorized signal computation
        target_pos = np.array(target_pos)

        # Standard Euclidean distance
        distances = np.sqrt((X - target_pos[0]) ** 2 + (Y - target_pos[1]) ** 2)

        # Vectorized signal calculation
        signal_grid = self.config["signal_max"] * np.exp(
            -self.config["signal_decay"] * distances
        )

        return X, Y, signal_grid
