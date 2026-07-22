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

        # Initialize spatial correlation field if enabled
        self.correlation_field = None
        self.correlation_interpolator = None
        if self.config.get("spatial_correlation_length", 0.0) > 0:
            self._initialize_correlation_field()

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
        lambda_true = self.ekf._h(self.target_pos, self.robot_pos)

        # Apply spatial correlation if enabled
        if self.correlation_interpolator is not None:
            # Get correlation value at robot position
            correlation_value = self.correlation_interpolator(
                [self.robot_pos[1], self.robot_pos[0]]
            )[0]

            # Apply log-normal transformation for correlated Poisson
            correlation_strength = self.config.get("spatial_correlation_strength", 0.5)
            log_lambda = np.log(lambda_true + 1e-10)  # Avoid log(0)
            correlated_log_lambda = (
                log_lambda + correlation_strength * correlation_value
            )
            lambda_correlated = np.exp(correlated_log_lambda)

            measurement = self.robot_rng.poisson(lambda_correlated)
        else:
            # Original independent Poisson sampling
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

        # Re-initialize correlation field if enabled
        if self.config.get("spatial_correlation_length", 0.0) > 0:
            self._initialize_correlation_field()

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

    def _initialize_correlation_field(self):
        """Initialize the spatial correlation field and interpolator."""
        X, Y, gaussian_field = self._generate_correlation_field()

        if gaussian_field is not None:
            # Store the field and set up interpolation
            self.correlation_field = gaussian_field

            # Create interpolator for fast lookup during simulation
            from scipy.interpolate import RegularGridInterpolator

            x_coords = X[0, :]  # First row gives x coordinates
            y_coords = Y[:, 0]  # First column gives y coordinates

            self.correlation_interpolator = RegularGridInterpolator(
                (y_coords, x_coords),  # Note: (y, x) order for RegularGridInterpolator
                gaussian_field,
                method="linear",
                bounds_error=False,
                fill_value=0.0,  # Outside arena gets zero correlation
            )

    def _generate_correlation_field(self):
        """Generate spatially correlated noise field for the entire arena."""
        if self.config["spatial_correlation_length"] <= 0:
            return None, None, None  # No correlation

        # Create coordinate grid for the entire arena
        arena_min = self.config["arena_min"]
        arena_max = self.config["arena_max"]

        # Use higher resolution for correlation field (can be tuned)
        grid_points = 100  # Fixed grid size for consistency
        x = np.linspace(arena_min, arena_max, grid_points)
        y = np.linspace(arena_min, arena_max, grid_points)
        X, Y = np.meshgrid(x, y)

        # Generate correlated Gaussian field using spectral method
        correlation_length = self.config["spatial_correlation_length"]
        gaussian_field = self._generate_correlated_gaussian_field(
            X, Y, correlation_length
        )

        return X, Y, gaussian_field

    def _generate_correlated_gaussian_field(self, X, Y, correlation_length):
        """Generate spatially correlated Gaussian random field using spectral method."""
        # Get grid dimensions
        ny, nx = X.shape

        # Create frequency grids
        dx = X[0, 1] - X[0, 0] if nx > 1 else 1.0
        dy = Y[1, 0] - Y[0, 0] if ny > 1 else 1.0
        kx = np.fft.fftfreq(nx, d=dx)
        ky = np.fft.fftfreq(ny, d=dy)
        KX, KY = np.meshgrid(kx, ky)

        # Compute power spectral density for exponential correlation
        k_squared = KX**2 + KY**2
        # Avoid division by zero at k=0
        psd = np.where(
            k_squared > 0,
            (2 * np.pi * correlation_length**2)
            / (1 + (2 * np.pi * correlation_length) ** 2 * k_squared),
            2 * np.pi * correlation_length**2,  # Limit as k->0
        )

        # Generate complex white noise in frequency domain
        white_noise_real = self.robot_rng.normal(0, 1, (ny, nx))
        white_noise_imag = self.robot_rng.normal(0, 1, (ny, nx))
        white_noise_fft = white_noise_real + 1j * white_noise_imag

        # Scale by square root of PSD to get correlated field
        correlated_fft = white_noise_fft * np.sqrt(psd)

        # Transform back to spatial domain and take real part
        correlated_field = np.real(np.fft.ifft2(correlated_fft))

        # Normalize to have unit variance
        std_dev = np.std(correlated_field)
        if std_dev > 1e-10:  # Avoid division by zero
            correlated_field = correlated_field / std_dev

        return correlated_field
