import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
from scipy.signal import convolve2d, fftconvolve
from scipy.stats import poisson
from collections import deque
import numba
from functools import lru_cache


class BayesianNavigation:
    def __init__(self, config=None):
        self.config = {
            "grid_size": 100,
            "initial_belief": None,
            "true_target_pos": None,
            "process_sigma": 0.5,  # Actual noise in robot motion
            "min_motion_sigma": 0.001,  # Minimum uncertainty in motion model (D_base)
            "adaptive_rate": 8,  # How quickly uncertainty decreases with signal (k)
            "signal_max": 1,
            "signal_decay": 0.3,
            "step_size": 0.1,
            "kernel_size": 5,
            "target_reach_threshold": 5.0,
            "innovation_window_size": 20,  # Size of window for innovation statistics
            "adaptation_rate": 0.4,  # Rate of adaptation for measurement and process noise
            "measurement_sigma_estimate": 0.5,  # Initial estimate of measurement noise
            "process_sigma_estimate": 0.1,  # Initial estimate of process noise
            "min_allowed_variance": 1e-6,  # Minimum allowed variance to prevent numerical issues
            "adaptive_filtering": False,  # Enable adaptive filtering
            "adaptive_process_variance": "error_based",  # "error_based" or "exponential"
            "noise_model": "poisson",
            "noise_std": 1,
        }
        if config is not None:
            self.config.update(config)

        # if self.config["noise_model"] == "poisson":
        #     if self.config["noise_std"] > 0:
        #         print("Poisson noise model selected, thus noise_std is ignored")

        self.grid_size = self.config["grid_size"]
        if self.config["initial_belief"] is None:
            self.belief = np.ones((self.grid_size, self.grid_size)) / (
                self.grid_size * self.grid_size
            )
        else:
            self.belief = self.config["initial_belief"]

        if self.config["true_target_pos"] is None:
            self.true_target_pos = (self.grid_size * 4 // 5, self.grid_size * 4 // 5)
            self.config["true_target_pos"] = self.true_target_pos
        else:
            self.true_target_pos = self.config["true_target_pos"]

        # Precompute kernel for motion update
        kernel_size = self.config["kernel_size"]
        center = kernel_size // 2
        y, x = np.ogrid[:kernel_size, :kernel_size]
        self.kernel_mat = np.sqrt((x - center) ** 2 + (y - center) ** 2)

        # Precompute grid coordinates for performance
        self._setup_grid_cache()

        # Cache for motion kernels
        self._kernel_cache = {}
        self._cache_hit_count = 0
        self._cache_miss_count = 0

        # Set adaptive process variance flag
        self.adaptive_process_variance = self.config["adaptive_process_variance"]

        # Adaptive Kalman Filter parameters
        self.adaptive_filtering = self.config["adaptive_filtering"]
        self.measurement_variance = self.config["measurement_sigma_estimate"] ** 2
        self.process_variance = self.config["process_sigma_estimate"] ** 2

        # Use deque for better performance instead of lists
        window_size = self.config["innovation_window_size"]
        self.innovation_history = deque(maxlen=window_size)
        self.predicted_variance_history = deque(maxlen=window_size)

        # For error-based adaptive process variance
        self.previous_intended_pos = None

        # Pre-generate random numbers for batch processing
        self._random_buffer = None
        self._random_buffer_idx = 0
        self._buffer_size = 1000

    def _setup_grid_cache(self):
        """Precompute grid coordinates and distance matrices for performance"""
        # Create coordinate grids once
        x = np.arange(self.grid_size, dtype=np.float32)
        y = np.arange(self.grid_size, dtype=np.float32)
        self.xx, self.yy = np.meshgrid(x, y, indexing="ij")

        # Precompute grid indices for belief calculations
        self.grid_indices = np.indices(self.belief.shape, dtype=np.float32)

        # Cache for signal grids (will be populated as needed)
        self._signal_cache = {}

    @lru_cache(maxsize=128)
    def _get_distance_grid(self, target_pos):
        """Cached computation of distance grid for a given target position"""
        target_x, target_y = target_pos
        distance = np.sqrt((self.xx - target_x) ** 2 + (self.yy - target_y) ** 2)
        return np.maximum(distance, 1e-8)  # Avoid division by zero

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _compute_signal_fast(distance_grid, signal_max, decay_exp):
        """JIT-compiled fast signal computation"""
        return signal_max * np.exp(-decay_exp * distance_grid)

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _compute_likelihood_gaussian(measurement, expected_signal, variance):
        """JIT-compiled Gaussian likelihood computation"""
        diff_sq = (measurement - expected_signal) ** 2
        return np.exp(-diff_sq / (2 * variance)) / np.sqrt(2 * np.pi * variance)

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _compute_variance_stats(grid_indices, belief):
        """JIT-compiled variance computation"""
        μx = np.sum(grid_indices[0] * belief)
        μy = np.sum(grid_indices[1] * belief)
        var_x = np.sum(((grid_indices[0] - μx) ** 2) * belief)
        var_y = np.sum(((grid_indices[1] - μy) ** 2) * belief)
        return var_x, var_y

    def get_optimal_motion_kernel(self, theta_0, adaptive_sigma):
        """
        Optimized version with kernel caching and stable matrix operations.
        Returns a 2D Gaussian kernel based on the filter's belief about motion uncertainty.
        """
        # Create cache key (round to avoid floating point precision issues)
        cache_key = (round(theta_0, 6), round(adaptive_sigma, 6))

        if cache_key in self._kernel_cache:
            self._cache_hit_count += 1
            return self._kernel_cache[cache_key]

        self._cache_miss_count += 1

        s = self.config["step_size"]
        sigma_theta = adaptive_sigma
        sigma_d = 0.1 * s

        cos_t = np.cos(theta_0)
        sin_t = np.sin(theta_0)

        # Mean displacement
        dx = s * cos_t
        dy = s * sin_t
        mu = np.array([dx, dy], dtype=np.float32)

        # Covariance matrix
        var_x = (s**2) * (sin_t**2) * (sigma_theta**2) + (cos_t**2) * (sigma_d**2)
        var_y = (s**2) * (cos_t**2) * (sigma_theta**2) + (sin_t**2) * (sigma_d**2)
        cov_xy = sin_t * cos_t * ((sigma_d**2) - (s**2) * (sigma_theta**2))
        cov = np.array([[var_x, cov_xy], [cov_xy, var_y]], dtype=np.float32)

        # Generate grid
        size = self.config["kernel_size"]
        half = size // 2
        y, x = np.mgrid[-half : half + 1, -half : half + 1]
        pos = np.stack([x, y], axis=-1, dtype=np.float32)  # shape (size, size, 2)

        # Evaluate multivariate Gaussian using Cholesky decomposition for stability
        diff = pos - mu

        # Add regularization and use Cholesky decomposition
        reg_cov = cov + np.eye(2, dtype=np.float32) * 1e-6
        try:
            L = np.linalg.cholesky(reg_cov)
            # Solve L * y = diff for each point
            y_vals = np.linalg.solve(L, diff.reshape(-1, 2).T).T
            exponent = np.sum(y_vals**2, axis=1).reshape(size, size)
        except np.linalg.LinAlgError:
            # Fallback to standard method if Cholesky fails
            inv_cov = np.linalg.solve(reg_cov, np.eye(2))
            exponent = np.einsum("...i,ij,...j->...", diff, inv_cov, diff)

        kernel = np.exp(-0.5 * exponent)
        kernel /= kernel.sum()

        # Cache the result if cache is not too large
        if len(self._kernel_cache) < 256:
            self._kernel_cache[cache_key] = kernel

        return kernel

    def get_expected_signal(self, xx, yy, target_pos):
        distance = np.sqrt((xx - target_pos[0]) ** 2 + (yy - target_pos[1]) ** 2)
        distance = np.maximum(distance, 1e-8)  # Avoid division by zero
        return self.config["signal_max"] * np.exp(
            -self.config["signal_decay"] * distance
        )

    def compute_all_expected_signal(self, target_pos):
        """Optimized version using cached grids and JIT-compiled signal computation"""
        # Use cached distance grid and JIT-compiled computation
        distance_grid = self._get_distance_grid(target_pos)
        signal = self._compute_signal_fast(
            distance_grid,
            self.config["signal_max"],
            self.config["signal_decay"],
        )
        return signal

    def get_noisy_measurement(self, pos, target_pos):
        expected_signal = self.get_expected_signal(pos[0], pos[1], target_pos)

        if self.config["noise_model"] == "poisson":
            signal = np.random.poisson(expected_signal)
        elif self.config["noise_model"] == "gaussian":
            signal = np.random.normal(expected_signal, self.config["noise_std"])
        else:
            raise ValueError(f"Invalid noise model: {self.config['noise_model']}")
        return signal

    def _diag_state_cov(self):
        """Var(x), Var(y) of the posterior belief – JIT-optimized version."""
        # Use JIT-compiled variance computation
        return self._compute_variance_stats(self.grid_indices, self.belief)

    def _get_batch_random(self, count=2):
        """Get random numbers from pre-generated batch for better performance"""
        if self._random_buffer is None or self._random_buffer_idx + count > len(
            self._random_buffer
        ):
            # Regenerate buffer
            self._random_buffer = np.random.normal(0, 1, self._buffer_size)
            self._random_buffer_idx = 0

        result = self._random_buffer[
            self._random_buffer_idx : self._random_buffer_idx + count
        ]
        self._random_buffer_idx += count
        return result

    def get_signal_gradient(self, robot_pos, beacon_pos):
        """
        Computes the partial derivatives of the expected signal
        with respect to the beacon (target) position.

        ∂signal/∂x_target and ∂signal/∂y_target,
        based on exponential decay.

        Returns:
            df_dx, df_dy: partial derivatives of signal w.r.t. beacon x and y
        """
        x_r, y_r = robot_pos
        x_t, y_t = beacon_pos

        dx = x_r - x_t
        dy = y_r - y_t
        distance = np.sqrt(dx**2 + dy**2)

        A = self.config["signal_max"]
        k = self.config["signal_decay"]

        # Ensure distance is never zero
        if distance < 1e-8:
            return 0.0, 0.0  # No gradient when at target

        decay = np.exp(-k * distance)
        df_dr = -A * k * decay
        df_dx = df_dr * (dx / distance)
        df_dy = df_dr * (dy / distance)

        return df_dx, df_dy

    def update_belief(self, measurement, robot_pos):
        """
        Update the belief using proper adaptive Kalman filtering.
        The measurement variance is dynamically adjusted based on innovation statistics.
        """
        # Calculate expected measurement at each grid position
        expected_signal_grid = self.compute_all_expected_signal(robot_pos)

        # Get estimated target position from belief
        estimated_target_pos = np.unravel_index(
            np.argmax(self.belief), self.belief.shape
        )

        # Calculate expected measurement based on this estimate
        expected_measurement = self.get_expected_signal(
            robot_pos[0], robot_pos[1], estimated_target_pos
        )
        innovation = measurement - expected_measurement

        # Compute partial derivatives of signal with respect to beacon position
        df_dx, df_dy = self.get_signal_gradient(robot_pos, estimated_target_pos)

        # Get variance in belief
        var_x, var_y = self._diag_state_cov()

        # Approximate variance in predicted signal
        signal_variance = (df_dx**2) * var_x + (df_dy**2) * var_y
        predicted_variance = signal_variance + self.measurement_variance

        # Store innovation and predicted variance for adaptation (deque handles size automatically)
        self.innovation_history.append(innovation)
        self.predicted_variance_history.append(predicted_variance)

        # Update measurement and process variance based on innovation statistics
        if (
            len(self.innovation_history) >= 3 and self.adaptive_filtering
        ):  # Need enough samples for statistics
            self.adapt_noise_parameters()
            # pass

        # Now perform the Bayesian measurement update with adaptive measurement variance
        likelihood = self.get_likelihood(measurement, expected_signal_grid)

        # Update belief
        self.belief *= likelihood
        self.belief /= np.sum(self.belief)  # Normalize

        return innovation, self.measurement_variance

    def get_likelihood(self, measurement, expected_signal_grid):
        if self.config["noise_model"] == "poisson":
            likelihood = poisson.pmf(measurement, expected_signal_grid)
            likelihood = np.clip(likelihood, 1e-12, None)
        elif self.config["noise_model"] == "gaussian":
            # Use JIT-compiled Gaussian likelihood computation
            likelihood = self._compute_likelihood_gaussian(
                measurement, expected_signal_grid, self.measurement_variance
            )
            likelihood = np.clip(likelihood, 1e-12, None)
        else:
            raise ValueError(f"Invalid noise model: {self.config['noise_model']}")
        return likelihood

    def adapt_noise_parameters(self):
        """
        Adapt measurement and process noise parameters based on innovation statistics.
        This is the key component of the adaptive Kalman filter.
        """
        emp_variance = np.var(self.innovation_history, ddof=1)
        avg_pred_var = np.mean(self.predicted_variance_history)
        signal_variance = max(avg_pred_var - self.measurement_variance, 0)

        if signal_variance < emp_variance:
            estimated_R = emp_variance - signal_variance
        else:
            estimated_R = emp_variance  # IMPORTANT: respects the kalman filter assumption that innovation variance is at least the measurement variance, and that the innovation variance is also at least the signal variance
        adaptation_rate = self.config["adaptation_rate"]

        # Exponential smoothing
        self.measurement_variance = (
            1 - adaptation_rate
        ) * self.measurement_variance + adaptation_rate * estimated_R

        # Canonical process noise adaptation via innovation statistics
        # if self.config.get("adaptive_process_variance") == "statistical":
        #     estimated_Q = max(
        #         emp_variance - self.measurement_variance,
        #         self.config["min_allowed_variance"],
        #     )
        #     self.process_variance = (
        #         1 - adaptation_rate
        #     ) * self.process_variance + adaptation_rate * estimated_Q

    def get_adaptive_motion_sigma(self, signal_strength):
        """
        Calculate motion model uncertainty based on signal strength
        D(s) = D_base + (D_max - D_base) * exp(-k * s)
        """
        D_base = self.config["min_motion_sigma"]
        D_max = self.config["process_sigma_estimate"]
        k = self.config["adaptive_rate"]

        return D_base + (D_max - D_base) * np.exp(-k * signal_strength)

    def motion_update(self, current_signal, current_pos=None):
        if self.adaptive_process_variance == "exponential":
            adaptive_sigma = self.get_adaptive_motion_sigma(current_signal)
        elif (
            self.adaptive_process_variance == "error_based" and current_pos is not None
        ):
            # Store intended position for next iteration (before motion happens)
            action = self.get_next_intended_action(current_pos)
            theta_0 = np.arctan2(action[1], action[0])
            intended_dx = self.config["step_size"] * np.cos(theta_0)
            intended_dy = self.config["step_size"] * np.sin(theta_0)
            self.previous_intended_pos = (
                current_pos[0] + intended_dx,
                current_pos[1] + intended_dy,
            )

            adaptive_sigma = np.sqrt(self.process_variance)
        else:  # Default case for "none" or other invalid values
            adaptive_sigma = self.config["process_sigma_estimate"]

        kernel = np.exp(-(self.kernel_mat**2) / (2 * adaptive_sigma**2))
        kernel /= kernel.sum()

        # Use FFT-based convolution for better performance with larger grids
        if self.grid_size > 50:
            self.belief = fftconvolve(self.belief, kernel, mode="same")
        else:
            self.belief = convolve2d(
                self.belief, kernel, mode="same", boundary="fill", fillvalue=0
            )
        self.belief /= np.sum(self.belief)

        return adaptive_sigma

    def get_next_intended_action(self, robot_pos):
        """get unnormalized step vector"""
        estimated_target_pos = np.unravel_index(
            np.argmax(self.belief), self.belief.shape
        )

        dx = estimated_target_pos[0] - robot_pos[0]
        dy = estimated_target_pos[1] - robot_pos[1]
        return dx, dy

    def update_position(self, true_pos, action):
        """
        Update the true state (position) of the robot based on the intended action.
        Uses an isotropic Gaussian motion model with standard deviation process_sigma.
        """
        s = self.config["step_size"]
        sigma = self.config["process_sigma"]

        # Normalize action direction
        norm = np.linalg.norm(action)
        direction = np.array(action) / norm

        # Intended step
        dx = s * direction[0]
        dy = s * direction[1]

        # Add isotropic Gaussian noise using batch random generation
        random_vals = self._get_batch_random(2) * sigma
        noisy_dx = dx + random_vals[0]
        noisy_dy = dy + random_vals[1]

        # Compute new position
        new_x = np.clip(true_pos[0] + noisy_dx, 0, self.grid_size - 1)
        new_y = np.clip(true_pos[1] + noisy_dy, 0, self.grid_size - 1)

        return (new_x, new_y)

    def update_process_variance_from_motion_error(self, actual_pos_after_motion):
        """
        Update process variance based on observed motion error.
        Compares actual position after motion with intended position to estimate motion noise.
        """
        if self.previous_intended_pos is None:
            return  # No previous intended position to compare with

        # Calculate error between intended and actual position
        dx = actual_pos_after_motion[0] - self.previous_intended_pos[0]
        dy = actual_pos_after_motion[1] - self.previous_intended_pos[1]
        observed_error_sq = dx**2 + dy**2

        adaptation_rate = self.config["adaptation_rate"]
        self.process_variance = (
            1 - adaptation_rate
        ) * self.process_variance + adaptation_rate * max(
            observed_error_sq, self.config["min_allowed_variance"]
        )


def run_navigation_simulation(config=None, steps=100, verbose=False):
    env = BayesianNavigation(config)

    robot_pos = (env.grid_size // 5, env.grid_size // 5)
    trajectory = [robot_pos]
    sigmas = [env.config["process_sigma_estimate"]]
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
    np.random.seed(5)
    example_config = {
        "grid_size": 100,
        "process_sigma": 0.5,
        "process_sigma_estimate": 0.5,  # process_sigma * step_size
        "adaptive_rate": 0.8,  # Irrelevant when min == max
        "signal_max": 2,
        "signal_decay": 0.3,
        "step_size": 1,
        "kernel_size": 5,
        "adaptive_filtering": False,
        "adaptive_process_variance": "none",
        "noise_model": "gaussian",
        "noise_std": 0.1,
        "measurement_sigma_estimate": 0.1,
    }
    trajectory, env, sigmas, innovations, measurement_variances = (
        run_navigation_simulation(config=example_config, steps=100000, verbose=True)
    )

    trajectory = np.array(trajectory)
    sigmas = np.array(sigmas)
    innovations = np.array(innovations)
    measurement_variances = np.array(measurement_variances)

    # Create a more comprehensive visualization
    fig = plt.figure(figsize=(15, 10))

    # 2x4 subplot grid
    # Signal map and trajectory in log scale
    ax1 = fig.add_subplot(241)  # Signal map
    signal_map = env.compute_all_expected_signal(env.true_target_pos)
    avg_signal_strength = np.mean(signal_map)
    print(f"Average signal strength over entire space: {avg_signal_strength:.4f}")
    im = ax1.imshow(signal_map, cmap="Greens", interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label("Signal Strength")
    ax1.plot(trajectory[:, 1], trajectory[:, 0], "b-", label="Robot Path")
    ax1.plot(
        env.true_target_pos[1],
        env.true_target_pos[0],
        "g*",
        markersize=15,
        label="Target",
    )
    ax1.plot(trajectory[0, 1], trajectory[0, 0], "bs", label="Start")
    ax1.legend()

    # Final belief state
    ax2 = fig.add_subplot(242)  # Belief state
    ax2.imshow(env.belief, cmap="viridis", interpolation="nearest")
    ax2.plot(
        env.true_target_pos[1],
        env.true_target_pos[0],
        "r*",
        markersize=15,
        label="True Target",
    )
    estimated_pos = np.unravel_index(np.argmax(env.belief), env.belief.shape)
    ax2.plot(
        estimated_pos[1],
        estimated_pos[0],
        "bx",
        markersize=15,
        label="Estimated Target",
    )
    ax2.legend()

    # Innovation histogram
    ax3 = fig.add_subplot(243)  # Innovation histogram
    ax3.hist(
        innovations,
        bins=50,
        density=True,
        alpha=0.7,
        color="skyblue",
        label="Innovation Histogram",
    )
    ax3.axvline(
        np.median(innovations),
        color="red",
        linestyle="--",
        label=f"Median = {np.median(innovations):.4f}",
    )
    ax3.axvline(
        np.std(innovations),
        color="green",
        linestyle="--",
        label=f"Std = {np.std(innovations):.4f}",
    )
    ax3.set_xlabel("Innovation")
    ax3.set_ylabel("Density")
    ax3.legend()

    # Motion sigma evolution
    ax4 = fig.add_subplot(244)  # Motion sigma
    ax4.plot(sigmas, "r-", label="Motion Sigma")
    ax4.set_xlabel("Time Step")
    ax4.set_ylabel("Sigma")
    ax4.legend()

    # Measurement variance plot
    ax5 = fig.add_subplot(245)  # Measurement variance
    ax5.plot(measurement_variances, "g-", label="Measurement Variance")
    ax5.plot(
        np.ones_like(measurement_variances) * env.config["noise_std"] ** 2,
        "r--",
        label="True Noise Variance",
    )
    ax5.set_yscale("log")
    ax5.set_xlabel("Time Step")
    ax5.set_ylabel("Measurement Variance")
    ax5.legend()

    # Innovation line plot
    ax6 = fig.add_subplot(246)  # Innovation over time
    ax6.plot(innovations, "b-", alpha=0.7, label="Innovation")
    ax6.set_xlabel("Time Step")
    ax6.set_ylabel("Innovation")
    ax6.legend()

    # Motion sigma histogram (new subplot)
    ax7 = fig.add_subplot(247)  # Motion sigma histogram
    ax7.hist(
        sigmas,
        bins=50,
        density=True,
        color="salmon",
        alpha=0.7,
        label="Motion Sigma Histogram",
    )
    ax7.axvline(
        np.median(sigmas),
        color="blue",
        linestyle="--",
        label=f"Median = {np.median(sigmas):.4f}",
    )
    ax7.set_xlabel("Motion Sigma")
    ax7.set_ylabel("Density")
    ax7.set_title("Histogram of Motion Sigma")
    ax7.legend()

    plt.tight_layout()
    plt.show()
