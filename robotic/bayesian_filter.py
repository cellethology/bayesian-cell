"""
Bayesian filter for navigation system.
Handles belief updates, adaptive filtering, and innovation tracking.
"""

import numpy as np
from scipy.signal import convolve2d, fftconvolve
from collections import deque
from utils import compute_variance_stats


class BayesianFilter:
    """Manages belief state updates and adaptive Kalman filtering."""

    def __init__(self, config, grid_cache, signal_model, motion_model):
        self.config = config
        self.grid_cache = grid_cache
        self.signal_model = signal_model
        self.motion_model = motion_model

        # Initialize belief state
        grid_size = config["grid_size"]
        if config.get("initial_belief") is None:
            self.belief = np.ones((grid_size, grid_size)) / (grid_size * grid_size)
        else:
            self.belief = config["initial_belief"]

        # Adaptive Kalman Filter parameters
        self.adaptive_filtering = config["adaptive_filtering"]
        self.measurement_variance = config["initial_measurement_sigma"] ** 2

        # Use deque for better performance instead of lists
        window_size = config["innovation_window_size"]
        self.innovation_history = deque(maxlen=window_size)
        self.predicted_variance_history = deque(maxlen=window_size)

    def update_belief(self, measurement, robot_pos):
        """
        Update the belief using adaptive Kalman filtering.
        The measurement variance is dynamically adjusted based on innovation statistics.
        """
        # Calculate expected measurement at each grid position
        expected_signal_grid = self.signal_model.compute_all_expected_signal(robot_pos)

        # Get estimated target position from belief
        estimated_target_pos = np.unravel_index(
            np.argmax(self.belief), self.belief.shape
        )

        # Calculate expected measurement based on this estimate
        expected_measurement = self.signal_model.get_expected_signal(
            robot_pos[0], robot_pos[1], estimated_target_pos
        )
        innovation = measurement - expected_measurement

        # Compute partial derivatives of signal with respect to beacon position
        df_dx, df_dy = self.signal_model.get_signal_gradient(
            robot_pos, estimated_target_pos
        )

        # Get variance in belief
        var_x, var_y = self._diag_state_cov()

        # Approximate variance in predicted signal
        signal_variance = (df_dx**2) * var_x + (df_dy**2) * var_y
        predicted_variance = signal_variance + self.measurement_variance

        # Store innovation and predicted variance for adaptation (deque handles size automatically)
        self.innovation_history.append(innovation)
        self.predicted_variance_history.append(predicted_variance)

        # Update measurement and process variance based on innovation statistics
        if len(self.innovation_history) >= 3 and self.adaptive_filtering:
            self.adapt_noise_parameters()

        # Perform the Bayesian measurement update with adaptive measurement variance
        likelihood = self.signal_model.get_likelihood(
            measurement, expected_signal_grid, self.measurement_variance
        )

        # Update belief
        self.belief *= likelihood
        self.belief /= np.sum(self.belief)  # Normalize

        return innovation, self.measurement_variance

    def motion_update(self, current_signal, current_pos, adaptive_process_variance):
        """Apply motion update to belief state."""
        if adaptive_process_variance == "exponential":
            adaptive_sigma = self.motion_model.get_adaptive_motion_sigma(current_signal)
        elif adaptive_process_variance == "error_based" and current_pos is not None:
            # Store intended position for next iteration (before motion happens)
            action = self.motion_model.get_next_intended_action(
                current_pos, self.belief
            )
            self.motion_model.store_intended_position(current_pos, action)
            adaptive_sigma = np.sqrt(self.motion_model.process_variance)
        else:  # Default case for "none" or other invalid values
            adaptive_sigma = self.config["initial_process_sigma"]

        kernel = self.motion_model.get_simple_motion_kernel(adaptive_sigma)

        # Use same convolution method as old implementation
        self.belief = convolve2d(
            self.belief, kernel, mode="same", boundary="fill", fillvalue=0
        )
        self.belief /= np.sum(self.belief)

        return adaptive_sigma

    def _diag_state_cov(self):
        """Var(x), Var(y) of the posterior belief – JIT-optimized version."""
        return compute_variance_stats(self.grid_cache.grid_indices, self.belief)

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
            estimated_R = emp_variance

        adaptation_rate = self.config["adaptation_rate"]

        # Exponential smoothing
        self.measurement_variance = (
            1 - adaptation_rate
        ) * self.measurement_variance + adaptation_rate * estimated_R
