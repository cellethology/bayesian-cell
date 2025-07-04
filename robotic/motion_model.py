"""
Motion model for Bayesian navigation system.
Handles robot movement, motion kernels, and adaptive process variance.
"""

import numpy as np
from numpy.random import normal


class MotionModel:
    """Manages robot motion and target motion uncertainty estimation.

    Note: Motion kernels represent uncertainty in TARGET dynamics, not robot motion.
    Robot motion has separate noise modeling for realistic actuator behavior.
    """

    def __init__(self, config):
        self.config = config
        self.grid_size = config["grid_size"]

        # Precompute kernel distance matrix
        kernel_size = config["kernel_size"]
        center = kernel_size // 2
        y, x = np.ogrid[:kernel_size, :kernel_size]
        self.kernel_mat = np.sqrt((x - center) ** 2 + (y - center) ** 2)

        # Removed kernel caching for simplicity

        # Adaptive process variance tracking
        self.previous_intended_pos = None

    def get_optimal_motion_kernel(self, theta_0, adaptive_sigma):
        """
        Simplified version without caching or complex optimizations.
        Returns a 2D Gaussian kernel based on motion uncertainty.
        """
        s = self.config["step_size"]
        sigma_theta = adaptive_sigma
        sigma_d = 0.1 * s

        cos_t = np.cos(theta_0)
        sin_t = np.sin(theta_0)

        # Mean displacement
        dx = s * cos_t
        dy = s * sin_t
        mu = np.array([dx, dy])

        # Covariance matrix
        var_x = (s**2) * (sin_t**2) * (sigma_theta**2) + (cos_t**2) * (sigma_d**2)
        var_y = (s**2) * (cos_t**2) * (sigma_theta**2) + (sin_t**2) * (sigma_d**2)
        cov_xy = sin_t * cos_t * ((sigma_d**2) - (s**2) * (sigma_theta**2))
        cov = np.array([[var_x, cov_xy], [cov_xy, var_y]])

        # Generate grid
        size = self.config["kernel_size"]
        half = size // 2
        y, x = np.mgrid[-half : half + 1, -half : half + 1]
        pos = np.stack([x, y], axis=-1)

        # Evaluate multivariate Gaussian - simple version
        diff = pos - mu
        inv_cov = np.linalg.inv(cov + np.eye(2) * 1e-6)  # Add small regularization
        exponent = np.einsum("...i,ij,...j->...", diff, inv_cov, diff)

        kernel = np.exp(-0.5 * exponent)
        kernel /= kernel.sum()

        return kernel

    def get_simple_motion_kernel(self, adaptive_sigma):
        """
        Simple isotropic Gaussian motion kernel for target motion uncertainty.

        This kernel represents uncertainty in target movement, not robot motion.
        Adaptive_sigma scales with signal strength: strong signals suggest
        stationary target (low uncertainty), weak signals suggest moving target.
        """
        kernel = np.exp(-(self.kernel_mat**2) / (2 * adaptive_sigma**2))
        kernel /= kernel.sum()
        return kernel

    def get_adaptive_motion_sigma(self, signal_strength):
        """
        Adapt target motion uncertainty based on signal strength.

        Supports multiple decay types:
        - power_law: D_max / (k * s + 1)^p where p is power_exponent
        """
        D_max = self.config["process_sigma_estimate"]
        decay_type = self.config.get("adaptive_decay_type", "exponential")

        if decay_type == "power_law":
            # Use separate power_exponent parameter
            power_exp = self.config.get("power_exponent", 1.0)
            return D_max / (signal_strength + 1) ** power_exp
        else:
            raise ValueError(
                f"Unknown adaptive_decay_type: {decay_type}. Must be 'exponential' or 'power_law'"
            )

    def get_next_intended_action(self, robot_pos, belief):
        """Get unnormalized step vector towards estimated target."""
        if np.any(np.isnan(belief)):
            print(
                f"NaN in belief! Shape: {belief.shape}, NaN count: {np.sum(np.isnan(belief))}"
            )
        estimated_target_pos = np.unravel_index(np.argmax(belief), belief.shape)

        dx = estimated_target_pos[0] - robot_pos[0]
        dy = estimated_target_pos[1] - robot_pos[1]
        return dx, dy

    def update_position(self, true_pos, action):
        """
        Update the true robot position with realistic actuator noise.

        This models actual robot motion uncertainty (separate from target tracking).
        """
        s = self.config["step_size"]

        # Normalize action direction
        norm = np.linalg.norm(action)
        if norm == 0:
            return true_pos

        direction = np.array(action) / norm

        # Intended step
        dx = s * direction[0]
        dy = s * direction[1]

        # Default isotropic Gaussian noise
        sigma = self.config["process_sigma"]
        noisy_dx = dx + normal(0, sigma)
        noisy_dy = dy + normal(0, sigma)

        # Compute new position
        new_x = np.clip(true_pos[0] + noisy_dx, 0, self.grid_size - 1)
        new_y = np.clip(true_pos[1] + noisy_dy, 0, self.grid_size - 1)

        return (new_x, new_y)

    def store_intended_position(self, current_pos, action):
        """Store intended position for error-based adaptation."""
        theta_0 = np.arctan2(action[1], action[0])
        intended_dx = self.config["step_size"] * np.cos(theta_0)
        intended_dy = self.config["step_size"] * np.sin(theta_0)
        self.previous_intended_pos = (
            current_pos[0] + intended_dx,
            current_pos[1] + intended_dy,
        )
