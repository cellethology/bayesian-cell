"""
Extended Kalman Filter implementation for target tracking.
Supports adaptive process noise based on signal strength.
"""

import numpy as np

eps = 0.001


class ExtendedKalmanFilter:
    """Extended Kalman Filter for target tracking with signal-based observations."""

    def __init__(self, config):
        """
        Initialize EKF with configuration parameters.

        Args:
            config: Dictionary containing EKF configuration parameters
        """
        self.config = config

        # Extract parameters
        self.c0 = config.get("signal_max", 40.0)
        self.lam = config.get("signal_decay", 0.03)
        self.sigma_Q = config.get("baseline_process_noise", 0.5)
        self.sigma_u = config.get("actuator_noise", 0.2)
        self.is_adaptive = config.get("adaptive_process_noise", False)
        self.alpha_R = config.get("alpha_R", 0.01)
        self.adaptive_measurement_noise = config.get(
            "adaptive_measurement_noise", False
        )

        # Initialize state
        self.mu = np.array(config.get("initial_belief_mean", [100.0, 100.0]))
        self.Sigma = np.eye(2) * config.get("initial_belief_variance", 100.0)

        # Set measurement noise based on initial geometry
        initial_robot_pos = np.array(config.get("robot_start_pos", [80.0, 80.0]))
        self.sigma_z = np.sqrt(self._h(self.mu, initial_robot_pos))
        self.R_est = self.sigma_z**2

        # Store sigma evolution for visualization
        self.sigma_history = []
        self.R_est_history = []

    def _h(self, mu, r):
        """Expected signal at robot pose r for target mean mu."""
        d = np.linalg.norm(mu - r)
        return self.c0 * np.exp(-self.lam * d)

    def _jacobian_h(self, mu, r):
        """Jacobian of measurement function h with respect to target position."""
        d = np.linalg.norm(mu - r) + 1e-9  # avoid division by zero
        h_val = self._h(mu, r)
        return (-self.lam * h_val * (mu - r) / d).reshape(1, 2)  # 1×2 matrix

    def predict_and_update(self, measurement, robot_pos):
        """
        Perform one complete EKF prediction and update step.

        Args:
            measurement: Observed signal measurement (scalar)
            robot_pos: Current robot position (2D numpy array)

        Returns:
            tuple: (updated_mu, updated_Sigma, current_sigma_Q)
        """
        # 1. Determine process noise (adaptive or fixed)
        if self.is_adaptive:
            sigma_Q_current = self.sigma_Q / (eps + measurement)
        else:
            sigma_Q_current = self.sigma_Q

        Q = (sigma_Q_current**2) * np.eye(2)
        self.sigma_history.append(sigma_Q_current)
        self.R_est_history.append(self.R_est)

        # 2. Prediction step
        mu_pred = self.mu  # No motion model for target
        Sigma_pred = self.Sigma + Q

        # 3. Linearize measurement function
        h_pred = self._h(mu_pred, robot_pos)
        H = self._jacobian_h(mu_pred, robot_pos)
        R = self.R_est if self.adaptive_measurement_noise else self.sigma_z**2

        # 4. Innovation and Kalman gain
        innovation = measurement - h_pred
        S = (H @ Sigma_pred @ H.T + R).item() + 1e-12  # scalar innovation covariance with numerical stability
        K = (Sigma_pred @ H.T / S).reshape(
            2,
        )  # Kalman gain vector

        # 5. Update step
        self.mu = mu_pred + K * innovation
        self.Sigma = (np.eye(2) - np.outer(K, H)) @ Sigma_pred

        # Numerical stability: ensure covariance symmetry
        self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)

        # 6. Adaptive measurement noise updates (innovation-based)
        if self.adaptive_measurement_noise:
            self.R_est = (1 - self.alpha_R) * self.R_est + self.alpha_R * (
                innovation**2
            )

        return self.mu.copy(), self.Sigma.copy(), sigma_Q_current

    def get_belief_state(self):
        """Get current belief state (mean and covariance)."""
        return self.mu.copy(), self.Sigma.copy()

    def get_sigma_history(self):
        """Get history of sigma_Q values for visualization."""
        return np.array(self.sigma_history)

    def get_R_est_history(self):
        """Get history of R_est values for visualization."""
        return np.array(self.R_est_history)

    def reset(self, initial_mean=None, initial_covariance=None):
        """Reset filter to initial state."""
        if initial_mean is not None:
            self.mu = np.array(initial_mean)
        else:
            self.mu = np.array(self.config.get("initial_belief_mean", [100.0, 100.0]))

        if initial_covariance is not None:
            self.Sigma = initial_covariance
        else:
            self.Sigma = np.eye(2) * self.config.get("initial_belief_variance", 100.0)

        self.sigma_history = []
        self.R_est_history = []

        # Reset adaptive measurement noise estimate
        if hasattr(self, "config"):
            initial_robot_pos = np.array(
                self.config.get("robot_start_pos", [80.0, 80.0])
            )
            self.R_est = self._h(self.mu, initial_robot_pos)
