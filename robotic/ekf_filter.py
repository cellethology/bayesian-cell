"""
Extended Kalman Filter implementation for target tracking.
Supports adaptive process noise based on signal strength.
"""

import numpy as np
from base_filter import BaseFilter

eps = 1


class ExtendedKalmanFilter(BaseFilter):
    """Extended Kalman Filter for target tracking with signal-based observations."""

    def __init__(self, config):
        """
        Initialize EKF with configuration parameters.

        Args:
            config: Dictionary containing EKF configuration parameters
        """
        super().__init__(config)

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
        sigma_Q_current = self._determine_process_noise(measurement)
        Q = (sigma_Q_current**2) * np.eye(2)
        
        # Store for visualization
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
        S = (
            H @ Sigma_pred @ H.T + R
        ).item() + 1e-12  # scalar innovation covariance with numerical stability
        K = (Sigma_pred @ H.T / S).reshape(
            2,
        )  # Kalman gain vector

        # 5. Update step
        self.mu = mu_pred + K * innovation
        self.Sigma = (np.eye(2) - np.outer(K, H)) @ Sigma_pred

        # Numerical stability: ensure covariance symmetry and positive definiteness
        self.Sigma = self._ensure_positive_definite(self.Sigma)

        # 6. Adaptive measurement noise updates (innovation-based)
        self._update_adaptive_measurement_noise(innovation)

        return self.mu.copy(), self.Sigma.copy(), sigma_Q_current

