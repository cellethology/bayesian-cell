"""
FilterPy-based Extended Kalman Filter implementation.
Based on the official FilterPy examples from the GitHub notebooks.
"""

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from base_filter import BaseFilter


class FilterPyExtendedKalmanFilter(BaseFilter):
    """Extended Kalman Filter using FilterPy library."""

    def __init__(self, config):
        """
        Initialize EKF using FilterPy with configuration parameters.

        Args:
            config: Dictionary containing EKF configuration parameters
        """
        super().__init__(config)

        # Initialize FilterPy EKF - correct way
        self.filterpy_ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1)

        # Set initial state and covariance
        self.filterpy_ekf.x = self.mu.copy()
        self.filterpy_ekf.P = self.Sigma.copy()

        # Set initial measurement noise
        self.filterpy_ekf.R = np.array([[self.sigma_z**2]])

        # Process model is identity for stationary target
        self.filterpy_ekf.F = np.eye(2)

        # We'll set Q dynamically in predict_and_update
        self.filterpy_ekf.Q = np.eye(2) * (self.sigma_Q**2)

    def _hx(self, x):
        """
        Measurement function for FilterPy EKF.

        Args:
            x: State vector [x, y]

        Returns:
            z: Expected measurement (scalar in array)
        """
        signal = self._h(x, self._current_robot_pos)
        return np.array([signal])

    def _HJacobian(self, x):
        """
        Jacobian of measurement function for FilterPy EKF.

        Args:
            x: State vector [x, y]

        Returns:
            H: Jacobian matrix (1x2)
        """
        # Get current robot position from stored value
        robot_pos = getattr(self, "_current_robot_pos", np.array([0.0, 0.0]))

        d = np.linalg.norm(x - robot_pos) + 1e-9  # avoid division by zero
        h_val = self._h(x, robot_pos)
        jacobian = (-self.lam * h_val * (x - robot_pos) / d).reshape(1, 2)
        return jacobian

    def predict_and_update(self, measurement, robot_pos):
        """
        Perform one complete EKF prediction and update step using FilterPy.

        Args:
            measurement: Observed signal measurement (scalar)
            robot_pos: Current robot position (2D numpy array)

        Returns:
            tuple: (updated_mu, updated_Sigma, current_sigma_Q)
        """
        # Store robot position for use in measurement functions
        self._current_robot_pos = robot_pos.copy()

        # 1. Determine process noise (adaptive or fixed)
        sigma_Q_current = self._determine_process_noise(measurement)
        Q = (sigma_Q_current**2) * np.eye(2)

        # Store for visualization
        self.sigma_history.append(sigma_Q_current)
        self.R_est_history.append(self.R_est)

        # 2. Update FilterPy filter parameters
        self.filterpy_ekf.Q = Q

        # Set measurement noise (add small numerical stability term)
        R_current = self.R_est if self.adaptive_measurement_noise else self.sigma_z**2
        self.filterpy_ekf.R = np.array([[R_current + 1e-12]])

        # 3. Prediction step (FilterPy handles this)
        self.filterpy_ekf.predict()

        # 4. Update step - CORRECT FilterPy usage
        z = np.array([measurement])
        self.filterpy_ekf.update(z, self._HJacobian, self._hx)

        # 5. Extract results
        self.mu = self.filterpy_ekf.x.copy()
        self.Sigma = self.filterpy_ekf.P.copy()

        # 6. Adaptive measurement noise update
        innovation = measurement - self._h(self.mu, robot_pos)
        self._update_adaptive_measurement_noise(innovation)

        return self.mu.copy(), self.Sigma.copy(), sigma_Q_current

    def reset(self, initial_mean=None, initial_covariance=None):
        """Reset filter to initial state."""
        # Call parent reset
        super().reset(initial_mean, initial_covariance)

        # Update FilterPy filter
        self.filterpy_ekf.x = self.mu.copy()
        self.filterpy_ekf.P = self.Sigma.copy()
        self.filterpy_ekf.R = np.array([[self.sigma_z**2]])
        self.filterpy_ekf.Q = np.eye(2) * (self.sigma_Q**2)

    def get_filterpy_filter(self):
        """Get the underlying FilterPy filter for advanced usage."""
        return self.filterpy_ekf

    def get_innovation_stats(self):
        """Get innovation statistics from FilterPy filter."""
        if hasattr(self.filterpy_ekf, "y") and hasattr(self.filterpy_ekf, "S"):
            return {
                "innovation": (
                    self.filterpy_ekf.y.copy()
                    if self.filterpy_ekf.y is not None
                    else None
                ),
                "innovation_covariance": (
                    self.filterpy_ekf.S.copy()
                    if self.filterpy_ekf.S is not None
                    else None
                ),
                "log_likelihood": (
                    self.filterpy_ekf.log_likelihood
                    if hasattr(self.filterpy_ekf, "log_likelihood")
                    else None
                ),
            }
        return None
