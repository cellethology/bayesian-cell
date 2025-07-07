"""
FilterPy-based Unscented Kalman Filter implementation.
Based on the official FilterPy examples from the GitHub notebooks.
"""

import numpy as np
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
from .base_filter import BaseFilter


class FilterPyUnscentedKalmanFilter(BaseFilter):
    """Unscented Kalman Filter using FilterPy library."""

    def __init__(self, config):
        """
        Initialize UKF using FilterPy with configuration parameters.

        Args:
            config: Dictionary containing UKF configuration parameters
        """
        super().__init__(config)

        # UKF-specific parameters
        self.alpha = config.get("ukf_alpha", 0.1)
        self.beta = config.get("ukf_beta", 2.0)
        self.kappa = config.get("ukf_kappa", 0.0)

        # Create sigma points generator - correct way
        self.sigma_points = MerweScaledSigmaPoints(
            n=2,  # state dimension
            alpha=self.alpha,
            beta=self.beta,
            kappa=self.kappa
        )

        # Initialize FilterPy UKF - correct way
        self.filterpy_ukf = UnscentedKalmanFilter(
            dim_x=2,  # state dimension [x, y]
            dim_z=1,  # measurement dimension (scalar signal)
            dt=1.0,   # time step (not used for stationary target)
            hx=self._hx,  # measurement function
            fx=self._fx,  # state transition function
            points=self.sigma_points
        )

        # Set initial state and covariance
        self.filterpy_ukf.x = self.mu.copy()
        self.filterpy_ukf.P = self.Sigma.copy()

        # Set initial measurement noise
        self.filterpy_ukf.R = np.array([[self.sigma_z**2]])

        # We'll set Q dynamically in predict_and_update
        self.filterpy_ukf.Q = np.eye(2) * (self.sigma_Q**2)

    def _fx(self, x, dt):
        """
        State transition function for FilterPy UKF.
        For stationary target, this is identity.
        
        Args:
            x: State vector [x, y]
            dt: Time step (not used)
            
        Returns:
            x_next: Next state (same as input for stationary target)
        """
        return x.copy()

    def _hx(self, x):
        """
        Measurement function for FilterPy UKF.
        
        Args:
            x: State vector [x, y]
            
        Returns:
            z: Expected measurement (scalar in array)
        """
        # Get current robot position from stored value
        robot_pos = getattr(self, '_current_robot_pos', np.array([0.0, 0.0]))
        signal = self._h(x, robot_pos)
        return np.array([signal])

    def predict_and_update(self, measurement, robot_pos):
        """
        Perform one complete UKF prediction and update step using FilterPy.

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
        self.filterpy_ukf.Q = Q
        
        # Set measurement noise
        R_current = self.R_est if self.adaptive_measurement_noise else self.sigma_z**2
        self.filterpy_ukf.R = np.array([[R_current]])

        # 3. Prediction step (FilterPy handles this)
        self.filterpy_ukf.predict()

        # 4. Update step - CORRECT FilterPy usage
        z = np.array([measurement])
        self.filterpy_ukf.update(z)

        # 5. Extract results
        self.mu = self.filterpy_ukf.x.copy()
        self.Sigma = self.filterpy_ukf.P.copy()

        # 6. Adaptive measurement noise update
        innovation = measurement - self._h(self.mu, robot_pos)
        self._update_adaptive_measurement_noise(innovation)

        return self.mu.copy(), self.Sigma.copy(), sigma_Q_current

    def reset(self, initial_mean=None, initial_covariance=None):
        """Reset filter to initial state."""
        # Call parent reset
        super().reset(initial_mean, initial_covariance)
        
        # Update FilterPy filter
        self.filterpy_ukf.x = self.mu.copy()
        self.filterpy_ukf.P = self.Sigma.copy()
        self.filterpy_ukf.R = np.array([[self.sigma_z**2]])
        self.filterpy_ukf.Q = np.eye(2) * (self.sigma_Q**2)

    def get_filterpy_filter(self):
        """Get the underlying FilterPy filter for advanced usage."""
        return self.filterpy_ukf

    def get_sigma_points(self):
        """Get current sigma points for visualization."""
        # Generate sigma points using FilterPy's method
        sigmas = self.sigma_points.sigma_points(self.mu, self.Sigma)
        return sigmas

    def get_ukf_parameters(self):
        """Get UKF-specific parameters."""
        # Calculate lambda manually since attribute name varies in FilterPy versions
        lambda_param = self.alpha**2 * (2 + self.kappa) - 2
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "kappa": self.kappa,
            "lambda": lambda_param,
            "n_sigma_points": 2 * 2 + 1  # 2*n + 1 for 2D state
        }

    def get_innovation_stats(self):
        """Get innovation statistics from FilterPy filter."""
        if hasattr(self.filterpy_ukf, 'y') and hasattr(self.filterpy_ukf, 'S'):
            return {
                'innovation': self.filterpy_ukf.y.copy() if self.filterpy_ukf.y is not None else None,
                'innovation_covariance': self.filterpy_ukf.S.copy() if self.filterpy_ukf.S is not None else None,
                'log_likelihood': self.filterpy_ukf.log_likelihood if hasattr(self.filterpy_ukf, 'log_likelihood') else None
            }
        return None