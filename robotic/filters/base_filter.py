"""
Base filter interface for target tracking with signal-based observations.
Provides common interface for EKF and UKF implementations.
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseFilter(ABC):
    """Abstract base class for tracking filters."""

    def __init__(self, config):
        """
        Initialize filter with configuration parameters.

        Args:
            config: Dictionary containing filter configuration parameters
        """
        self.config = config

        # Environment parameters
        self.c0 = config.get("signal_max")
        self.lam = config.get("signal_decay")

        # Filter parameters
        self.sigma_Q = config.get("baseline_process_noise")
        self.sigma_u = config.get("actuator_noise")
        self.is_adaptive = config.get("adaptive_process_noise")
        self.alpha_R = config.get("alpha_R")
        self.adaptive_measurement_noise = config.get("adaptive_measurement_noise")
        self.eps = config.get("eps")

        # Initialize state
        self.mu = np.array(config.get("initial_belief_mean"))
        self.Sigma = np.eye(2) * config.get("initial_belief_variance")

        # Ensure positive definiteness
        self.Sigma += 1e-10 * np.eye(2)

        # Set measurement noise based on initial geometry
        initial_robot_pos = np.array(config.get("robot_start_pos"))
        self.sigma_z = np.sqrt(self._h(self.mu, initial_robot_pos))
        self.R_est = self.sigma_z**2

        # Store evolution for visualization
        self.sigma_history = []
        self.R_est_history = []

    def _h(self, mu, r):
        """Expected signal at robot pose r for target mean mu."""
        d = np.linalg.norm(mu - r)
        return self.c0 * np.exp(-self.lam * d)

    @abstractmethod
    def predict_and_update(self, measurement, robot_pos):
        """
        Perform one complete filter prediction and update step.

        Args:
            measurement: Observed signal measurement (scalar)
            robot_pos: Current robot position (2D numpy array)

        Returns:
            tuple: (updated_mu, updated_Sigma, current_sigma_Q)
        """
        pass

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
            self.mu = np.array(self.config.get("initial_belief_mean"))

        if initial_covariance is not None:
            self.Sigma = initial_covariance
        else:
            self.Sigma = np.eye(2) * self.config.get("initial_belief_variance")

        # Ensure positive definiteness
        self.Sigma += 1e-10 * np.eye(2)

        self.sigma_history = []
        self.R_est_history = []

        # Reset adaptive measurement noise estimate
        if hasattr(self, "config"):
            initial_robot_pos = np.array(self.config.get("robot_start_pos"))
            self.R_est = self._h(self.mu, initial_robot_pos)

    def _determine_process_noise(self, measurement):
        """Determine process noise (adaptive or fixed)."""
        if self.is_adaptive:
            # Use configurable epsilon to avoid division by zero, and add bounds
            sigma_Q_current = self.sigma_Q / np.sqrt(self.eps + measurement)

            # Add bounds to prevent extreme values
            sigma_Q_current = np.clip(sigma_Q_current, 0.01, 5.0)
        else:
            sigma_Q_current = self.sigma_Q

        return sigma_Q_current

    def _update_adaptive_measurement_noise(self, innovation):
        """Update adaptive measurement noise based on innovation."""
        if self.adaptive_measurement_noise:
            self.R_est = (1 - self.alpha_R) * self.R_est + self.alpha_R * (
                innovation**2
            )