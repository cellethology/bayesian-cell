"""
Unscented Kalman Filter implementation for target tracking.
Uses sigma points to handle nonlinear transformations more accurately than EKF.
"""

import numpy as np
from base_filter import BaseFilter


class UnscentedKalmanFilter(BaseFilter):
    """Unscented Kalman Filter for target tracking with signal-based observations."""

    def __init__(self, config):
        """
        Initialize UKF with configuration parameters.

        Args:
            config: Dictionary containing UKF configuration parameters
        """
        super().__init__(config)

        # UKF-specific parameters  
        self.alpha = config.get("ukf_alpha", 1.0)  # Spread parameter (larger for positive lambda)
        self.beta = config.get(
            "ukf_beta", 2.0
        )  # Distribution parameter (2 for Gaussian)
        self.kappa = config.get("ukf_kappa", 1.0)  # Secondary scaling parameter (3-n=1 for 2D)

        # State dimension
        self.n = 2  # 2D position

        # Compute lambda parameter
        self.lambda_param = self.alpha**2 * (self.n + self.kappa) - self.n

        # Compute weights
        self._compute_weights()

    def _compute_weights(self):
        """Compute weights for sigma points."""
        # Number of sigma points
        n_sigma = 2 * self.n + 1

        # Weights for mean
        self.Wm = np.zeros(n_sigma)
        self.Wm[0] = self.lambda_param / (self.n + self.lambda_param)
        self.Wm[1:] = 1.0 / (2 * (self.n + self.lambda_param))

        # Weights for covariance
        self.Wc = np.zeros(n_sigma)
        self.Wc[0] = self.lambda_param / (self.n + self.lambda_param) + (
            1 - self.alpha**2 + self.beta
        )
        self.Wc[1:] = 1.0 / (2 * (self.n + self.lambda_param))

    def _generate_sigma_points(self, mu, Sigma):
        """
        Generate sigma points using scaled unscented transform.

        Args:
            mu: Mean vector (2D)
            Sigma: Covariance matrix (2x2)

        Returns:
            sigma_points: Array of sigma points (2n+1 x 2)
        """
        n = len(mu)
        sigma_points = np.zeros((2 * n + 1, n))

        # Compute matrix square root
        try:
            # Use Cholesky decomposition for positive definite matrices
            L = np.linalg.cholesky(Sigma)
            sqrt_matrix = L * np.sqrt(n + self.lambda_param)
        except np.linalg.LinAlgError:
            # Fallback to eigenvalue decomposition if Cholesky fails
            eigenvals, eigenvecs = np.linalg.eigh(Sigma)
            eigenvals = np.maximum(eigenvals, 1e-12)  # Ensure positive eigenvalues
            sqrt_matrix = eigenvecs @ np.diag(np.sqrt(eigenvals)) * np.sqrt(n + self.lambda_param)

        # First sigma point is the mean
        sigma_points[0] = mu

        # Positive direction sigma points
        for i in range(n):
            sigma_points[1 + i] = mu + sqrt_matrix[:, i]

        # Negative direction sigma points
        for i in range(n):
            sigma_points[1 + n + i] = mu - sqrt_matrix[:, i]

        return sigma_points

    def _predict_sigma_points(self, sigma_points):
        """
        Predict sigma points through process model.
        For stationary target, this is identity transformation.

        Args:
            sigma_points: Input sigma points (2n+1 x 2)

        Returns:
            predicted_sigma_points: Predicted sigma points (2n+1 x 2)
        """
        # For stationary target, prediction is identity
        # In practice, you might add process noise here
        return sigma_points.copy()

    def _transform_sigma_points_measurement(self, sigma_points, robot_pos):
        """
        Transform sigma points through measurement model.

        Args:
            sigma_points: Sigma points (2n+1 x 2)
            robot_pos: Current robot position (2D)

        Returns:
            transformed_points: Measurement predictions (2n+1,)
        """
        n_sigma = sigma_points.shape[0]
        transformed_points = np.zeros(n_sigma)

        for i in range(n_sigma):
            transformed_points[i] = self._h(sigma_points[i], robot_pos)

        return transformed_points

    def predict_and_update(self, measurement, robot_pos):
        """
        Perform one complete UKF prediction and update step.

        Args:
            measurement: Observed signal measurement (scalar)
            robot_pos: Current robot position (2D numpy array)

        Returns:
            tuple: (updated_mu, updated_Sigma, current_sigma_Q)
        """
        # 1. Determine process noise
        sigma_Q_current = self._determine_process_noise(measurement)
        Q = (sigma_Q_current**2) * np.eye(2)

        # Store for visualization
        self.sigma_history.append(sigma_Q_current)
        self.R_est_history.append(self.R_est)

        # 2. Generate sigma points
        sigma_points = self._generate_sigma_points(self.mu, self.Sigma)

        # 3. Predict sigma points (process model)
        predicted_sigma_points = self._predict_sigma_points(sigma_points)

        # 4. Compute predicted mean and covariance
        mu_pred = np.zeros(2)
        for i in range(len(predicted_sigma_points)):
            mu_pred += self.Wm[i] * predicted_sigma_points[i]

        Sigma_pred = np.zeros((2, 2))
        for i in range(len(predicted_sigma_points)):
            diff = predicted_sigma_points[i] - mu_pred
            Sigma_pred += self.Wc[i] * np.outer(diff, diff)

        # Add process noise
        Sigma_pred += Q

        # 5. Transform sigma points through measurement model
        measurement_sigma_points = self._transform_sigma_points_measurement(
            predicted_sigma_points, robot_pos
        )

        # 6. Compute predicted measurement mean and covariance
        z_pred = np.zeros(1)
        for i in range(len(measurement_sigma_points)):
            z_pred += self.Wm[i] * measurement_sigma_points[i]

        # Measurement covariance
        R = self.R_est if self.adaptive_measurement_noise else self.sigma_z**2
        S = np.zeros((1, 1))
        for i in range(len(measurement_sigma_points)):
            diff = measurement_sigma_points[i] - z_pred
            S += self.Wc[i] * diff * diff
        S += R

        # 7. Compute cross-covariance
        Pxz = np.zeros((2, 1))
        for i in range(len(predicted_sigma_points)):
            state_diff = predicted_sigma_points[i] - mu_pred
            meas_diff = measurement_sigma_points[i] - z_pred
            Pxz += self.Wc[i] * np.outer(state_diff, meas_diff)

        # 8. Compute Kalman gain
        K = Pxz / S.item()  # S is 1x1, so convert to scalar

        # 9. Update step
        innovation = measurement - z_pred.item()
        self.mu = mu_pred + K.flatten() * innovation
        
        # Use simplified Joseph form for UKF (scalar measurement)
        # For UKF, we can use: P = P_pred - K * S * K^T (which is what we had)
        # But ensure K is properly shaped
        K_reshaped = K.reshape(-1, 1)  # Make it a column vector
        self.Sigma = Sigma_pred - K_reshaped @ S @ K_reshaped.T

        # 10. Ensure numerical stability
        self.Sigma = self._ensure_positive_definite(self.Sigma)

        # 11. Update adaptive measurement noise
        self._update_adaptive_measurement_noise(innovation)

        return self.mu.copy(), self.Sigma.copy(), sigma_Q_current

    def get_sigma_points(self):
        """Get current sigma points for visualization."""
        return self._generate_sigma_points(self.mu, self.Sigma)

    def get_ukf_parameters(self):
        """Get UKF-specific parameters."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "kappa": self.kappa,
            "lambda": self.lambda_param,
            "n_sigma_points": 2 * self.n + 1,
        }
