"""
Signal model for Bayesian navigation system.
Handles signal strength computation, measurements, and signal gradients.
"""

import numpy as np
from scipy.stats import poisson


def compute_signal_simple(distance_grid, signal_max, decay_exp):
    """Simple signal computation without JIT."""
    return signal_max * np.exp(-decay_exp * distance_grid)


def compute_likelihood_gaussian_simple(measurement, expected_signal, variance):
    """Simple Gaussian likelihood computation without JIT."""
    diff_sq = (measurement - expected_signal) ** 2
    return np.exp(-diff_sq / (2 * variance))


class SignalModel:
    """Manages signal strength computations and noisy measurements."""

    def __init__(self, config, grid_cache=None):
        self.config = config
        self.grid_size = config["grid_size"]

    def get_expected_signal(self, x, y, target_pos):
        """Compute expected signal strength at position (x, y) from target."""
        distance = np.sqrt((x - target_pos[0]) ** 2 + (y - target_pos[1]) ** 2)
        distance = np.maximum(distance, 1e-8)  # Avoid division by zero
        return self.config["signal_max"] * np.exp(
            -self.config["signal_decay"] * distance
        )

    def compute_all_expected_signal(self, target_pos):
        """Direct computation exactly like old implementation."""
        x = np.arange(self.grid_size)
        y = np.arange(self.grid_size)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        distance = np.sqrt((xx - target_pos[0]) ** 2 + (yy - target_pos[1]) ** 2)
        distance = np.maximum(distance, 0.1)  # Avoid division by zero
        signal = compute_signal_simple(
            distance,
            self.config["signal_max"],
            self.config["signal_decay"],
        )
        return signal

    def get_noisy_measurement(self, pos, target_pos):
        """Generate noisy measurement at given position."""
        expected_signal = self.get_expected_signal(pos[0], pos[1], target_pos)

        if self.config["noise_model"] == "poisson":
            # Ensure expected_signal is positive and finite for Poisson distribution
            expected_signal = np.maximum(expected_signal, 1e-8)
            if not np.isfinite(expected_signal):
                expected_signal = 1e-8
            signal = np.random.poisson(expected_signal)
        elif self.config["noise_model"] == "gaussian":
            noise = np.random.normal(0, self.config["noise_std"])
            signal = expected_signal + noise
        else:
            raise ValueError(f"Invalid noise model: {self.config['noise_model']}")
        return signal

    def get_signal_gradient(self, robot_pos, beacon_pos):
        """
        Computes the partial derivatives of the expected signal
        with respect to the beacon (target) position.

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

    def get_likelihood(self, measurement, expected_signal_grid, measurement_variance):
        """Compute likelihood of measurement given expected signal grid."""

        likelihood = compute_likelihood_gaussian_simple(
            measurement, expected_signal_grid, measurement_variance
        )
        likelihood = np.clip(likelihood, 1e-12, None)

        return likelihood
