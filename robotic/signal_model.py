"""
Signal model for Bayesian navigation system.
Handles signal strength computation, measurements, and signal gradients.
"""
import numpy as np
import numba
from scipy.stats import poisson
from utils import GridCache


@numba.jit(nopython=True, cache=True)
def compute_signal_fast(distance_grid, signal_max, decay_exp):
    """JIT-compiled fast signal computation."""
    return signal_max * np.exp(-decay_exp * distance_grid)


@numba.jit(nopython=True, cache=True)
def compute_likelihood_gaussian(measurement, expected_signal, variance):
    """JIT-compiled Gaussian likelihood computation."""
    diff_sq = (measurement - expected_signal) ** 2
    return np.exp(-diff_sq / (2 * variance)) / np.sqrt(2 * np.pi * variance)


class SignalModel:
    """Manages signal strength computations and noisy measurements."""
    
    def __init__(self, config, grid_cache):
        self.config = config
        self.grid_cache = grid_cache
    
    def get_expected_signal(self, x, y, target_pos):
        """Compute expected signal strength at position (x, y) from target."""
        distance = np.sqrt((x - target_pos[0]) ** 2 + (y - target_pos[1]) ** 2)
        distance = np.maximum(distance, 1e-8)  # Avoid division by zero
        return self.config["signal_strength_max"] * np.exp(
            -self.config["signal_decay_exp"] * distance
        )
    
    def compute_all_expected_signal(self, target_pos):
        """Optimized version using cached grids and JIT-compiled signal computation."""
        distance_grid = self.grid_cache.get_distance_grid(target_pos)
        signal = compute_signal_fast(
            distance_grid,
            self.config["signal_strength_max"],
            self.config["signal_decay_exp"]
        )
        return signal
    
    def get_noisy_measurement(self, pos, target_pos):
        """Generate noisy measurement at given position."""
        expected_signal = self.get_expected_signal(pos[0], pos[1], target_pos)
        
        if self.config["noise_model"] == "poisson":
            signal = np.random.poisson(expected_signal)
        elif self.config["noise_model"] == "gaussian":
            signal = np.random.normal(expected_signal, self.config["noise_std"])
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
        
        A = self.config["signal_strength_max"]
        k = self.config["signal_decay_exp"]
        
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
        if self.config["noise_model"] == "poisson":
            likelihood = poisson.pmf(measurement, expected_signal_grid)
            likelihood = np.clip(likelihood, 1e-12, None)
        elif self.config["noise_model"] == "gaussian":
            # Use JIT-compiled Gaussian likelihood computation
            likelihood = compute_likelihood_gaussian(
                measurement, expected_signal_grid, measurement_variance
            )
            likelihood = np.clip(likelihood, 1e-12, None)
        else:
            raise ValueError(f"Invalid noise model: {self.config['noise_model']}")
        return likelihood