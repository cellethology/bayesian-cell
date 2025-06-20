"""
Motion model for Bayesian navigation system.
Handles robot movement, motion kernels, and adaptive process variance.
"""
import numpy as np
from numpy.random import normal
from utils import RandomBatchGenerator


class MotionModel:
    """Manages robot motion, kernels, and adaptive process variance."""
    
    def __init__(self, config):
        self.config = config
        self.grid_size = config["grid_size"]
        
        # Precompute kernel distance matrix
        kernel_size = config["kernel_size"]
        center = kernel_size // 2
        y, x = np.ogrid[:kernel_size, :kernel_size]
        self.kernel_mat = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        
        # Motion kernel cache
        self.kernel_cache = {}
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        # Random number generator for efficient batch processing
        self.random_gen = RandomBatchGenerator()
        
        # Adaptive process variance tracking
        self.process_variance = config["initial_process_sigma"] ** 2
        self.previous_intended_pos = None
    
    def get_optimal_motion_kernel(self, theta_0, adaptive_sigma):
        """
        Optimized version with kernel caching and stable matrix operations.
        Returns a 2D Gaussian kernel based on motion uncertainty.
        """
        # Create cache key (round to avoid floating point precision issues)
        cache_key = (round(theta_0, 6), round(adaptive_sigma, 6))
        
        if cache_key in self.kernel_cache:
            self.cache_hit_count += 1
            return self.kernel_cache[cache_key]
        
        self.cache_miss_count += 1
        
        s = self.config["step_size"]
        sigma_theta = adaptive_sigma
        sigma_d = 0.1 * s
        
        cos_t = np.cos(theta_0)
        sin_t = np.sin(theta_0)
        
        # Mean displacement
        dx = s * cos_t
        dy = s * sin_t
        mu = np.array([dx, dy], dtype=np.float32)
        
        # Covariance matrix
        var_x = (s**2) * (sin_t**2) * (sigma_theta**2) + (cos_t**2) * (sigma_d**2)
        var_y = (s**2) * (cos_t**2) * (sigma_theta**2) + (sin_t**2) * (sigma_d**2)
        cov_xy = sin_t * cos_t * ((sigma_d**2) - (s**2) * (sigma_theta**2))
        cov = np.array([[var_x, cov_xy], [cov_xy, var_y]], dtype=np.float32)
        
        # Generate grid
        size = self.config["kernel_size"]
        half = size // 2
        y, x = np.mgrid[-half : half + 1, -half : half + 1]
        pos = np.stack([x, y], axis=-1, dtype=np.float32)
        
        # Evaluate multivariate Gaussian using Cholesky decomposition for stability
        diff = pos - mu
        
        # Add regularization and use Cholesky decomposition
        reg_cov = cov + np.eye(2, dtype=np.float32) * 1e-6
        try:
            L = np.linalg.cholesky(reg_cov)
            y_vals = np.linalg.solve(L, diff.reshape(-1, 2).T).T
            exponent = np.sum(y_vals**2, axis=1).reshape(size, size)
        except np.linalg.LinAlgError:
            # Fallback to standard method if Cholesky fails
            inv_cov = np.linalg.solve(reg_cov, np.eye(2))
            exponent = np.einsum("...i,ij,...j->...", diff, inv_cov, diff)
        
        kernel = np.exp(-0.5 * exponent)
        kernel /= kernel.sum()
        
        # Cache the result if cache is not too large
        if len(self.kernel_cache) < 256:
            self.kernel_cache[cache_key] = kernel
        
        return kernel
    
    def get_adaptive_motion_sigma(self, signal_strength):
        """
        Calculate motion model uncertainty based on signal strength.
        D(s) = D_base + (D_max - D_base) * exp(-k * s)
        """
        D_base = self.config["min_motion_sigma"]
        D_max = self.config["initial_process_sigma"]
        k = self.config["motion_decay_rate"]
        
        return D_base + (D_max - D_base) * np.exp(-k * signal_strength)
    
    def get_next_intended_action(self, robot_pos, belief):
        """Get unnormalized step vector towards estimated target."""
        estimated_target_pos = np.unravel_index(np.argmax(belief), belief.shape)
        
        dx = estimated_target_pos[0] - robot_pos[0]
        dy = estimated_target_pos[1] - robot_pos[1]
        return dx, dy
    
    def update_position(self, true_pos, action):
        """
        Update the true robot position based on intended action.
        Uses isotropic Gaussian motion model with true_process_sigma noise.
        """
        s = self.config["step_size"]
        sigma = self.config["true_process_sigma"]
        
        # Normalize action direction
        norm = np.linalg.norm(action)
        direction = np.array(action) / norm
        
        # Intended step
        dx = s * direction[0]
        dy = s * direction[1]
        
        # Add isotropic Gaussian noise using batch random generation
        random_vals = self.random_gen.get_batch(2) * sigma
        noisy_dx = dx + random_vals[0]
        noisy_dy = dy + random_vals[1]
        
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
    
    def update_process_variance_from_motion_error(self, actual_pos_after_motion):
        """
        Update process variance based on observed motion error.
        Compares actual position after motion with intended position.
        """
        if self.previous_intended_pos is None:
            return  # No previous intended position to compare with
        
        # Calculate error between intended and actual position
        dx = actual_pos_after_motion[0] - self.previous_intended_pos[0]
        dy = actual_pos_after_motion[1] - self.previous_intended_pos[1]
        observed_error_sq = dx**2 + dy**2
        
        adaptation_rate = self.config["adaptation_rate"]
        self.process_variance = (
            1 - adaptation_rate
        ) * self.process_variance + adaptation_rate * max(
            observed_error_sq, self.config["min_allowed_variance"]
        )