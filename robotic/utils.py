"""
Utility functions and JIT-compiled helpers for Bayesian navigation.
"""
import numpy as np
import numba
from functools import lru_cache


@numba.jit(nopython=True, cache=True)
def compute_variance_stats(grid_indices, belief):
    """JIT-compiled variance computation for belief state."""
    μx = np.sum(grid_indices[0] * belief)
    μy = np.sum(grid_indices[1] * belief)
    var_x = np.sum(((grid_indices[0] - μx) ** 2) * belief)
    var_y = np.sum(((grid_indices[1] - μy) ** 2) * belief)
    return var_x, var_y


class GridCache:
    """Manages precomputed grids and coordinate systems."""
    
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.setup_grids()
    
    def setup_grids(self):
        """Precompute grid coordinates for performance."""
        x = np.arange(self.grid_size, dtype=np.float32)
        y = np.arange(self.grid_size, dtype=np.float32)
        self.xx, self.yy = np.meshgrid(x, y, indexing="ij")
        self.grid_indices = np.indices((self.grid_size, self.grid_size), dtype=np.float32)
    
    @lru_cache(maxsize=128)
    def get_distance_grid(self, target_pos):
        """Cached computation of distance grid for a given target position."""
        target_x, target_y = target_pos
        distance = np.sqrt((self.xx - target_x) ** 2 + (self.yy - target_y) ** 2)
        return np.maximum(distance, 1e-8)  # Avoid division by zero


class RandomBatchGenerator:
    """Efficient batch generation of random numbers."""
    
    def __init__(self, buffer_size=1000):
        self.buffer_size = buffer_size
        self.buffer = None
        self.index = 0
    
    def get_batch(self, count=2):
        """Get random numbers from pre-generated batch for better performance."""
        if self.buffer is None or self.index + count > len(self.buffer):
            # Regenerate buffer
            self.buffer = np.random.normal(0, 1, self.buffer_size)
            self.index = 0
        
        result = self.buffer[self.index:self.index + count]
        self.index += count
        return result