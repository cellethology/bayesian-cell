"""
EKF Environment that orchestrates target tracking simulation.
Main coordinator for Extended Kalman Filter navigation simulation.
"""

import numpy as np
from ekf_filter import ExtendedKalmanFilter


class EKFEnvironment:
    """Main EKF environment that coordinates target tracking simulation."""
    
    def __init__(self, config=None, verbose=False):
        """
        Initialize EKF environment with configuration.
        
        Args:
            config: Configuration dictionary with simulation parameters
            verbose: Whether to print debug information
        """
        # Set up configuration with defaults
        self.config = {
            # Arena parameters
            "arena_min": 0.0,
            "arena_max": 200.0,
            "distance_tolerance": 2.0,
            "periodic_boundaries": False,  # Enable toroidal topology
            
            # Signal parameters
            "signal_max": 40.0,  # c0
            "signal_decay": 0.03,  # lambda
            
            # Robot parameters
            "robot_start_pos": [80.0, 80.0],
            "robot_step_size": 0.3,
            "actuator_noise": 0.2,  # sigma_u
            
            # Target parameters
            "target_true_pos": [120.0, 120.0],
            "target_motion_sigma": 0.5,  # sigma_Q baseline
            
            # EKF parameters
            "initial_belief_mean": [100.0, 100.0],
            "initial_belief_variance": 100.0,
            "baseline_process_noise": 0.5,
            "adaptive_process_noise": False,
            "alpha_R": 0.01,
            "adaptive_measurement_noise": False,
            
            # Simulation parameters
            "max_steps": 1000000,
            "random_seed": 55,
        }
        
        if config is not None:
            self.config.update(config)
            
        self.verbose = verbose
        
        # Don't set random seed here - let comparison script control it
        # Only set seed if explicitly requested for single simulations
        
        # Initialize positions
        self.robot_pos = np.array(self.config["robot_start_pos"], dtype=float)
        self.target_pos = np.array(self.config["target_true_pos"], dtype=float)
        
        # Initialize EKF
        self.ekf = ExtendedKalmanFilter(self.config)
        
        # Trajectory storage
        self.robot_trajectory = [self.robot_pos.copy()]
        self.target_trajectory = [self.target_pos.copy()]
        self.belief_history = []
        self.measurements = []
        
        if verbose:
            print(f"EKF Environment initialized:")
            print(f"  Robot start: {self.robot_pos}")
            print(f"  Target start: {self.target_pos}")
            print(f"  Adaptive process noise: {self.config['adaptive_process_noise']}")
            print(f"  Periodic boundaries: {self.config['periodic_boundaries']}")
    
    def _wrap_position(self, pos):
        """Wrap position around periodic boundaries."""
        if not self.config["periodic_boundaries"]:
            return np.clip(pos, self.config["arena_min"], self.config["arena_max"])
        
        arena_size = self.config["arena_max"] - self.config["arena_min"]
        wrapped = (pos - self.config["arena_min"]) % arena_size + self.config["arena_min"]
        return wrapped
    
    def _periodic_distance(self, pos1, pos2):
        """Calculate distance considering periodic boundaries."""
        if not self.config["periodic_boundaries"]:
            return np.linalg.norm(pos1 - pos2)
        
        arena_size = self.config["arena_max"] - self.config["arena_min"]
        diff = pos1 - pos2
        
        # Find shortest distance in each dimension considering wrapping
        for i in range(len(diff)):
            if abs(diff[i]) > arena_size / 2:
                if diff[i] > 0:
                    diff[i] -= arena_size
                else:
                    diff[i] += arena_size
        
        return np.linalg.norm(diff)
    
    def _get_shortest_direction(self, from_pos, to_pos):
        """Get direction vector for shortest path considering periodic boundaries."""
        if not self.config["periodic_boundaries"]:
            return to_pos - from_pos
        
        arena_size = self.config["arena_max"] - self.config["arena_min"]
        diff = to_pos - from_pos
        
        # Adjust each dimension to take shortest path
        for i in range(len(diff)):
            if diff[i] > arena_size / 2:
                diff[i] -= arena_size
            elif diff[i] < -arena_size / 2:
                diff[i] += arena_size
        
        return diff
    
    def get_signal_measurement(self):
        """Generate noisy signal measurement at current robot position."""
        # True expected signal using periodic distance
        distance = self._periodic_distance(self.target_pos, self.robot_pos)
        lambda_true = self.config["signal_max"] * np.exp(-self.config["signal_decay"] * distance)
        
        # Poisson noise
        measurement = np.random.poisson(lambda_true)
        self.measurements.append(measurement)
        
        return measurement
    
    def update_belief(self, measurement):
        """Update target belief using EKF."""
        mu, Sigma, sigma_Q = self.ekf.predict_and_update(measurement, self.robot_pos)
        self.belief_history.append((mu.copy(), Sigma.copy()))
        return mu, Sigma, sigma_Q
    
    def move_robot(self):
        """Move robot towards current target estimate."""
        # Get current target estimate
        mu, _ = self.ekf.get_belief_state()
        
        # Calculate direction vector considering periodic boundaries
        direction = self._get_shortest_direction(self.robot_pos, mu)
        if np.linalg.norm(direction) > 1e-6:
            # Move towards estimate
            self.robot_pos += self.config["robot_step_size"] * direction / np.linalg.norm(direction)
        
        # Add actuator noise
        self.robot_pos += np.random.randn(2) * self.config["actuator_noise"]
        
        # Apply boundary conditions (clip for non-periodic, wrap for periodic)
        self.robot_pos = self._wrap_position(self.robot_pos)
        
        self.robot_trajectory.append(self.robot_pos.copy())
    
    def move_target(self):
        """Move target with random walk."""
        self.target_pos += np.random.randn(2) * self.config["target_motion_sigma"]
        
        # Apply boundary conditions (clip for non-periodic, wrap for periodic)
        self.target_pos = self._wrap_position(self.target_pos)
        
        self.target_trajectory.append(self.target_pos.copy())
    
    def check_target_reached(self):
        """Check if robot is close enough to target."""
        distance = self._periodic_distance(self.robot_pos, self.target_pos)
        return distance < self.config["distance_tolerance"]
    
    def run_simulation(self, max_steps=None):
        """
        Run complete EKF simulation.
        
        Args:
            max_steps: Maximum number of simulation steps (overrides config)
            
        Returns:
            dict: Simulation results including trajectories and final state
        """
        if max_steps is None:
            max_steps = self.config["max_steps"]
        
        if self.verbose:
            print(f"Starting EKF simulation for up to {max_steps} steps...")
        
        for step in range(max_steps):
            # 1. Robot senses environment
            measurement = self.get_signal_measurement()
            
            # 2. Update belief with EKF
            mu, Sigma, sigma_Q = self.update_belief(measurement)
            
            # 3. Move robot towards estimated target
            self.move_robot()
            
            # 4. Move target randomly
            self.move_target()
            
            # 5. Check termination condition
            if self.check_target_reached():
                if self.verbose:
                    print(f"Target reached at step {step}!")
                    print(f"Final target position: {self.target_pos}")
                    print(f"Final robot position: {self.robot_pos}")
                break
            
            # Progress reporting
            if self.verbose and step % 10000 == 0:
                distance = self._periodic_distance(self.robot_pos, self.target_pos)
                print(f"Step {step}: Distance to target = {distance:.2f}")
        
        # Prepare results
        results = {
            "steps_completed": step + 1,
            "target_reached": self.check_target_reached(),
            "robot_trajectory": np.array(self.robot_trajectory),
            "target_trajectory": np.array(self.target_trajectory),
            "belief_history": self.belief_history,
            "measurements": np.array(self.measurements),
            "sigma_history": self.ekf.get_sigma_history(),
            "R_est_history": self.ekf.get_R_est_history(),
            "final_belief": self.ekf.get_belief_state(),
            "final_target_pos": self.target_pos.copy(),
            "final_robot_pos": self.robot_pos.copy(),
            "config": self.config.copy()
        }
        
        if self.verbose:
            print(f"Simulation completed in {results['steps_completed']} steps")
            print(f"Target reached: {results['target_reached']}")
        
        return results
    
    def reset(self):
        """Reset environment to initial state."""
        # Don't reset random seed here - let comparison script control it
        
        # Reset positions
        self.robot_pos = np.array(self.config["robot_start_pos"], dtype=float)
        self.target_pos = np.array(self.config["target_true_pos"], dtype=float)
        
        # Reset EKF
        self.ekf.reset()
        
        # Clear trajectories
        self.robot_trajectory = [self.robot_pos.copy()]
        self.target_trajectory = [self.target_pos.copy()]
        self.belief_history = []
        self.measurements = []
    
    def compute_signal_field(self, target_pos=None):
        """
        Compute signal field over the entire arena for visualization.
        
        Args:
            target_pos: Target position for signal computation (uses current if None)
            
        Returns:
            tuple: (x_grid, y_grid, signal_grid)
        """
        if target_pos is None:
            target_pos = self.target_pos
        
        # Create grid
        arena_range = self.config["arena_max"] - self.config["arena_min"]
        grid_size = int(arena_range)  # 1 unit resolution
        x = np.linspace(self.config["arena_min"], self.config["arena_max"], grid_size)
        y = np.linspace(self.config["arena_min"], self.config["arena_max"], grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Compute signal at each grid point
        signal_grid = np.zeros_like(X)
        target_pos = np.array(target_pos)
        
        for i in range(grid_size):
            for j in range(grid_size):
                pos = np.array([X[i, j], Y[i, j]])
                distance = self._periodic_distance(pos, target_pos)
                signal_grid[i, j] = self.config["signal_max"] * np.exp(-self.config["signal_decay"] * distance)
        
        return X, Y, signal_grid