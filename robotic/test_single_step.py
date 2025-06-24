"""
Single-step analysis to compare what happens in one iteration.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add current directory to path to import modules
sys.path.append(os.getcwd())

from bayes_navigation_old import BayesianNavigation
from navigation_env import NavigationEnvironment

def compare_single_step(noise_level=0.5, seed=42):
    """Compare what happens in a single step between implementations."""
    
    print("SINGLE STEP ANALYSIS")
    print("="*50)
    print(f"Comparing one iteration with noise level: {noise_level}")
    print(f"Using seed: {seed}")
    print()
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Configuration for both implementations
    config = {
        "grid_size": 100,
        "signal_strength_max": 0.2,
        "signal_decay_exp": 0.3,
        "kernel_size": 5,
        "target_reach_threshold": 5.0,
        "motion_decay_rate": 0.8,
    }
    
    # Old implementation config
    old_config = config.copy()
    old_config.update({
        "true_motion_sigma": noise_level,
        "min_motion_sigma": 0.5,
        "max_motion_sigma": 0.5,
        "measurement_noise_factor": 0.06,
        "movement_step_size": 0.2,
    })
    
    # New implementation config
    new_config = config.copy()
    new_config.update({
        "motion_noise_type": "angular",
        "angular_noise_sigma": noise_level,
        "magnitude_noise_sigma": 0.0,
        "initial_process_sigma": 0.5,
        "step_size": 0.2,
        "adaptive_filtering": False,
        "adaptive_process_variance": "none",
        "noise_model": "gaussian",
        "noise_std": 0.06,
        "initial_measurement_sigma": 0.06,
    })
    
    # Initialize both implementations
    old_env = BayesianNavigation(old_config)
    new_env = NavigationEnvironment(new_config)
    
    # Starting position
    robot_pos = (20, 20)
    
    print(f"Initial robot position: {robot_pos}")
    print(f"Target position: {old_env.true_target_pos}")
    print()
    
    # Step 1: Get measurements (should be identical)
    np.random.seed(seed)
    old_measurement = old_env.get_noisy_measurement(robot_pos, old_env.true_target_pos)
    
    np.random.seed(seed)
    new_measurement = new_env.get_noisy_measurement(robot_pos, new_env.true_target_pos)
    
    print(f"Measurements:")
    print(f"  Old: {old_measurement:.6f}")
    print(f"  New: {new_measurement:.6f}")
    print(f"  Identical: {np.isclose(old_measurement, new_measurement)}")
    print()
    
    # Step 2: Store initial beliefs
    old_belief_initial = old_env.belief.copy()
    new_belief_initial = new_env.belief.copy()
    
    belief_diff_initial = np.max(np.abs(old_belief_initial - new_belief_initial))
    print(f"Initial belief difference (max): {belief_diff_initial:.2e}")
    
    # Step 3: Measurement update
    old_env.measurement_update(old_measurement, robot_pos)
    innovation, measure_var = new_env.update_belief(new_measurement, robot_pos)
    
    old_belief_after_measurement = old_env.belief.copy()
    new_belief_after_measurement = new_env.belief.copy()
    
    belief_diff_measurement = np.max(np.abs(old_belief_after_measurement - new_belief_after_measurement))
    print(f"Belief difference after measurement update (max): {belief_diff_measurement:.2e}")
    
    # Step 4: Motion update
    old_sigma = old_env.motion_update(old_measurement)
    new_sigma = new_env.motion_update(new_measurement, robot_pos)
    
    old_belief_after_motion = old_env.belief.copy()
    new_belief_after_motion = new_env.belief.copy()
    
    belief_diff_motion = np.max(np.abs(old_belief_after_motion - new_belief_after_motion))
    print(f"Belief difference after motion update (max): {belief_diff_motion:.2e}")
    
    print(f"Motion sigmas:")
    print(f"  Old: {old_sigma:.6f}")
    print(f"  New: {new_sigma:.6f}")
    print()
    
    # Step 5: Get actions
    old_action = old_env.get_next_intended_action(robot_pos)
    new_action = new_env.get_next_intended_action(robot_pos)
    
    print(f"Intended actions:")
    print(f"  Old: {old_action}")
    print(f"  New: {new_action}")
    print(f"  Identical: {np.allclose(old_action, new_action)}")
    print()
    
    # Step 6: Update positions
    np.random.seed(seed + 1000)  # Use different seed for motion noise
    old_new_pos = old_env.update_position(robot_pos, old_action)
    
    np.random.seed(seed + 1000)
    new_new_pos = new_env.update_position(robot_pos, new_action)
    
    print(f"New positions after motion:")
    print(f"  Old: {old_new_pos}")
    print(f"  New: {new_new_pos}")
    print(f"  Distance difference: {np.linalg.norm(np.array(old_new_pos) - np.array(new_new_pos)):.6f}")
    print()
    
    # Step 7: Analyze belief evolution
    print("BELIEF ANALYSIS:")
    print(f"  Old belief entropy: {-np.sum(old_belief_after_motion * np.log(old_belief_after_motion + 1e-12)):.3f}")
    print(f"  New belief entropy: {-np.sum(new_belief_after_motion * np.log(new_belief_after_motion + 1e-12)):.3f}")
    
    # Find belief peaks
    old_peak = np.unravel_index(np.argmax(old_belief_after_motion), old_belief_after_motion.shape)
    new_peak = np.unravel_index(np.argmax(new_belief_after_motion), new_belief_after_motion.shape)
    
    print(f"  Old belief peak: {old_peak} (strength: {old_belief_after_motion[old_peak]:.6f})")
    print(f"  New belief peak: {new_peak} (strength: {new_belief_after_motion[new_peak]:.6f})")
    
    # Calculate distances to true target
    old_peak_dist = np.linalg.norm(np.array(old_peak) - np.array(old_env.true_target_pos))
    new_peak_dist = np.linalg.norm(np.array(new_peak) - np.array(new_env.true_target_pos))
    
    print(f"  Old peak distance to target: {old_peak_dist:.3f}")
    print(f"  New peak distance to target: {new_peak_dist:.3f}")
    
    # Check if beliefs are significantly different
    if belief_diff_measurement > 1e-10:
        print(f"\n→ SIGNIFICANT DIFFERENCE found in measurement update!")
        print(f"  This suggests the measurement update implementations differ")
    elif belief_diff_motion > 1e-10:
        print(f"\n→ SIGNIFICANT DIFFERENCE found in motion update!")
        print(f"  This suggests the motion update implementations differ")
    else:
        print(f"\n→ Beliefs are essentially identical after both updates")
        print(f"  Differences must be in accumulation over time")
    
    # Visualize beliefs
    visualize_beliefs(old_belief_after_motion, new_belief_after_motion, old_env.true_target_pos, robot_pos)

def visualize_beliefs(old_belief, new_belief, target_pos, robot_pos):
    """Visualize belief states for comparison."""
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    # Old belief
    im1 = ax1.imshow(old_belief, cmap='hot', interpolation='nearest')
    ax1.plot(target_pos[1], target_pos[0], 'g*', markersize=15, label='Target')
    ax1.plot(robot_pos[1], robot_pos[0], 'bs', markersize=8, label='Robot')
    ax1.set_title('Old Implementation Belief')
    ax1.legend()
    plt.colorbar(im1, ax=ax1)
    
    # New belief
    im2 = ax2.imshow(new_belief, cmap='hot', interpolation='nearest')
    ax2.plot(target_pos[1], target_pos[0], 'g*', markersize=15, label='Target')
    ax2.plot(robot_pos[1], robot_pos[0], 'bs', markersize=8, label='Robot')
    ax2.set_title('New Implementation Belief')
    ax2.legend()
    plt.colorbar(im2, ax=ax2)
    
    # Difference
    diff = np.abs(old_belief - new_belief)
    im3 = ax3.imshow(diff, cmap='viridis', interpolation='nearest')
    ax3.plot(target_pos[1], target_pos[0], 'g*', markersize=15, label='Target')
    ax3.plot(robot_pos[1], robot_pos[0], 'bs', markersize=8, label='Robot')
    ax3.set_title('Absolute Difference')
    ax3.legend()
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.savefig('single_step_beliefs.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n→ Belief visualizations saved as 'single_step_beliefs.png'")

if __name__ == "__main__":
    compare_single_step(noise_level=0.5, seed=42)