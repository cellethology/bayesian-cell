"""
Investigate tie-breaking behavior in argmax for belief peak detection.
"""

import numpy as np
import sys
import os

# Add current directory to path to import modules
sys.path.append(os.getcwd())

from bayes_navigation_old import BayesianNavigation
from navigation_env import NavigationEnvironment

def analyze_tie_breaking(noise_level=0.5, seed=42):
    """Analyze how argmax handles ties in belief peak detection."""
    
    print("TIE-BREAKING ANALYSIS")
    print("="*50)
    print(f"Investigating argmax behavior with noise level: {noise_level}")
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
    
    # Get measurements and update beliefs
    np.random.seed(seed)
    old_measurement = old_env.get_noisy_measurement(robot_pos, old_env.true_target_pos)
    
    np.random.seed(seed)
    new_measurement = new_env.get_noisy_measurement(robot_pos, new_env.true_target_pos)
    
    # Apply updates
    old_env.measurement_update(old_measurement, robot_pos)
    old_env.motion_update(old_measurement)
    
    new_env.update_belief(new_measurement, robot_pos)
    new_env.motion_update(new_measurement, robot_pos)
    
    # Analyze belief peaks
    old_belief = old_env.belief
    new_belief = new_env.belief
    
    print("BELIEF PEAK ANALYSIS:")
    print(f"Belief array shapes: {old_belief.shape}")
    print()
    
    # Find maximum values
    old_max_val = np.max(old_belief)
    new_max_val = np.max(new_belief)
    
    print(f"Maximum belief values:")
    print(f"  Old: {old_max_val:.12f}")
    print(f"  New: {new_max_val:.12f}")
    print(f"  Difference: {abs(old_max_val - new_max_val):.2e}")
    print()
    
    # Find all locations with maximum value
    old_max_locations = np.where(old_belief == old_max_val)
    new_max_locations = np.where(new_belief == new_max_val)
    
    old_max_coords = list(zip(old_max_locations[0], old_max_locations[1]))
    new_max_coords = list(zip(new_max_locations[0], new_max_locations[1]))
    
    print(f"Number of cells with maximum value:")
    print(f"  Old: {len(old_max_coords)}")
    print(f"  New: {len(new_max_coords)}")
    print()
    
    if len(old_max_coords) > 1:
        print(f"Old implementation has TIES at locations:")
        for i, coord in enumerate(old_max_coords[:10]):  # Show first 10
            print(f"  {i+1}: {coord} (value: {old_belief[coord]:.12f})")
        if len(old_max_coords) > 10:
            print(f"  ... and {len(old_max_coords) - 10} more")
    
    if len(new_max_coords) > 1:
        print(f"New implementation has TIES at locations:")
        for i, coord in enumerate(new_max_coords[:10]):  # Show first 10
            print(f"  {i+1}: {coord} (value: {new_belief[coord]:.12f})")
        if len(new_max_coords) > 10:
            print(f"  ... and {len(new_max_coords) - 10} more")
    
    print()
    
    # Get argmax results
    old_argmax = np.unravel_index(np.argmax(old_belief), old_belief.shape)
    new_argmax = np.unravel_index(np.argmax(new_belief), new_belief.shape)
    
    print(f"Argmax results:")
    print(f"  Old: {old_argmax} (value: {old_belief[old_argmax]:.12f})")
    print(f"  New: {new_argmax} (value: {new_belief[new_argmax]:.12f})")
    print(f"  Same location: {old_argmax == new_argmax}")
    print()
    
    # Check if the beliefs are actually identical
    belief_diff = np.abs(old_belief - new_belief)
    max_diff = np.max(belief_diff)
    mean_diff = np.mean(belief_diff)
    
    print(f"Belief matrix comparison:")
    print(f"  Maximum difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    print(f"  Are beliefs identical: {np.allclose(old_belief, new_belief, atol=1e-15)}")
    print()
    
    # If beliefs are essentially identical but argmax differs, investigate further
    if max_diff < 1e-10 and old_argmax != new_argmax:
        print("→ CRITICAL FINDING: Beliefs are essentially identical but argmax differs!")
        print("  This suggests numerical precision or tie-breaking differences")
        
        # Check values at both argmax locations in both arrays
        print(f"\nDetailed value comparison:")
        print(f"  old_belief[old_argmax] = {old_belief[old_argmax]:.16f}")
        print(f"  old_belief[new_argmax] = {old_belief[new_argmax]:.16f}")
        print(f"  new_belief[old_argmax] = {new_belief[old_argmax]:.16f}")
        print(f"  new_belief[new_argmax] = {new_belief[new_argmax]:.16f}")
        
        # Check if it's truly a tie
        old_tie = np.isclose(old_belief[old_argmax], old_belief[new_argmax], atol=1e-15)
        new_tie = np.isclose(new_belief[old_argmax], new_belief[new_argmax], atol=1e-15)
        
        print(f"\nTie analysis:")
        print(f"  Old belief has tie: {old_tie}")
        print(f"  New belief has tie: {new_tie}")
        
    elif max_diff > 1e-10:
        print("→ Beliefs have significant numerical differences")
        print("  The argmax difference is due to actual belief differences")
        
        # Find where the largest differences occur
        max_diff_location = np.unravel_index(np.argmax(belief_diff), belief_diff.shape)
        print(f"  Largest difference at: {max_diff_location}")
        print(f"  Old value: {old_belief[max_diff_location]:.12f}")
        print(f"  New value: {new_belief[max_diff_location]:.12f}")
    else:
        print("→ Beliefs are identical and argmax agrees")
        print("  No tie-breaking issues detected")

if __name__ == "__main__":
    analyze_tie_breaking(noise_level=0.5, seed=42)