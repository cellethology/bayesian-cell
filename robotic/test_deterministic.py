"""
Deterministic test with zero noise to isolate pure algorithmic differences.
"""

import numpy as np
import sys
import os

# Add current directory to path to import modules
sys.path.append(os.getcwd())

from bayes_navigation_old import run_navigation_simulation as run_old
from navigation_env import run_navigation_simulation as run_new

def test_deterministic(num_seeds=5, max_steps=1000):
    """Test both implementations with zero noise to isolate algorithmic differences."""
    
    # Shared configuration with ZERO noise
    config = {
        "grid_size": 100,
        "signal_strength_max": 0.2,
        "signal_decay_exp": 0.3,
        "kernel_size": 5,
        "target_reach_threshold": 5.0,
        "motion_decay_rate": 0.8,
    }
    
    # Old implementation config - ZERO NOISE
    old_config = config.copy()
    old_config.update({
        "true_motion_sigma": 0.0,  # ZERO motion noise
        "min_motion_sigma": 0.5,
        "max_motion_sigma": 0.5,
        "measurement_noise_factor": 0.0,  # ZERO measurement noise
        "movement_step_size": 0.2,
    })
    
    # New implementation config - ZERO NOISE
    new_config = config.copy()
    new_config.update({
        "motion_noise_type": "angular",
        "angular_noise_sigma": 0.0,  # ZERO angular noise
        "magnitude_noise_sigma": 0.0,  # ZERO magnitude noise
        "initial_process_sigma": 0.5,
        "step_size": 0.2,
        "adaptive_filtering": False,
        "adaptive_process_variance": "none",
        "noise_model": "gaussian",
        "noise_std": 0.0,  # ZERO measurement noise
        "initial_measurement_sigma": 0.0,  # ZERO measurement variance
    })
    
    print("DETERMINISTIC TEST (Zero Noise)")
    print("="*50)
    print("Testing pure algorithmic differences without any randomness")
    print()
    
    old_steps = []
    new_steps = []
    old_reached = []
    new_reached = []
    
    print("Seed | Old Steps | New Steps | Old Reached | New Reached | Identical?")
    print("-" * 65)
    
    for seed in range(num_seeds):
        # Run old implementation
        np.random.seed(seed)
        try:
            old_trajectory, old_env, old_sigmas = run_old(old_config, steps=max_steps)
            old_step_count = len(old_trajectory) - 1
            old_target_reached = old_step_count < max_steps
        except Exception as e:
            print(f"Old implementation failed at seed {seed}: {e}")
            old_step_count = max_steps
            old_target_reached = False
        
        # Run new implementation
        np.random.seed(seed)
        try:
            new_trajectory, new_env, new_sigmas, new_innovations, new_measurement_variances = run_new(new_config, steps=max_steps, verbose=False)
            new_step_count = len(new_trajectory) - 1
            new_target_reached = new_step_count < max_steps
        except Exception as e:
            print(f"New implementation failed at seed {seed}: {e}")
            new_step_count = max_steps
            new_target_reached = False
        
        old_steps.append(old_step_count)
        new_steps.append(new_step_count)
        old_reached.append(old_target_reached)
        new_reached.append(new_target_reached)
        
        # Check if results are identical
        identical = (old_step_count == new_step_count) and (old_target_reached == new_target_reached)
        identical_str = "YES" if identical else "NO"
        
        print(f"{seed:4d} | {old_step_count:9d} | {new_step_count:9d} | {old_target_reached:11} | {new_target_reached:11} | {identical_str:9}")
    
    # Calculate statistics
    old_steps = np.array(old_steps)
    new_steps = np.array(new_steps)
    old_reached = np.array(old_reached)
    new_reached = np.array(new_reached)
    
    print("\n" + "="*50)
    print("DETERMINISTIC TEST RESULTS")
    print("="*50)
    
    print(f"Success Rate:")
    print(f"  Old: {np.mean(old_reached)*100:.1f}% ({np.sum(old_reached)}/{num_seeds})")
    print(f"  New: {np.mean(new_reached)*100:.1f}% ({np.sum(new_reached)}/{num_seeds})")
    
    print(f"\nSteps to Target:")
    print(f"  Old: {np.mean(old_steps):.1f} ± {np.std(old_steps):.1f}")
    print(f"  New: {np.mean(new_steps):.1f} ± {np.std(new_steps):.1f}")
    
    # Check if results are identical across all seeds
    identical_results = np.array_equal(old_steps, new_steps) and np.array_equal(old_reached, new_reached)
    
    if identical_results:
        print(f"\n✓ RESULT: Implementations are IDENTICAL with zero noise")
        print(f"  → Differences are purely due to noise handling")
    else:
        print(f"\n✗ RESULT: Implementations differ even with zero noise")
        print(f"  → There are fundamental algorithmic differences")
        
        # Show specific differences
        step_diffs = new_steps - old_steps
        max_diff = np.max(np.abs(step_diffs))
        print(f"  → Maximum step difference: {max_diff}")
        if max_diff > 0:
            print(f"  → Step differences: {step_diffs}")

if __name__ == "__main__":
    test_deterministic(num_seeds=5, max_steps=1000)