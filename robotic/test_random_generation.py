"""
Test to isolate random number generation differences between implementations.
"""

import numpy as np
import sys
import os

# Add current directory to path to import modules
sys.path.append(os.getcwd())

from bayes_navigation_old import run_navigation_simulation as run_old
from navigation_env import run_navigation_simulation as run_new

def test_random_generation(num_seeds=10, max_steps=5000):
    """Test if RandomBatchGenerator vs direct random calls causes the difference."""
    
    # Shared configuration
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
        "true_motion_sigma": 0.5,
        "min_motion_sigma": 0.5,
        "max_motion_sigma": 0.5,
        "measurement_noise_factor": 0.06,
        "movement_step_size": 0.2,
    })
    
    # New implementation config (batch generator)
    new_batch_config = config.copy()
    new_batch_config.update({
        "motion_noise_type": "angular",
        "angular_noise_sigma": 0.5,
        "magnitude_noise_sigma": 0.0,
        "initial_process_sigma": 0.5,
        "step_size": 0.2,
        "adaptive_filtering": False,
        "adaptive_process_variance": "none",
        "noise_model": "gaussian",
        "noise_std": 0.06,
        "initial_measurement_sigma": 0.06,
    })
    
    # New implementation config (direct random)
    new_direct_config = new_batch_config.copy()
    
    print("RANDOM GENERATION COMPARISON")
    print("="*60)
    print("Testing if RandomBatchGenerator vs direct np.random calls causes difference")
    print()
    
    old_steps = []
    new_batch_steps = []
    new_direct_steps = []
    
    print("Seed | Old Steps | New (Batch) | New (Direct) | Batch vs Direct")
    print("-" * 65)
    
    for seed in range(num_seeds):
        # Run old implementation
        np.random.seed(seed)
        try:
            old_trajectory, old_env, old_sigmas = run_old(old_config, steps=max_steps)
            old_step_count = len(old_trajectory) - 1
        except Exception as e:
            print(f"Old implementation failed at seed {seed}: {e}")
            old_step_count = max_steps
        
        # Run new implementation with batch generator
        np.random.seed(seed)
        try:
            new_trajectory, new_env, new_sigmas, new_innovations, new_measurement_variances = run_new(new_batch_config, steps=max_steps, verbose=False)
            # Ensure batch generator is used
            new_env.motion_model.use_direct_random = False
            new_batch_step_count = len(new_trajectory) - 1
        except Exception as e:
            print(f"New implementation (batch) failed at seed {seed}: {e}")
            new_batch_step_count = max_steps
        
        # Run new implementation with direct random calls
        np.random.seed(seed)
        try:
            new_trajectory, new_env, new_sigmas, new_innovations, new_measurement_variances = run_new(new_direct_config, steps=max_steps, verbose=False)
            # Force direct random usage
            new_env.motion_model.use_direct_random = True
            # Re-run to get results with direct random
            np.random.seed(seed)
            new_trajectory, new_env, new_sigmas, new_innovations, new_measurement_variances = run_new(new_direct_config, steps=max_steps, verbose=False)
            new_direct_step_count = len(new_trajectory) - 1
        except Exception as e:
            print(f"New implementation (direct) failed at seed {seed}: {e}")
            new_direct_step_count = max_steps
        
        old_steps.append(old_step_count)
        new_batch_steps.append(new_batch_step_count)
        new_direct_steps.append(new_direct_step_count)
        
        # Compare batch vs direct
        batch_direct_diff = new_direct_step_count - new_batch_step_count
        batch_direct_str = f"{batch_direct_diff:+d}" if batch_direct_diff != 0 else "0"
        
        print(f"{seed:4d} | {old_step_count:9d} | {new_batch_step_count:11d} | {new_direct_step_count:12d} | {batch_direct_str:12}")
    
    # Calculate statistics
    old_steps = np.array(old_steps)
    new_batch_steps = np.array(new_batch_steps)
    new_direct_steps = np.array(new_direct_steps)
    
    print("\n" + "="*60)
    print("RANDOM GENERATION TEST RESULTS")
    print("="*60)
    
    print(f"Steps to Target (mean ± std):")
    print(f"  Old:           {np.mean(old_steps):.1f} ± {np.std(old_steps):.1f}")
    print(f"  New (Batch):   {np.mean(new_batch_steps):.1f} ± {np.std(new_batch_steps):.1f}")
    print(f"  New (Direct):  {np.mean(new_direct_steps):.1f} ± {np.std(new_direct_steps):.1f}")
    
    # Compare differences
    old_vs_batch = np.mean(old_steps) - np.mean(new_batch_steps)
    old_vs_direct = np.mean(old_steps) - np.mean(new_direct_steps)
    batch_vs_direct = np.mean(new_batch_steps) - np.mean(new_direct_steps)
    
    print(f"\nPerformance Differences:")
    print(f"  Old vs New (Batch):   {old_vs_batch:+.1f} steps")
    print(f"  Old vs New (Direct):  {old_vs_direct:+.1f} steps")
    print(f"  Batch vs Direct:      {batch_vs_direct:+.1f} steps")
    
    # Check if random generation method explains the difference
    if abs(batch_vs_direct) > 100:
        print(f"\n→ Random generation method makes a significant difference!")
        if abs(old_vs_direct) < abs(old_vs_batch):
            print(f"  Direct random calls are closer to old implementation")
        else:
            print(f"  Batch generator is closer to old implementation")
    else:
        print(f"\n→ Random generation method is NOT the main cause of difference")
        print(f"  The difference is elsewhere in the algorithm")

if __name__ == "__main__":
    test_random_generation(num_seeds=10, max_steps=5000)