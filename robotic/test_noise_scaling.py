"""
Noise scaling experiment to identify at what noise level differences emerge.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add current directory to path to import modules
sys.path.append(os.getcwd())

from bayes_navigation_old import run_navigation_simulation as run_old
from navigation_env import run_navigation_simulation as run_new

def test_noise_scaling(noise_levels=None, num_seeds=5, max_steps=5000):
    """Test how performance difference scales with noise level."""
    
    if noise_levels is None:
        noise_levels = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Shared configuration
    base_config = {
        "grid_size": 100,
        "signal_strength_max": 0.2,
        "signal_decay_exp": 0.3,
        "kernel_size": 5,
        "target_reach_threshold": 5.0,
        "motion_decay_rate": 0.8,
    }
    
    results = {
        'noise_levels': noise_levels,
        'old_means': [],
        'old_stds': [],
        'new_means': [],
        'new_stds': [],
        'differences': [],
        'old_success_rates': [],
        'new_success_rates': []
    }
    
    print("NOISE SCALING EXPERIMENT")
    print("="*60)
    print(f"Testing {len(noise_levels)} noise levels with {num_seeds} seeds each")
    print()
    
    for noise_level in noise_levels:
        print(f"Testing noise level: {noise_level}")
        
        # Old implementation config
        old_config = base_config.copy()
        old_config.update({
            "true_motion_sigma": noise_level,
            "min_motion_sigma": 0.5,
            "max_motion_sigma": 0.5,
            "measurement_noise_factor": 0.06,  # Keep measurement noise constant
            "movement_step_size": 0.2,
        })
        
        # New implementation config
        new_config = base_config.copy()
        new_config.update({
            "motion_noise_type": "angular",
            "angular_noise_sigma": noise_level,
            "magnitude_noise_sigma": 0.0,
            "initial_process_sigma": 0.5,
            "step_size": 0.2,
            "adaptive_filtering": False,
            "adaptive_process_variance": "none",
            "noise_model": "gaussian",
            "noise_std": 0.06,  # Keep measurement noise constant
            "initial_measurement_sigma": 0.06,
        })
        
        old_steps = []
        new_steps = []
        old_reached = []
        new_reached = []
        
        for seed in range(num_seeds):
            # Run old implementation
            np.random.seed(seed)
            try:
                old_trajectory, old_env, old_sigmas = run_old(old_config, steps=max_steps)
                old_step_count = len(old_trajectory) - 1
                old_target_reached = old_step_count < max_steps
            except Exception as e:
                old_step_count = max_steps
                old_target_reached = False
            
            # Run new implementation
            np.random.seed(seed)
            try:
                new_trajectory, new_env, new_sigmas, new_innovations, new_measurement_variances = run_new(new_config, steps=max_steps, verbose=False)
                new_step_count = len(new_trajectory) - 1
                new_target_reached = new_step_count < max_steps
            except Exception as e:
                new_step_count = max_steps
                new_target_reached = False
            
            old_steps.append(old_step_count)
            new_steps.append(new_step_count)
            old_reached.append(old_target_reached)
            new_reached.append(new_target_reached)
        
        # Calculate statistics
        old_steps = np.array(old_steps)
        new_steps = np.array(new_steps)
        old_reached = np.array(old_reached)
        new_reached = np.array(new_reached)
        
        old_mean = np.mean(old_steps)
        old_std = np.std(old_steps)
        new_mean = np.mean(new_steps)
        new_std = np.std(new_steps)
        difference = old_mean - new_mean
        
        old_success_rate = np.mean(old_reached)
        new_success_rate = np.mean(new_reached)
        
        results['old_means'].append(old_mean)
        results['old_stds'].append(old_std)
        results['new_means'].append(new_mean)
        results['new_stds'].append(new_std)
        results['differences'].append(difference)
        results['old_success_rates'].append(old_success_rate)
        results['new_success_rates'].append(new_success_rate)
        
        print(f"  Noise {noise_level:5.3f}: Old={old_mean:6.1f}±{old_std:4.1f}, New={new_mean:6.1f}±{new_std:4.1f}, Diff={difference:+6.1f}")
    
    # Print summary
    print("\n" + "="*60)
    print("NOISE SCALING RESULTS")
    print("="*60)
    
    print("Noise  | Old Steps | New Steps | Difference | Old Success | New Success")
    print("-" * 70)
    for i, noise in enumerate(noise_levels):
        print(f"{noise:5.3f} | {results['old_means'][i]:9.1f} | {results['new_means'][i]:9.1f} | {results['differences'][i]:+10.1f} | {results['old_success_rates'][i]:10.1%} | {results['new_success_rates'][i]:10.1%}")
    
    # Find critical noise level
    critical_noise = None
    for i, diff in enumerate(results['differences']):
        if abs(diff) > 100:  # Significant difference threshold
            critical_noise = noise_levels[i]
            break
    
    if critical_noise is not None:
        print(f"\n→ Significant differences emerge at noise level: {critical_noise}")
    else:
        print(f"\n→ No significant differences found at tested noise levels")
    
    # Create visualization
    plot_noise_scaling_results(results)
    
    return results

def plot_noise_scaling_results(results):
    """Create plots showing how performance scales with noise."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    noise_levels = results['noise_levels']
    
    # Plot 1: Mean steps vs noise level
    ax1.errorbar(noise_levels, results['old_means'], yerr=results['old_stds'], 
                label='Old Implementation', marker='o', capsize=5)
    ax1.errorbar(noise_levels, results['new_means'], yerr=results['new_stds'], 
                label='New Implementation', marker='s', capsize=5)
    ax1.set_xlabel('Motion Noise Level')
    ax1.set_ylabel('Steps to Target')
    ax1.set_title('Performance vs Noise Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance difference vs noise level
    ax2.plot(noise_levels, results['differences'], marker='o', color='red')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Motion Noise Level')
    ax2.set_ylabel('Difference (Old - New)')
    ax2.set_title('Performance Difference vs Noise')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Success rates vs noise level
    ax3.plot(noise_levels, results['old_success_rates'], marker='o', label='Old Implementation')
    ax3.plot(noise_levels, results['new_success_rates'], marker='s', label='New Implementation')
    ax3.set_xlabel('Motion Noise Level')
    ax3.set_ylabel('Success Rate')
    ax3.set_title('Success Rate vs Noise Level')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Log scale for differences
    ax4.semilogy(noise_levels, np.abs(results['differences']), marker='o', color='red')
    ax4.set_xlabel('Motion Noise Level')
    ax4.set_ylabel('|Difference| (log scale)')
    ax4.set_title('Absolute Difference (Log Scale)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('noise_scaling_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n→ Plots saved as 'noise_scaling_analysis.png'")

if __name__ == "__main__":
    # Use fewer parameters for faster testing
    noise_levels = [0.01, 0.1, 0.3, 0.5]
    results = test_noise_scaling(noise_levels=noise_levels, num_seeds=3, max_steps=2000)