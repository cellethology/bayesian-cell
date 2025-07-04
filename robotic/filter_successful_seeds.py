"""
Filter signal_max_comparison_results.csv for seeds where both methods are successful.
Selects seeds where both Signal-aware EKF and Standard EKF have steps_to_target < 3000.
"""

import pandas as pd
import numpy as np


def filter_successful_seeds(input_file="signal_max_comparison_results.csv", 
                          output_file="filtered_successful_seeds.csv",
                          threshold=3000):
    """
    Filter results for seeds where both Signal-aware EKF and Standard EKF are successful.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output filtered CSV file
        threshold: Maximum steps_to_target to consider successful (default 3000)
    
    Returns:
        pandas.DataFrame: Filtered results
    """
    print(f"Loading data from {input_file}...")
    
    # Load the results
    df = pd.read_csv(input_file)
    
    print(f"Original data shape: {df.shape}")
    print(f"Methods found: {df['method'].unique()}")
    print(f"Unique seeds: {len(df['seed'].unique())}")
    
    # Focus on the two main methods
    target_methods = ["Signal-aware EKF", "Standard EKF"]
    df_filtered = df[df['method'].isin(target_methods)].copy()
    
    print(f"Data shape after filtering methods: {df_filtered.shape}")
    
    # For each seed and signal_max combination, check if both methods are successful
    successful_combinations = []
    
    # Group by seed and signal_max
    for (seed, signal_max), group in df_filtered.groupby(['seed', 'signal_max']):
        # Check if we have both methods for this combination
        methods_present = set(group['method'].unique())
        
        if len(methods_present) == 2:  # Both methods present
            # Check if both are successful (steps_to_target < threshold)
            success_conditions = []
            
            for method in target_methods:
                method_data = group[group['method'] == method]
                if len(method_data) > 0:
                    steps = method_data['steps_to_target'].iloc[0]
                    success_conditions.append(steps < threshold)
                else:
                    success_conditions.append(False)
            
            # If both methods are successful, check performance ratio
            if all(success_conditions):
                # Get steps for each method
                signal_aware_steps = group[group['method'] == 'Signal-aware EKF']['steps_to_target'].iloc[0]
                standard_steps = group[group['method'] == 'Standard EKF']['steps_to_target'].iloc[0]
                
                # Additional filter: Signal-aware should be less than half of Standard
                if signal_aware_steps < (standard_steps / 2):
                    successful_combinations.extend(group.index.tolist())
    
    # Filter the original dataframe to keep only successful combinations
    filtered_df = df_filtered.loc[successful_combinations].copy()
    
    print(f"\nFiltering results:")
    print(f"Threshold: steps_to_target < {threshold}")
    print(f"Additional filter: Signal-aware steps < (Standard steps / 2)")
    print(f"Successful combinations found: {len(successful_combinations)}")
    print(f"Final filtered data shape: {filtered_df.shape}")
    
    # Summary statistics
    if len(filtered_df) > 0:
        unique_seeds = len(filtered_df['seed'].unique())
        unique_signal_max = len(filtered_df['signal_max'].unique())
        
        print(f"Unique successful seeds: {unique_seeds}")
        print(f"Signal max values: {sorted(filtered_df['signal_max'].unique())}")
        
        # Show summary by method
        print("\nSummary by method:")
        for method in target_methods:
            method_data = filtered_df[filtered_df['method'] == method]
            if len(method_data) > 0:
                mean_steps = method_data['steps_to_target'].mean()
                median_steps = method_data['steps_to_target'].median()
                print(f"{method}: {len(method_data)} runs, mean={mean_steps:.1f}, median={median_steps:.1f}")
        
        # Show performance ratios
        print("\nPerformance ratios (Signal-aware / Standard):")
        for (seed, signal_max), group in filtered_df.groupby(['seed', 'signal_max']):
            if len(group) == 2:  # Both methods present
                signal_aware_steps = group[group['method'] == 'Signal-aware EKF']['steps_to_target'].iloc[0]
                standard_steps = group[group['method'] == 'Standard EKF']['steps_to_target'].iloc[0]
                ratio = signal_aware_steps / standard_steps
                print(f"Seed {seed}, Signal max {signal_max}: {ratio:.3f} ({signal_aware_steps} / {standard_steps})")
    
    # Save filtered results
    filtered_df.to_csv(output_file, index=False)
    print(f"\nFiltered results saved to {output_file}")
    
    return filtered_df


if __name__ == "__main__":
    # Run the filtering
    filtered_results = filter_successful_seeds()
    
    print("\nFiltering complete!")