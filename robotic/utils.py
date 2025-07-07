"""
Shared utility functions for the robotic target tracking simulation.
Contains common functions used across multiple modules.
"""

import numpy as np


def bootstrap_median_ci(data, n_bootstrap=1000, confidence=0.95):
    """
    Calculate bootstrap confidence interval for median.

    Args:
        data: Array-like data
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0.95 for 95% CI)

    Returns:
        tuple: (median, ci_lower, ci_upper)
    """
    if len(data) == 0:
        return np.nan, np.nan, np.nan

    data = np.array(data)
    median_val = np.median(data)

    # Bootstrap samples
    bootstrap_medians = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_medians.append(np.median(bootstrap_sample))

    # Calculate confidence interval
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_medians, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_medians, 100 * (1 - alpha / 2))

    return median_val, ci_lower, ci_upper


def bootstrap_mean_ci(data, n_bootstrap=1000, confidence=0.95):
    """
    Calculate bootstrap confidence interval for mean.

    Args:
        data: Array-like data
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0.95 for 95% CI)

    Returns:
        tuple: (mean, ci_lower, ci_upper)
    """
    if len(data) == 0:
        return np.nan, np.nan, np.nan

    data = np.array(data)
    mean_val = np.mean(data)

    # Bootstrap samples
    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    # Calculate confidence interval
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return mean_val, ci_lower, ci_upper


def validate_config_keys(config, required_keys, optional_keys=None):
    """
    Validate that a configuration dictionary has required keys.

    Args:
        config: Configuration dictionary to validate
        required_keys: List of required keys
        optional_keys: Dictionary of optional keys with default values

    Returns:
        dict: Validated and completed configuration

    Raises:
        ValueError: If required keys are missing
    """
    # Check required keys
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")

    # Add optional keys with defaults
    if optional_keys:
        for key, default_value in optional_keys.items():
            if key not in config:
                config[key] = default_value

    return config


def ensure_output_directory(path):
    """
    Ensure output directory exists, create if necessary.

    Args:
        path: Path to output file or directory
    """
    import os
    
    if os.path.isfile(path):
        # If it's a file path, get the directory
        directory = os.path.dirname(path)
    else:
        # Assume it's a directory path
        directory = path
    
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def format_time_delta(seconds):
    """
    Format time duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def safe_divide(numerator, denominator, default=0.0):
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if denominator is zero

    Returns:
        float: Result of division or default value
    """
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator


def clip_to_bounds(value, min_val, max_val):
    """
    Clip value to specified bounds.

    Args:
        value: Value to clip
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clipped value
    """
    return np.clip(value, min_val, max_val)


def calculate_distance(pos1, pos2):
    """
    Calculate Euclidean distance between two positions.

    Args:
        pos1: First position (array-like)
        pos2: Second position (array-like)

    Returns:
        float: Euclidean distance
    """
    diff = np.array(pos1) - np.array(pos2)
    return np.sqrt(np.sum(diff**2))


def get_timestamp_string():
    """
    Get current timestamp as a string for file naming.

    Returns:
        str: Timestamp string in format YYYYMMDD_HHMMSS
    """
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_config_from_file(filepath):
    """
    Load configuration from JSON file.

    Args:
        filepath: Path to JSON configuration file

    Returns:
        dict: Configuration dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file contains invalid JSON
    """
    import json
    import os
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file {filepath}: {e}")


def save_config_to_file(config, filepath):
    """
    Save configuration to JSON file.

    Args:
        config: Configuration dictionary
        filepath: Path to save JSON file
    """
    import json
    
    ensure_output_directory(filepath)
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2, default=str)


def print_config_summary(config, title="Configuration Summary"):
    """
    Print a formatted summary of configuration parameters.

    Args:
        config: Configuration dictionary
        title: Title for the summary
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    # Group related parameters
    groups = {
        "Filter": ["filter_type", "adaptive_process_noise", "adaptive_measurement_noise"],
        "Arena": ["arena_min", "arena_max", "distance_tolerance"],
        "Signal": ["signal_max", "signal_decay"],
        "Robot": ["robot_start_pos", "robot_step_size", "actuator_noise"],
        "Target": ["target_true_pos", "target_motion_sigma"],
        "Simulation": ["max_steps", "random_seed"]
    }
    
    for group_name, keys in groups.items():
        group_items = [(key, config.get(key)) for key in keys if key in config]
        if group_items:
            print(f"\n{group_name}:")
            for key, value in group_items:
                print(f"  {key}: {value}")
    
    # Print any remaining parameters
    covered_keys = set()
    for keys in groups.values():
        covered_keys.update(keys)
    
    remaining = [(key, value) for key, value in config.items() if key not in covered_keys]
    if remaining:
        print(f"\nOther:")
        for key, value in remaining:
            print(f"  {key}: {value}")
    
    print(f"{'='*60}")


def generate_default_output_path(base_name, extension="png", include_timestamp=True):
    """
    Generate a default output file path.

    Args:
        base_name: Base name for the file
        extension: File extension (without dot)
        include_timestamp: Whether to include timestamp in filename

    Returns:
        str: Generated file path
    """
    import os
    
    if include_timestamp:
        timestamp = get_timestamp_string()
        filename = f"{base_name}_{timestamp}.{extension}"
    else:
        filename = f"{base_name}.{extension}"
    
    return os.path.join("output", filename)


def validate_filter_config(config):
    """
    Validate filter configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check filter type
    filter_type = config.get("filter_type")
    if filter_type not in ["FilterPy_EKF", "FilterPy_UKF"]:
        raise ValueError(f"Invalid filter_type: {filter_type}. Must be 'FilterPy_EKF' or 'FilterPy_UKF'")
    
    # Check required numeric parameters
    required_numeric = [
        "signal_max", "signal_decay", "baseline_process_noise",
        "actuator_noise", "initial_belief_variance"
    ]
    
    for param in required_numeric:
        if param not in config:
            raise ValueError(f"Missing required parameter: {param}")
        if not isinstance(config[param], (int, float)) or config[param] <= 0:
            raise ValueError(f"Parameter {param} must be a positive number")
    
    # Check required array parameters
    required_arrays = ["initial_belief_mean", "robot_start_pos", "target_true_pos"]
    for param in required_arrays:
        if param not in config:
            raise ValueError(f"Missing required parameter: {param}")
        if not isinstance(config[param], (list, tuple)) or len(config[param]) != 2:
            raise ValueError(f"Parameter {param} must be a 2-element array")
    
    # Check boolean parameters
    boolean_params = ["adaptive_process_noise", "adaptive_measurement_noise"]
    for param in boolean_params:
        if param in config and not isinstance(config[param], bool):
            raise ValueError(f"Parameter {param} must be boolean")
    
    # UKF-specific parameter validation
    if filter_type == "FilterPy_UKF":
        ukf_params = {"ukf_alpha": (0, 1), "ukf_beta": (0, None), "ukf_kappa": (None, None)}
        for param, (min_val, max_val) in ukf_params.items():
            if param in config:
                val = config[param]
                if not isinstance(val, (int, float)):
                    raise ValueError(f"UKF parameter {param} must be numeric")
                if min_val is not None and val <= min_val:
                    raise ValueError(f"UKF parameter {param} must be > {min_val}")
                if max_val is not None and val >= max_val:
                    raise ValueError(f"UKF parameter {param} must be < {max_val}")
    
    return True