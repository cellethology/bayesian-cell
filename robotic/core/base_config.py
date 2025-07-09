"""
Base configuration for EKF comparison experiments.
This file contains the shared parameters used across different comparison studies.
"""


def get_base_config():
    """
    Get the base configuration for EKF experiments.

    Returns:
        dict: Base configuration parameters
    """
    return {
        # Filter configuration
        "filter_type": "EKF",
        # Arena parameters
        "arena_min": 0.0,
        "arena_max": 200.0,
        "distance_tolerance": 3.0,
        # Signal parameters
        "signal_decay": 0.05,
        "signal_max": 50.0,
        # Robot parameters
        "robot_start_pos": [50.0, 50.0],
        "robot_step_size": 0.1,  # Smaller steps reduce overshooting
        "actuator_noise": 1.0,
        # Target parameters
        "target_true_pos": [120.0, 120.0],
        "target_motion_sigma": 0.5,
        # Filter initialization
        "initial_belief_mean": [100.0, 100.0],
        "initial_belief_variance": 10000.0,
        # Process noise parameters
        "baseline_process_noise": 1.0,  # Higher process noise reduces oscillations
        "adaptive_process_noise": False,
        "eps": 1.0,  # Epsilon parameter for adaptive process noise
        "adaptive_measurement_noise": False,
        "alpha_R": 0.01,  # Lower learning rate for more stable estimates
        # Simulation parameters
        "max_steps": 500000,
    }


def get_method_configs():
    """
    Get the standard method configurations for EKF comparisons.

    Returns:
        dict: Method configurations
    """
    return {
        "Standard EKF": {
            "adaptive_process_noise": False,
            "adaptive_measurement_noise": False,
        },
        "Signal-aware EKF": {
            "adaptive_process_noise": True,
            "adaptive_measurement_noise": False,
        },
        "Adaptive EKF": {
            "adaptive_process_noise": False,
            "adaptive_measurement_noise": True,
        },
    }


def get_signal_max_study_config():
    """
    Get specific configuration for signal max comparison study.

    Returns:
        dict: Study-specific parameters
    """
    return {
        "signal_decay": 0.05,  # Fixed for this study
        "n_runs": 400,
        "max_steps": 1200000,
        "signal_max_values": [5, 10, 15, 20, 25, 30, 35, 40],
    }
