"""
Filter factory for creating EKF or UKF instances based on configuration.
Provides unified interface for FilterPy-based filter creation and validation.
"""

from filterpy_ekf import FilterPyExtendedKalmanFilter
from filterpy_ukf import FilterPyUnscentedKalmanFilter


class FilterFactory:
    """Factory class for creating FilterPy-based tracking filters."""

    @staticmethod
    def create_filter(config):
        """
        Create filter instance based on configuration.

        Args:
            config: Dictionary containing filter configuration
                   Must include 'filter_type' key with value 'FilterPy_EKF' or 'FilterPy_UKF'

        Returns:
            Filter instance (FilterPy EKF or UKF)

        Raises:
            ValueError: If filter_type is not supported
        """
        filter_type = config.get("filter_type", "FilterPy_EKF").upper()

        if filter_type == "FILTERPY_EKF":
            return FilterPyExtendedKalmanFilter(config)
        elif filter_type == "FILTERPY_UKF":
            return FilterPyUnscentedKalmanFilter(config)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}. Use 'FilterPy_EKF' or 'FilterPy_UKF'")

    @staticmethod
    def get_default_config(filter_type="FilterPy_EKF"):
        """
        Get default configuration for specified filter type.

        Args:
            filter_type: Type of filter ("FilterPy_EKF" or "FilterPy_UKF")

        Returns:
            Dictionary with default configuration parameters
        """
        # Base configuration common to all filters
        base_config = {
            # Arena parameters
            "arena_min": 0.0,
            "arena_max": 200.0,
            "distance_tolerance": 5.0,
            # Signal parameters
            "signal_max": 50.0,
            "signal_decay": 0.01,
            # Robot parameters
            "robot_start_pos": [60.0, 60.0],
            "robot_step_size": 0.3,
            "actuator_noise": 0.5,
            # Target parameters
            "target_true_pos": [140.0, 140.0],
            "target_motion_sigma": 0.3,
            # Filter parameters
            "initial_belief_mean": [100.0, 100.0],
            "initial_belief_variance": 600.0,
            "baseline_process_noise": 0.3,
            "adaptive_process_noise": True,
            "alpha_R": 0.1,
            "adaptive_measurement_noise": False,
            "eps": 1.0,  # Epsilon parameter for adaptive process noise
            # Simulation parameters
            "max_steps": 100000,
            "random_seed": 42,
        }

        filter_type = filter_type.upper()
        
        if filter_type == "FILTERPY_EKF":
            base_config["filter_type"] = "FilterPy_EKF"
            
        elif filter_type == "FILTERPY_UKF":
            base_config["filter_type"] = "FilterPy_UKF"
            # FilterPy UKF-specific parameters
            base_config.update({
                "ukf_alpha": 0.1,    # FilterPy works well with smaller alpha
                "ukf_beta": 2.0,     # Distribution parameter
                "ukf_kappa": 0.0,    # Secondary scaling parameter
            })
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        return base_config

    @staticmethod
    def validate_config(config):
        """
        Validate filter configuration.

        Args:
            config: Configuration dictionary

        Returns:
            bool: True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        required_keys = [
            "signal_max", "signal_decay", "initial_belief_mean", 
            "initial_belief_variance", "baseline_process_noise"
        ]
        
        # Optional parameters with defaults
        optional_params = {
            "eps": 1.0,
            "alpha_R": 0.1,
            "adaptive_process_noise": False,
            "adaptive_measurement_noise": False
        }

        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        # Set defaults for optional parameters
        for key, default_value in optional_params.items():
            if key not in config:
                config[key] = default_value

        # Validate eps parameter
        if config.get("eps", 1.0) <= 0:
            raise ValueError("eps must be positive (> 0)")

        filter_type = config.get("filter_type", "FilterPy_EKF").upper()
        
        if filter_type not in ["FILTERPY_EKF", "FILTERPY_UKF"]:
            raise ValueError(f"Invalid filter_type: {filter_type}. Must be 'FilterPy_EKF' or 'FilterPy_UKF'")

        # Validate UKF-specific parameters
        if filter_type == "FILTERPY_UKF":
            ukf_keys = ["ukf_alpha", "ukf_beta", "ukf_kappa"]
            for key in ukf_keys:
                if key not in config:
                    # Set default values if missing
                    defaults = {"ukf_alpha": 0.1, "ukf_beta": 2.0, "ukf_kappa": 0.0}
                    config[key] = defaults[key]

            # Validate UKF parameter ranges
            if not (0 < config["ukf_alpha"] <= 1):
                raise ValueError("ukf_alpha must be in range (0, 1]")
            if config["ukf_beta"] < 0:
                raise ValueError("ukf_beta must be non-negative (2.0 optimal for Gaussian)")

        return True

    @staticmethod
    def get_supported_filters():
        """Get list of supported filter types."""
        return ["FilterPy_EKF", "FilterPy_UKF"]

    @staticmethod
    def compare_filters():
        """Get comparison information between filter types."""
        return {
            "FilterPy_EKF": {
                "description": "FilterPy Extended Kalman Filter (RECOMMENDED)",
                "advantages": [
                    "Robust, well-tested implementation",
                    "Excellent numerical stability",
                    "Advanced features (innovation stats)",
                    "Validated against official examples",
                    "Active maintenance and community"
                ],
                "disadvantages": [
                    "External dependency"
                ],
                "best_for": "Most EKF applications - production ready"
            },
            "FilterPy_UKF": {
                "description": "FilterPy Unscented Kalman Filter (RECOMMENDED)",
                "advantages": [
                    "Robust, well-tested implementation",
                    "Excellent numerical stability",
                    "No linearization required",
                    "Advanced sigma point methods",
                    "Better handling of nonlinear systems"
                ],
                "disadvantages": [
                    "External dependency",
                    "Computationally more expensive"
                ],
                "best_for": "Highly nonlinear systems - production ready"
            }
        }


def create_filter(config):
    """Convenience function for creating filters."""
    return FilterFactory.create_filter(config)


def get_default_config(filter_type="FilterPy_EKF"):
    """Convenience function for getting default configuration."""
    return FilterFactory.get_default_config(filter_type)