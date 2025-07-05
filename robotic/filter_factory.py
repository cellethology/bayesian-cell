"""
Filter factory for creating EKF or UKF instances based on configuration.
Provides unified interface for filter creation and validation.
"""

from ekf_filter import ExtendedKalmanFilter
from ukf_filter import UnscentedKalmanFilter
from filterpy_ekf import FilterPyExtendedKalmanFilter
from filterpy_ukf import FilterPyUnscentedKalmanFilter
from filterpy_ekf_corrected import FilterPyExtendedKalmanFilterCorrected
from filterpy_ukf_corrected import FilterPyUnscentedKalmanFilterCorrected


class FilterFactory:
    """Factory class for creating tracking filters."""

    @staticmethod
    def create_filter(config):
        """
        Create filter instance based on configuration.

        Args:
            config: Dictionary containing filter configuration
                   Must include 'filter_type' key with value 'EKF', 'UKF', 'FilterPy_EKF', or 'FilterPy_UKF'

        Returns:
            Filter instance (EKF or UKF)

        Raises:
            ValueError: If filter_type is not supported
        """
        filter_type = config.get("filter_type", "FilterPy_EKF_Corrected").upper()

        if filter_type == "EKF":
            return ExtendedKalmanFilter(config)
        elif filter_type == "UKF":
            return UnscentedKalmanFilter(config)
        elif filter_type == "FILTERPY_EKF":
            return FilterPyExtendedKalmanFilter(config)
        elif filter_type == "FILTERPY_UKF":
            return FilterPyUnscentedKalmanFilter(config)
        elif filter_type == "FILTERPY_EKF_CORRECTED":
            return FilterPyExtendedKalmanFilterCorrected(config)
        elif filter_type == "FILTERPY_UKF_CORRECTED":
            return FilterPyUnscentedKalmanFilterCorrected(config)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}. Use 'EKF', 'UKF', 'FilterPy_EKF', 'FilterPy_UKF', 'FilterPy_EKF_Corrected', or 'FilterPy_UKF_Corrected'")

    @staticmethod
    def get_default_config(filter_type="FilterPy_EKF_Corrected"):
        """
        Get default configuration for specified filter type.

        Args:
            filter_type: Type of filter ("EKF", "UKF", "FilterPy_EKF", or "FilterPy_UKF")

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
            # Simulation parameters
            "max_steps": 100000,
            "random_seed": 42,
        }

        filter_type = filter_type.upper()
        
        if filter_type == "EKF":
            base_config["filter_type"] = "EKF"
            # Custom EKF-specific parameters (none currently)
            
        elif filter_type == "UKF":
            base_config["filter_type"] = "UKF"
            # Custom UKF-specific parameters
            base_config.update({
                "ukf_alpha": 1.0,    # Spread parameter (ensures positive lambda)
                "ukf_beta": 2.0,     # Distribution parameter
                "ukf_kappa": 1.0,    # Secondary scaling parameter (3-n for 2D)
            })
        elif filter_type == "FILTERPY_EKF":
            base_config["filter_type"] = "FilterPy_EKF"
            # FilterPy EKF-specific parameters (none currently)
            
        elif filter_type == "FILTERPY_UKF":
            base_config["filter_type"] = "FilterPy_UKF"
            # FilterPy UKF-specific parameters
            base_config.update({
                "ukf_alpha": 0.1,    # FilterPy works well with smaller alpha
                "ukf_beta": 2.0,     # Distribution parameter
                "ukf_kappa": 0.0,    # Secondary scaling parameter
            })
        elif filter_type == "FILTERPY_EKF_CORRECTED":
            base_config["filter_type"] = "FilterPy_EKF_Corrected"
            # Corrected FilterPy EKF-specific parameters (none currently)
            
        elif filter_type == "FILTERPY_UKF_CORRECTED":
            base_config["filter_type"] = "FilterPy_UKF_Corrected"
            # Corrected FilterPy UKF-specific parameters
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

        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        filter_type = config.get("filter_type", "EKF").upper()
        
        if filter_type not in ["EKF", "UKF"]:
            raise ValueError(f"Invalid filter_type: {filter_type}. Must be 'EKF' or 'UKF'")

        # Validate UKF-specific parameters
        if filter_type == "UKF":
            ukf_keys = ["ukf_alpha", "ukf_beta", "ukf_kappa"]
            for key in ukf_keys:
                if key not in config:
                    # Set default values if missing
                    defaults = {"ukf_alpha": 0.001, "ukf_beta": 2.0, "ukf_kappa": 0.0}
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
        return ["EKF", "UKF", "FilterPy_EKF", "FilterPy_UKF", "FilterPy_EKF_Corrected", "FilterPy_UKF_Corrected"]

    @staticmethod
    def compare_filters():
        """Get comparison information between filter types."""
        return {
            "EKF": {
                "description": "Custom Extended Kalman Filter",
                "advantages": [
                    "Custom implementation",
                    "Direct control over algorithm",
                    "Good for learning/research"
                ],
                "disadvantages": [
                    "Potential numerical issues",
                    "Limited testing",
                    "May have implementation bugs"
                ],
                "best_for": "Research and algorithm development"
            },
            "UKF": {
                "description": "Custom Unscented Kalman Filter",
                "advantages": [
                    "Custom implementation",
                    "No linearization required",
                    "Direct control over sigma points"
                ],
                "disadvantages": [
                    "Complex implementation",
                    "Numerical stability issues",
                    "Parameter tuning challenges"
                ],
                "best_for": "Research on UKF algorithms"
            },
            "FilterPy_EKF": {
                "description": "FilterPy Extended Kalman Filter (RECOMMENDED)",
                "advantages": [
                    "Robust, well-tested implementation",
                    "Excellent numerical stability",
                    "Advanced features (innovation stats)",
                    "Widely used in production",
                    "Better error handling"
                ],
                "disadvantages": [
                    "External dependency",
                    "Less direct control"
                ],
                "best_for": "Production systems requiring reliable EKF performance"
            },
            "FilterPy_UKF": {
                "description": "FilterPy Unscented Kalman Filter (RECOMMENDED)",
                "advantages": [
                    "Robust, well-tested implementation",
                    "Excellent numerical stability",
                    "No linearization required",
                    "Advanced sigma point methods",
                    "Better covariance handling"
                ],
                "disadvantages": [
                    "External dependency",
                    "Computationally more expensive"
                ],
                "best_for": "Production systems requiring robust nonlinear filtering"
            },
            "FilterPy_EKF_Corrected": {
                "description": "Corrected FilterPy Extended Kalman Filter (BEST CHOICE)",
                "advantages": [
                    "Correct FilterPy API usage patterns",
                    "Based on official FilterPy notebooks",
                    "Robust, well-tested implementation",
                    "Excellent numerical stability",
                    "Advanced features (innovation stats)",
                    "Validated against official examples"
                ],
                "disadvantages": [
                    "External dependency"
                ],
                "best_for": "All EKF applications - replaces other EKF implementations"
            },
            "FilterPy_UKF_Corrected": {
                "description": "Corrected FilterPy Unscented Kalman Filter (BEST CHOICE)",
                "advantages": [
                    "Correct FilterPy API usage patterns",
                    "Based on official FilterPy notebooks",
                    "Robust, well-tested implementation",
                    "Excellent numerical stability",
                    "No linearization required",
                    "Advanced sigma point methods"
                ],
                "disadvantages": [
                    "External dependency",
                    "Computationally more expensive"
                ],
                "best_for": "All UKF applications - replaces other UKF implementations"
            }
        }


def create_filter(config):
    """Convenience function for creating filters."""
    return FilterFactory.create_filter(config)


def get_default_config(filter_type="EKF"):
    """Convenience function for getting default configuration."""
    return FilterFactory.get_default_config(filter_type)