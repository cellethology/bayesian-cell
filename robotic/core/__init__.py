"""
Core simulation components for target tracking.

This package contains the essential components for running simulations:
- EKFEnvironment: Main simulation environment coordinator
- EKFVisualizer: Visualization and plotting utilities
- Configuration functions for simulation parameters
"""

from .environment import EKFEnvironment
from .visualization import EKFVisualizer
from .base_config import get_base_config, get_method_configs, get_signal_max_study_config

__all__ = [
    'EKFEnvironment',
    'EKFVisualizer',
    'get_base_config',
    'get_method_configs', 
    'get_signal_max_study_config'
]