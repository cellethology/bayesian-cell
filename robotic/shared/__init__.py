"""
Shared utilities for the robotic target tracking simulation.

This package contains utility functions used across multiple modules:
- Bootstrap statistical methods
- Configuration validation
- File I/O utilities
- Common mathematical operations
"""

from .utils import (
    bootstrap_median_ci,
    bootstrap_mean_ci,
    validate_config_keys,
    ensure_output_directory,
    format_time_delta,
    safe_divide,
    clip_to_bounds,
    calculate_distance,
    get_timestamp_string,
    load_config_from_file,
    save_config_to_file,
    print_config_summary,
    generate_default_output_path,
    validate_filter_config
)

__all__ = [
    'bootstrap_median_ci',
    'bootstrap_mean_ci', 
    'validate_config_keys',
    'ensure_output_directory',
    'format_time_delta',
    'safe_divide',
    'clip_to_bounds',
    'calculate_distance',
    'get_timestamp_string',
    'load_config_from_file',
    'save_config_to_file',
    'print_config_summary',
    'generate_default_output_path',
    'validate_filter_config'
]