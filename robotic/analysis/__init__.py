"""
Analysis and comparison tools for target tracking simulations.

This package contains tools for comparing different filter configurations
and analyzing simulation results:
- FilterComparison: Multi-configuration comparison with statistical testing
- Signal max comparison utilities
- Results plotting and visualization
"""

from .comparison import FilterComparison

__all__ = [
    'FilterComparison'
]