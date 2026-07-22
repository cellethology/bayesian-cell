"""cellsim -- Python port of the MATLAB cell-migration simulation in ``legacy/src/``.

Models a cell navigating a tissue ligand gradient by Bayesian inference
implemented in receptor biochemistry: the receptor density profile around the
membrane *is* the belief over source bearing, activity-driven recruitment is the
likelihood update, lateral diffusion is the motion model, and the finite
receptor pool enforces normalisation.

Covers the paths exercised by ``legacy/run.m``: feedback and uniform receptor schemes,
localisation on tissue environments, point and edge sources, the three decoders,
and signal-coupled diffusivity. Not ported: the explicit ``bayes`` scheme, the
retention/growth-cone geometry, and ``envmodel='grad'``.

Trajectories will not match MATLAB run-for-run -- the interpolator and RNG
differ, and the system diverges chaotically from perturbations at machine
precision. Ensemble statistics should agree; see ``tests/`` for the checks.
"""

from .cell import CellResult, cell_boundary, simulate_cell
from .decoder import next_position
from .ensemble import EnsembleResult, N_RUN, race, start_positions
from .environment import LigandField, RadialExpField, make_field
from .params import Params, conc2count, ellipse_perimeter, membrane_angles
from .receptor import cn_matrices, coupled_diffusivity, hill, receptor_activity

__all__ = [
    "Params", "conc2count", "ellipse_perimeter", "membrane_angles",
    "LigandField", "RadialExpField", "make_field",
    "hill", "receptor_activity", "cn_matrices", "coupled_diffusivity",
    "next_position",
    "simulate_cell", "cell_boundary", "CellResult",
    "race", "start_positions", "EnsembleResult", "N_RUN",
]
