"""source_kf -- linear-Gaussian source localization as an EXACT Kalman filter.

The agent knows its own position but not the source's. The concentration field is a
quadratic bump peaked at the source, so its gradient ``s - p`` is linear in the source
position; sensing that gradient (with noise) gives a direct linear observation of the
source. The belief over the source stays Gaussian and its covariance shrinks like
σ/√N -- an exact Kalman filter, now genuinely driven by sensing (blind => uniform
belief, agent lost).

This is the linear-Gaussian counterpart of the nonlinear ``source_seek`` (bearing +
distance, grid filter), and the source-finding counterpart of ``kalman_nav`` (which
estimated the agent's own position with a handed direction).

Usage::

    python -m source_kf                   # run and save a figure
    from source_kf import Config, run, plot_run
    r = run(Config(), seed=0); plot_run(r, save_path="out.png")
"""

from .plot import plot_run
from .sim import Config, Result, SourceKF, run

__all__ = ["Config", "Result", "SourceKF", "run", "plot_run"]
