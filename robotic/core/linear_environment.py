"""
LinearGaussianEnvironment = EKFEnvironment with exactly TWO things changed:
  1. the filter is the exact linear KF instead of the EKF, and
  2. get_signal_measurement returns a 2D gradient readout instead of a scalar Poisson
     count.
Everything else (move_robot, move_target, check_target_reached, run_simulation,
trajectory bookkeeping, results dict) is inherited untouched.
"""

import numpy as np

from .environment import EKFEnvironment
from ..shared import linear_field as lf
from ..filters.linear_kf import GradientMeasurement, LinearGaussianKalmanFilter


class LinearGaussianEnvironment(EKFEnvironment):
    def __init__(self, config, verbose=False, random_seed=None):
        # EKFEnvironment.__init__ only knows "EKF"/"UKF"; let it build a throwaway EKF
        # (cheap, no RNG draws) and then swap in the linear filter.  This keeps
        # core/environment.py untouched.
        super().__init__(config, verbose=verbose, random_seed=random_seed)
        self.filter = LinearGaussianKalmanFilter(self.config)
        self.ekf = self.filter
        self.dirs = lf.receptor_dirs(self.config)

    def get_signal_measurement(self, step):
        """2D gradient readout + a scalar signal-strength channel."""
        g, conc, m = lf.sense_gradient(
            self.robot_pos, self.target_pos, self.config, self.robot_rng, self.dirs
        )
        channel = self.config.get("linear_signal_channel", "mean")
        signal = float(np.sum(m)) if channel == "sum" else float(np.mean(m))
        signal = max(signal, 0.0)  # the coupling channel is a "count"; keep it >= 0

        self.measurements[step] = signal  # stays 1-D, so downstream code is unaffected
        return GradientMeasurement(gradient=g, signal=signal)

    def compute_signal_field(self, target_pos=None):
        """Quadratic bump, for visualization only."""
        if target_pos is None:
            target_pos = self.target_pos
        arena_range = self.config["arena_max"] - self.config["arena_min"]
        grid = int(arena_range)
        x = np.linspace(self.config["arena_min"], self.config["arena_max"], grid)
        y = np.linspace(self.config["arena_min"], self.config["arena_max"], grid)
        X, Y = np.meshgrid(x, y)
        kap = lf.field_kappa(self.config)
        d2 = (X - target_pos[0]) ** 2 + (Y - target_pos[1]) ** 2
        return X, Y, np.maximum(self.config["signal_max"] - 0.5 * kap * d2, 0.0)
