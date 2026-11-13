"""
Exact linear-Gaussian Kalman filter over the TARGET POSITION (same state as the EKF,
so core/environment.py's move_robot / check_target_reached work unchanged).

Measurement is the 2D gradient readout g ~= (target - robot) + noise, so with
z := robot_pos + g the model is exactly z = x + noise, i.e. H = I.

This is the linear-Gaussian counterpart of filters/filterpy_ekf.py, built to isolate
which ingredient of the nonlinear EKF's Standard-vs-adaptive gap is really responsible:
here the measurement is full-rank (H = I) and bounded-gain (0 <= K <= I), so the
"lost mode" that drives the EKF gap cannot occur.
"""

from collections import namedtuple

import numpy as np
from filterpy.kalman import KalmanFilter

from .base_filter import BaseFilter
from ..shared import linear_field as lf

# The measurement "packet" handed from the environment to the filter.  `gradient` is the
# 2D vector readout used by the update; `signal` is the SCALAR signal-strength channel
# that plays the role of the Poisson count in the nonlinear setup (it is what the
# Coupled arm couples its process noise to).
GradientMeasurement = namedtuple("GradientMeasurement", ["gradient", "signal"])


class LinearGaussianKalmanFilter(BaseFilter):
    """Exact KF for the linear-Gaussian variant.  Same three arms as the EKF:
    Standard (fixed Q, fixed R), Coupled (Q coupled to the signal), Adaptive (R adapted
    from the residual)."""

    def __init__(self, config):
        super().__init__(config)

        self.dirs = lf.receptor_dirs(config)

        # ---- measurement noise, mirroring base_filter.py:41-44 -------------------
        # BaseFilter froze a SCALAR sigma_z at the initial geometry.  Here the same
        # idea gives a frozen 2x2.  "frozen" reproduces the EKF's design (and its
        # defect); the other modes exist to test whether that defect matters here.
        initial_robot_pos = np.array(config.get("robot_start_pos"), dtype=float)
        self.R_mode = config.get("linear_R_mode", "frozen")
        self.R_fixed = lf.R_at(self.mu, initial_robot_pos, config, self.dirs)
        self.R_est = self.R_fixed.copy()

        # "vector" -> H = I (rank 2, the source_kf setup)
        # "radial" -> H = u^T (rank 1, bearing unobservable) but still Gaussian,
        #             still bounded gain, still distance-independent information.
        #             This isolates rank-1-ness from the vanishing Jacobian.
        self.meas_mode = config.get("linear_measurement_mode", "vector")
        dim_z = 2 if self.meas_mode == "vector" else 1

        self.kf = KalmanFilter(dim_x=2, dim_z=dim_z)
        self.kf.x = self.mu.copy()
        self.kf.P = self.Sigma.copy()
        self.kf.F = np.eye(2)  # static target, as in filterpy_ekf.py:34
        self.kf.H = np.eye(2)  # <-- the whole point: full-rank, distance-independent
        self.kf.R = self.R_fixed.copy()
        if self.meas_mode != "vector":
            self.kf.H = np.array([[1.0, 0.0]])
            self.kf.R = np.array([[float(self.R_fixed[0, 0])]])
        self.kf.Q = np.eye(2) * (self.sigma_Q**2)

    # -- R selection -------------------------------------------------------------
    def _R_current(self, robot_pos):
        if self.adaptive_measurement_noise:
            return self.R_est
        if self.R_mode == "oracle":  # recompute from the current belief each step
            return lf.R_at(self.mu, robot_pos, self.config, self.dirs)
        return self.R_fixed  # "frozen" -- the EKF's behaviour

    # -- multivariate version of base_filter.py:117-121 ---------------------------
    def _update_adaptive_measurement_noise(self, innovation):
        if not self.adaptive_measurement_noise:
            return
        innovation = np.atleast_1d(innovation)
        if innovation.size == 2:
            outer = np.outer(innovation, innovation)
        else:  # radial mode: a scalar residual -> isotropic R_est
            outer = float(innovation[0]) ** 2 * np.eye(2)
        self.R_est = (1 - self.alpha_R) * self.R_est + self.alpha_R * outer

    def predict_and_update(self, measurement, robot_pos):
        g = np.asarray(measurement.gradient, dtype=float)
        signal = float(measurement.signal)

        # 1. process noise -- identical code path to the EKF, fed the scalar channel
        sigma_Q_current = self._determine_process_noise(signal)
        self.kf.Q = (sigma_Q_current**2) * np.eye(2)

        self.sigma_history.append(sigma_Q_current)
        self.R_est_history.append(float(np.trace(self._R_current(robot_pos)) / 2.0))

        # 2. measurement noise
        R2 = self._R_current(robot_pos) + 1e-12 * np.eye(2)

        # 3/4. exact predict + update
        self.kf.predict()
        z = np.asarray(robot_pos, dtype=float) + g  # z = target + noise
        if self.meas_mode == "vector":
            self.kf.R = R2
            self.kf.update(z)
        else:
            # rank-1 radial projection, evaluated at the PRIOR mean like the EKF's
            # Jacobian: only the range along u is observed, bearing is not.
            v = self.kf.x - np.asarray(robot_pos, dtype=float)
            n = np.linalg.norm(v)
            u = v / n if n > 1e-9 else np.array([1.0, 0.0])
            self.kf.H = u.reshape(1, 2)
            self.kf.R = np.array([[float(u @ R2 @ u)]])
            self.kf.update(np.array([float(u @ z)]))

        self.mu = self.kf.x.copy()
        self.Sigma = self.kf.P.copy()

        # 5. residual-based R adaptation.  Mirrors filterpy_ekf.py:111 exactly, i.e. it
        # uses the POST-fit residual by default (same known bias as the EKF arm).  Set
        # "adaptive_R_use_prefit": True for the textbook pre-fit innovation.
        if self.config.get("adaptive_R_use_prefit", False):
            innovation = np.asarray(self.kf.y, dtype=float).ravel()
        else:
            innovation = z - self.mu
            if self.meas_mode != "vector":
                innovation = np.array([float(u @ innovation)])
        self._update_adaptive_measurement_noise(innovation)

        return self.mu.copy(), self.Sigma.copy(), sigma_Q_current

    def reset(self, initial_mean=None, initial_covariance=None):
        super().reset(initial_mean, initial_covariance)
        self.R_est = self.R_fixed.copy()
        self.kf.x = self.mu.copy()
        self.kf.P = self.Sigma.copy()
        self.kf.R = self.R_fixed.copy()
        self.kf.Q = np.eye(2) * (self.sigma_Q**2)
