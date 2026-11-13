"""Linear-Gaussian source localization -- an EXACT Kalman filter, with a motion model.

State = the VECTOR TO THE SOURCE, ``d = s - p`` (what the gradient actually measures).
The SOURCE drifts (a diffusing target); the agent moves deterministically. This folds
the target's motion into the filter exactly:

    d_{t+1} = s_{t+1} - p_{t+1} = (s_t + v_t) - (p_t + u_t) = d_t - u_t + v_t

so a commanded move ``u_t`` is a known control (predict: d_hat -= u) and the source
diffusion ``v_t ~ N(0, sigma_source^2 I)`` is the process noise (predict: P += Q). Linear
dynamics + Gaussian noise, so the predict step is exact -- just like the update step.
Because the target keeps moving, the belief no longer collapses to zero; it settles at a
steady-state floor set by sigma_source against the sensing noise.

Sensing (mechanistic, as a cell would). Field is a quadratic bump ``c(x) = peak -
0.5*kappa*|s - x|^2`` -- kappa is set (see ``kappa()``) so the field peaks at ``peak``
at the source and stays >= 0 across the whole arena. Its gradient ``kappa*(s - p)`` is
still linear in s. Receptors evenly spaced on the membrane (radius r) each read the
local concentration with Gaussian noise; the cell's linear vector-sum readout, with the
``1/kappa`` scaling folded in,

    g_hat = (2 / (N r kappa)) sum_i u(phi_i) m(phi_i)  =  (s - p) + noise  =  d + noise

recovers the vector to the source (the common offset cancels since the unit vectors sum
to zero). So the measurement is ``y = g_hat = d + noise`` with H = I. Per-receptor noise
``sigma_receptor`` becomes ``effective_sigma = sigma_receptor*sqrt(2/N)/(r*kappa)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    arena: float = 200.0
    source: tuple[float, float] = (150.0, 150.0)   # TRUE source, unknown to the agent
    start: tuple[float, float] = (40.0, 40.0)
    speed: float = 1.0
    n_receptors: int = 8           # receptors evenly spaced around the membrane
    cell_radius: float = 5.0       # membrane radius (gradient baseline)
    peak: float = 20.0             # ligand concentration at the source (field runs 0..peak)
    sigma_receptor: float = 1.0    # receptor noise scale
    poisson: bool = True           # shot noise: per-receptor variance = sigma_receptor^2 * mean conc
    adapt_R: bool = False          # if True, R = the exact/oracle Poisson covariance (true conc)
    sigma_source: float = 0.5      # per-step SOURCE diffusion std (process noise); agent is exact
    sigma_actuator: float = 0.0    # per-step actuator noise on the agent's own move (it still
                                   # KNOWS where it landed, so this adds no process noise)
    prior_std: float = 100.0       # initial belief std over the source (broad)
    reach: float = 6.0             # "arrived" when within this distance of the source


def kappa(cfg: Config) -> float:
    """Field steepness: peaks at cfg.peak, touches 0 at the farthest arena corner (all >= 0)."""
    s = np.asarray(cfg.source, dtype=float)
    corners = np.array([[0, 0], [0, cfg.arena], [cfg.arena, 0], [cfg.arena, cfg.arena]], float)
    return 2.0 * cfg.peak / float(np.max(np.sum((corners - s) ** 2, axis=1)))


def effective_sigma(cfg: Config) -> float:
    """Std of the linear gradient readout in position units: sigma_receptor*sqrt(2/N)/(r*kappa)."""
    return cfg.sigma_receptor * np.sqrt(2.0 / cfg.n_receptors) / (cfg.cell_radius * kappa(cfg))


def R_true(conc, cfg: Config, dirs) -> np.ndarray:
    """Exact 2x2 measurement covariance under Poisson shot noise, from the TRUE per-receptor
    concentrations (an oracle -- the sim knows conc). Each receptor's variance is
    sigma_receptor^2*conc_i, and the linear readout maps it to
    (2/(N r kappa))^2 * sigma_receptor^2 * sum_i conc_i u_i u_i^T."""
    pref = (2.0 / (cfg.n_receptors * cfg.cell_radius * kappa(cfg))) ** 2 * cfg.sigma_receptor ** 2
    return pref * (dirs.T @ (np.maximum(conc, 0.0)[:, None] * dirs))


def _receptor_dirs(cfg: Config) -> np.ndarray:
    phi = np.arange(cfg.n_receptors) * (2.0 * np.pi / cfg.n_receptors)
    return np.stack([np.cos(phi), np.sin(phi)], axis=1)      # (N, 2) unit directions


def sense_gradient(p, s, cfg: Config, rng, dirs) -> np.ndarray:
    """Read every receptor, combine with the linear vector-sum readout -> ~ (s - p)."""
    kap = kappa(cfg)
    mem = p + cfg.cell_radius * dirs                         # (N, 2) receptor positions
    conc = cfg.peak - 0.5 * kap * np.sum((s - mem) ** 2, axis=1)   # (N,) ligand conc in [0, peak]
    if cfg.poisson:
        std = cfg.sigma_receptor * np.sqrt(np.maximum(conc, 0.0))  # shot noise: var = mean conc
    else:
        std = cfg.sigma_receptor                                   # constant Gaussian
    m = conc + rng.standard_normal(cfg.n_receptors) * std
    g = (2.0 / (cfg.n_receptors * cfg.cell_radius * kap)) * (dirs.T @ m)
    return g, conc                           # gradient readout, and TRUE per-receptor conc


class SourceKF:
    """Kalman filter over the vector-to-source ``d = s - p``. Measurement y = d + noise."""

    def __init__(self, cfg: Config):
        # prior: source somewhere near the arena centre -> d ~ centre - start
        self.m = np.array([cfg.arena / 2, cfg.arena / 2], dtype=float) - np.asarray(cfg.start, float)
        self.P = np.eye(2) * cfg.prior_std**2
        self.R = np.eye(2) * effective_sigma(cfg) ** 2
        self.Q = np.eye(2) * cfg.sigma_source**2

    def predict(self, u):
        self.m = self.m - u                                 # commanded move (known control)
        self.P = self.P + self.Q                            # movement error (process noise)

    def update(self, y):
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)
        self.m = self.m + K @ (y - self.m)
        self.P = (np.eye(2) - K) @ self.P


@dataclass
class Result:
    path: np.ndarray         # (n, 2) agent positions
    est: np.ndarray          # (n, 2) belief-mean source estimates (p + d_hat)
    snaps: list              # [(step, source_estimate, cov), ...] belief over the source
    source_path: np.ndarray  # (n, 2) MOVING source positions
    arrived: bool

    @property
    def n_steps(self) -> int:
        return len(self.path) - 1

    @property
    def final_error(self) -> float:
        n = min(len(self.est), len(self.source_path))
        return float(np.hypot(*(self.est[n - 1] - self.source_path[n - 1])))


def run(cfg: Config | None = None, seed: int = 0, max_steps: int = 300,
        sense: bool = True) -> Result:
    """Seek the source. ``sense=False`` is the blind control (belief never corrected)."""
    cfg = cfg or Config()
    rng = np.random.default_rng(seed)
    p = np.array(cfg.start, dtype=float)
    s = np.array(cfg.source, dtype=float)
    kf = SourceKF(cfg)
    dirs = _receptor_dirs(cfg)

    if sense:
        g, conc = sense_gradient(p, s, cfg, rng, dirs)
        if cfg.adapt_R:
            kf.R = R_true(conc, cfg, dirs)
        kf.update(g)
    path, est = [p.copy()], [p + kf.m]
    snaps, spath = [(0, p + kf.m, kf.P.copy())], [s.copy()]
    arrived = False
    for _ in range(max_steps):
        if float(np.hypot(*(s - p))) <= cfg.reach:
            arrived = True
            break
        nrm = float(np.hypot(*kf.m))                        # head toward the belief (direction of d)
        if nrm < 1e-9:
            break
        u = cfg.speed * kf.m / nrm
        if cfg.sigma_actuator > 0.0:                        # noisy actuation, but the agent
            u = u + rng.normal(0.0, cfg.sigma_actuator, 2)  # observes its realized displacement
        kf.predict(u)                                       # predict with the REALIZED move; Q stays source-only
        p = np.clip(p + u, 0.0, cfg.arena)
        s = np.clip(s + rng.normal(0.0, cfg.sigma_source, 2), 0.0, cfg.arena)  # source diffuses
        if sense:
            g, conc = sense_gradient(p, s, cfg, rng, dirs)
            if cfg.adapt_R:
                kf.R = R_true(conc, cfg, dirs)
            kf.update(g)
        path.append(p.copy())
        est.append(p + kf.m)
        snaps.append((len(path) - 1, p + kf.m, kf.P.copy()))
        spath.append(s.copy())

    return Result(np.array(path), np.array(est), snaps, np.array(spath), arrived)
