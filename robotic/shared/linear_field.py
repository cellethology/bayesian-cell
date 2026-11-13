"""
Pure-function physics for the LINEAR-GAUSSIAN variant of the tracking problem
(the source_kf setup, re-expressed in the robotic harness's coordinates).

Field is a quadratic bump centred on the target:

    c(x) = peak - 0.5 * kappa * ||target - x||^2

with kappa chosen (see field_kappa) so the field stays >= 0 anywhere in the arena.
Its gradient kappa*(target - x) is LINEAR in the target position.  N receptors evenly
spaced on a membrane of radius r each read the local concentration with noise; the
linear vector-sum readout

    g = (2 / (N r kappa)) * sum_i u_i m_i  =  (target - robot) + noise

recovers the vector to the target, so the filter sees H = I with a bounded gain.

This module deliberately has NO intra-package imports so that both `filters/` and
`core/` can import it without creating a cycle.
"""

import numpy as np


def field_kappa(config):
    """Field steepness.

    Explicit via config["linear_kappa"], else auto: kappa = peak / span**2, which is
    the source_kf rule (2*peak / max_corner_distance^2) evaluated at the worst case
    max_corner_distance^2 = 2*span^2.  Guarantees c(x) >= 0 everywhere in the arena
    for ANY target position, so it never leaks the true target position.
    """
    explicit = config.get("linear_kappa")
    if explicit:
        return float(explicit)
    span = float(config["arena_max"]) - float(config["arena_min"])
    peak = float(config["signal_max"])  # signal_max is reused as the peak concentration
    return peak / (span**2)


def receptor_dirs(config):
    """(N, 2) unit vectors of the receptor positions on the membrane."""
    n = int(config.get("linear_n_receptors", 8))
    phi = np.arange(n) * (2.0 * np.pi / n)
    return np.stack([np.cos(phi), np.sin(phi)], axis=1)


def readout_prefactor(config):
    """The 2/(N r kappa) scaling that turns the receptor sum into position units."""
    n = int(config.get("linear_n_receptors", 8))
    r = float(config.get("linear_cell_radius", 5.0))
    return 2.0 / (n * r * field_kappa(config))


def effective_sigma(config):
    """Std (position units, per axis) of the gradient readout with CONSTANT receptor
    noise: sigma_receptor * sqrt(2/N) / (r * kappa)."""
    n = int(config.get("linear_n_receptors", 8))
    r = float(config.get("linear_cell_radius", 5.0))
    s = float(config.get("linear_sigma_receptor", 1.0))
    return s * np.sqrt(2.0 / n) / (r * field_kappa(config))


def concentrations(robot_pos, target_pos, config, dirs):
    """True ligand concentration at each receptor (clipped at 0)."""
    kap = field_kappa(config)
    peak = float(config["signal_max"])
    r = float(config.get("linear_cell_radius", 5.0))
    mem = np.asarray(robot_pos, dtype=float) + r * dirs
    c = peak - 0.5 * kap * np.sum((np.asarray(target_pos, float) - mem) ** 2, axis=1)
    return np.maximum(c, 0.0)


def range_variance_factor(distance, config):
    """ABLATION knob (default off).  Multiplies the readout VARIANCE by
    exp(linear_range_decay * distance), so the sensor's position-space precision decays
    exponentially with range exactly as the EKF's does -- while the measurement stays
    full-rank, unbiased and bounded-gain.  This isolates 'information vanishes far away'
    from 'the Jacobian is rank-1' and from 'the gain lever arm is unbounded'.

    distance is clamped to the arena diagonal so an unbounded belief mean cannot make
    exp(...) overflow.
    """
    rate = float(config.get("linear_range_decay", 0.0) or 0.0)
    if rate <= 0.0:
        return 1.0
    span = float(config["arena_max"]) - float(config["arena_min"])
    d = min(float(distance), span * np.sqrt(2.0))
    return float(np.exp(rate * d))


def sense_gradient(robot_pos, target_pos, config, rng, dirs):
    """Read every receptor and combine -> (gradient readout, true conc, noisy readings)."""
    conc = concentrations(robot_pos, target_pos, config, dirs)
    s = float(config.get("linear_sigma_receptor", 1.0))
    if config.get("linear_shot_noise", True):
        std = s * np.sqrt(conc)  # shot noise: per-receptor variance = mean conc
    else:
        std = s * np.ones(len(conc))
    d = float(np.linalg.norm(np.asarray(target_pos, float) - np.asarray(robot_pos, float)))
    std = std * np.sqrt(range_variance_factor(d, config))
    m = conc + rng.standard_normal(len(conc)) * std
    g = readout_prefactor(config) * (dirs.T @ m)
    return g, conc, m


def R_from_conc(conc, config, dirs, distance=0.0):
    """Exact 2x2 covariance of the gradient readout given per-receptor concentrations."""
    s = float(config.get("linear_sigma_receptor", 1.0))
    pref = readout_prefactor(config) ** 2 * s**2 * range_variance_factor(distance, config)
    if config.get("linear_shot_noise", True):
        return pref * (dirs.T @ (np.asarray(conc, float)[:, None] * dirs))
    n = len(dirs)
    return pref * (n / 2.0) * np.eye(2)


def R_at(target_est, robot_pos, config, dirs):
    """Predicted 2x2 measurement covariance from an ESTIMATE of the target position.
    This is what a filter can legitimately compute (no oracle)."""
    d = float(np.linalg.norm(np.asarray(target_est, float) - np.asarray(robot_pos, float)))
    return R_from_conc(
        concentrations(robot_pos, target_est, config, dirs), config, dirs, distance=d
    )
