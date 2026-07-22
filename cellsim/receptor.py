"""Receptor sensing and the Crank-Nicolson membrane operator.

Ports ``legacy/src/hillfun.m``, ``legacy/src/receptor_output.m`` and ``legacy/src/CNMatrix.m``.
"""

from __future__ import annotations

import numpy as np

from .params import Params


def hill(counts: np.ndarray, p: Params) -> np.ndarray:
    """Occupancy x rtot. Port of ``legacy/src/hillfun.m`` (Hill coefficient 1).

    The MATLAB has a commented-out basal-activity term using ``receptornoise``;
    it is disabled there and omitted here.
    """
    return counts / (counts + p.kd) * p.rtot


def receptor_activity(
    env: np.ndarray,
    rvec: np.ndarray,
    p: Params,
    rng: np.random.Generator | None = None,
    avg_output: bool = True,
) -> np.ndarray:
    """Per-bin receptor activity. Port of ``legacy/src/receptor_output.m``.

    Deterministic branch: ``a_i = r_i * c_i/(c_i + kd)``.

    Noisy branch: two stacked Poisson layers -- ligand counting noise, then
    receptor activation noise -- averaged over ``nsamp`` samples. The second
    layer is collapsed: a sum of independent Poissons is Poisson of the summed
    rate, so drawing nsamp rows and averaging is distributionally identical to
    one row drawn at the summed rate (verified by KS test, p=0.99). The first
    layer cannot be collapsed because ``hill`` is nonlinear.
    """
    if not p.noisy:
        return rvec * hill(env, p) / p.rtot

    if rng is None:
        raise ValueError("a Generator is required when params.noisy is True")

    c_samp = rng.poisson(np.broadcast_to(env, (p.nsamp, env.size)))
    # rvec is the receptor profile f. Crank-Nicolson is not positivity-preserving,
    # so at very low signal (f small and spiky) a bin can undershoot slightly
    # negative; the physical activation rate cannot be, so clip. This is a no-op
    # wherever f stays positive (all validated regimes); it only guards the
    # low-ligand degenerate case, where MATLAB would instead poisson-NaN silently.
    rate = np.maximum(rvec * hill(c_samp, p) / p.rtot, 0.0)
    if avg_output:
        return rng.poisson(rate.sum(axis=0)) / p.nsamp
    return rng.poisson(rate)


def cn_matrices(p: Params, d: np.ndarray | float | None = None):
    """Crank-Nicolson matrices for the membrane. Port of ``legacy/src/CNMatrix.m``.

    Solves ``L w(t+1) = R w(t)`` with ``w = [f_1..f_N, F_cyto]``.

    ``d`` may be a scalar (uniform diffusivity) or a length-N vector. The vector
    case discretises the flux-conservative form d/dx( d(x) df/dx ) with d
    evaluated at bin FACES, so the column sums of both blocks telescope to 1/dt
    on the periodic ring and sum(f) is conserved exactly -- the property the
    augmented row relies on. For scalar d this reduces to the original
    constant-coefficient stencil.
    """
    N = p.N
    d = p.d if d is None else d
    d = np.broadcast_to(np.asarray(d, dtype=float), (N,))

    # diffusivity at faces: dface[i] sits between bin i and bin i+1 (periodic)
    dface = (d + np.roll(d, -1)) / 2.0
    ap = dface / (2 * p.dx**2)               # coupling from bin i to i+1
    am = np.roll(dface, 1) / (2 * p.dx**2)   # coupling from bin i to i-1

    DL = (np.diag(1 / p.dt + (am + ap))
          - np.diag(am[1:], -1) - np.diag(ap[:-1], 1))
    DL[0, -1] = -am[0]
    DL[-1, 0] = -ap[-1]

    DR = (np.diag(1 / p.dt - (am + ap))
          + np.diag(am[1:], -1) + np.diag(ap[:-1], 1))
    DR[0, -1] = am[0]
    DR[-1, 0] = ap[-1]

    L = np.zeros((N + 1, N + 1))
    R = np.zeros((N + 1, N + 1))
    L[:N, :N] = DL
    R[:N, :N] = DR
    # augmented row: an algebraic constraint enforcing sum(f) + F_cyto conservation
    L[N, :] = 1.0
    R[N, :] = 1.0
    return L, R


def coupled_diffusivity(env: np.ndarray, p: Params) -> np.ndarray:
    """Signal-coupled per-bin diffusivity.

        z_i = env_i / (env_i + kd)          Hill occupancy, in [0, 1)
        d_i = d0 / (1 + (z_i/z0)^n)         monotonically decreasing in z

    Two forms (``p.couple_form``):
      "hill"    : d0 / (1 + (z/z0)^n),  z = occupancy = c/(c+kd)
      "invsqrt" : d0 / sqrt(eps + c),   c = raw per-bin ligand count -- the direct
                  analogue of the EKF's sigma_Q = sigma_base / sqrt(eps + z).

    Optionally floored at ``p.dmin`` so the belief retains enough mobility to
    track a bearing that rotates as the cell advances -- the analogue of the
    clip on sigma_Q in the Python coupled EKF.
    """
    if p.couple_form == "invsqrt":
        d = p.d / np.sqrt(p.couple_eps + env)
    else:  # "hill"
        z = env / (env + p.kd)
        d = p.d / (1.0 + (z / p.z0) ** p.dn)
    if p.dmin is not None:
        d = np.maximum(d, p.dmin)
    return d
