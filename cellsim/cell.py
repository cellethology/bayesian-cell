"""Single-cell simulation.

Port of ``legacy/src/cell_move.m`` and ``legacy/src/update_receptor.m``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decoder import next_position
from .environment import LigandField
from .params import Params, membrane_angles
from .receptor import cn_matrices, coupled_diffusivity, receptor_activity


@dataclass
class CellResult:
    """Trajectory and internal state of one cell."""

    cellp: np.ndarray        # (n_steps, 2) position; zeros after an early stop
    f: np.ndarray            # (n_steps, N) receptor profile
    env: np.ndarray          # (n_steps, N) sampled ligand counts
    a: np.ndarray            # (n_steps, N) receptor activity
    FC: np.ndarray | None    # (n_steps,) cytosolic pool (feedback only)
    kfb: np.ndarray | None   # (n_steps,) mean feedback rate (feedback only)
    stat: np.ndarray         # [<h*a>, koff, <r_memb>, std r_memb]
    arrived: bool
    n_used: int              # steps actually simulated

    def positions_at_moves(self, move_rate: int) -> np.ndarray:
        """Subsample as ``results.cellp(1:move_rate:time_size,:)`` does."""
        return self.cellp[::move_rate]


def cell_boundary(p: Params) -> np.ndarray:
    """Membrane sample points relative to the cell centre (circular cell)."""
    phi = membrane_angles(p.N)
    return p.mean_cell_radius * np.column_stack([np.cos(phi), np.sin(phi)])


def _record_stat(p: Params, kfb: np.ndarray, f: np.ndarray, i: int) -> np.ndarray:
    """Port of ``legacy/src/record_stat.m``. Note the MATLAB index ranges: kfb(1:it-1)
    and f(2:it-1,:), which map to kfb[:i] and f[1:i] here. MATLAB's std is
    normalised by n-1, hence ddof=1."""
    avg_kfb = float(np.mean(kfb[:i])) if i > 0 else 0.0
    rcount = f[1:i].sum(axis=1)
    if rcount.size == 0:
        return np.array([avg_kfb, p.koff, 0.0, 0.0])
    return np.array([avg_kfb, p.koff, float(rcount.mean()), float(rcount.std(ddof=1))])


def simulate_cell(
    p: Params,
    field: LigandField,
    start: np.ndarray,
    receptor: str = "feedback",
    mode: str = "localization",
    decoder_method: str = "weighted_mean",
    rng: np.random.Generator | None = None,
) -> CellResult:
    """Simulate one cell. Port of ``legacy/src/cell_move.m``.

    Receptors update every ``dt``; the cell moves every ``move_rate`` steps.
    The loop breaks on arrival, leaving trailing zeros in ``cellp`` -- the
    ensemble driver uses that to recover arrival time, as the MATLAB does.
    """
    if receptor not in ("feedback", "uniform"):
        raise ValueError(f"receptor must be 'feedback' or 'uniform', got {receptor!r}")
    rng = np.random.default_rng() if rng is None else rng

    N, n_steps = p.N, p.n_steps
    boundary = cell_boundary(p)
    cellp = np.asarray(start, dtype=float).copy()

    f = np.zeros((n_steps, N))
    pos = np.zeros((n_steps, 2))
    env_hist = np.zeros((n_steps, N))
    a = np.zeros((n_steps, N))

    # --- initial state ---
    pos[0] = cellp
    env = field(cellp + boundary)
    env_hist[0] = env
    f[0] = 0.9 * p.rtot / N

    FC = kfb = None
    base_L = base_R = None
    hprop = 0.0
    if receptor == "feedback":
        base_L, base_R = cn_matrices(p)
        FC = np.zeros(n_steps)
        kfb = np.zeros(n_steps)
        FC[0] = p.rtot - f[0].sum()
        # normalise the feedback gain so <k_fb> = h regardless of ligand level;
        # this is the model's perfect-adaptation mechanism
        mean_act0 = np.mean(receptor_activity(env, f[0], p, rng))
        # A cell that samples zero signal at init (possible at very low ligand --
        # ~0.4% of cells at 10x dilution, never at the concentrations used for
        # validation) has an undefined adaptive gain. Fall back to no feedback so
        # it simply fails to polarise, instead of hprop=inf poisoning the CN solve
        # with NaN. The MATLAB (cell_move.m:59) has the same latent divide-by-zero
        # but produces silent NaNs via poissrnd rather than erroring.
        hprop = p.h / mean_act0 if mean_act0 > 0 else 0.0
        kfb[0] = p.h

    a[0] = receptor_activity(env, f[0], p, rng)

    stat = np.zeros(4)
    arrived = False
    n_used = n_steps
    diag = np.arange(N)
    stop_radius = 25.0 + p.mean_cell_radius
    stop_x = 7.0 + p.mean_cell_radius

    for i in range(1, n_steps):
        # ---- receptor update ----
        if receptor == "feedback":
            if p.dcouple:
                L0, R0 = cn_matrices(p, coupled_diffusivity(env, p))
            else:
                L0, R0 = base_L, base_R

            a[i] = receptor_activity(env, f[i - 1], p, rng)

            L = L0.copy()
            R = R0.copy()
            L[diag, diag] += p.koff / 2.0
            R[diag, diag] -= p.koff / 2.0
            L[:N, N] = -(hprop / 2.0) * a[i]
            R[:N, N] = +(hprop / 2.0) * a[i - 1]

            v = np.concatenate([f[i - 1], [FC[i - 1]]])
            w = np.linalg.solve(L, R @ v)
            f[i] = w[:N]
            FC[i] = w[N]
            kfb[i] = np.mean(hprop * a[i])
        else:  # uniform: receptors pinned at their initial distribution
            f[i] = f[0]

        # ---- movement ----
        if mode != "static" and (i + 1) % p.move_rate == 1:
            act = receptor_activity(env, f[i], p, rng)
            cellp2 = next_position(cellp, act, p.stepsz, decoder_method, rng,
                                   direction_noise=p.direction_noise)
            coord2 = cellp2 + boundary
            env_try = field(coord2)

            if field.in_bounds(coord2):
                if not np.all(env_try >= 0):
                    raise ValueError("negative concentration sampled")
                cellp = cellp2
                env = env_try

            if mode == "localization":
                if field.source == "edge":
                    stopping = cellp[0] < stop_x
                else:
                    stopping = np.hypot(*cellp) < stop_radius
                if stopping:
                    pos[i] = cellp
                    env_hist[i] = env
                    if receptor == "feedback":
                        stat = _record_stat(p, kfb, f, i)
                    arrived = True
                    n_used = i + 1
                    break

        pos[i] = cellp
        env_hist[i] = env

        if i == n_steps - 1 and receptor == "feedback":
            stat = _record_stat(p, kfb, f, i)

    return CellResult(
        cellp=pos, f=f, env=env_hist, a=a, FC=FC, kfb=kfb,
        stat=stat, arrived=arrived, n_used=n_used,
    )
