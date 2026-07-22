"""Turning an activity profile into a movement direction.

Port of ``legacy/src/nextpos.m``.
"""

from __future__ import annotations

import numpy as np

from .params import membrane_angles


def next_position(
    cellp: np.ndarray,
    activity: np.ndarray,
    stepsz: float,
    method: str,
    rng: np.random.Generator,
    direction_noise: float = 0.1,
) -> np.ndarray:
    """One movement step. Speed is fixed; only direction is decoded.

    optimal_noise : circular mean (first Fourier mode) of the activity profile --
        the ML bearing for a von Mises-shaped signal -- plus Gaussian orientation
        noise (std ``direction_noise`` rad) standing in for downstream
        cytoskeletal imprecision.
    perfect : argmax of the antipodal difference, i.e. largest front-minus-back
        contrast. Immune to a uniform baseline, and noiseless. Requires even N.
    randomwalk : uniformly random direction (chance-level control).
    """
    m = activity.size
    if method == "perfect" and m % 2 != 0:
        raise ValueError("number of membrane bins should be even for 'perfect'")

    phi = membrane_angles(m)

    if method == "optimal_noise":
        z1 = np.sum(np.cos(phi) * activity)
        z2 = np.sum(np.sin(phi) * activity)
        angle = np.arctan2(z2, z1) + rng.standard_normal() * direction_noise
    elif method == "perfect":
        idx = int(np.argmax(activity - np.roll(activity, m // 2)))
        angle = phi[idx]
    elif method == "randomwalk":
        angle = phi[rng.integers(0, m)]
    else:
        raise ValueError(
            f"unknown decoder_method {method!r}; "
            "expected optimal_noise, perfect or randomwalk"
        )

    return cellp + stepsz * np.array([np.cos(angle), np.sin(angle)])
