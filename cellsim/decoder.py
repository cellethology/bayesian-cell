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
    direction_noise: float = 0.0,
) -> np.ndarray:
    """One movement step. Speed is fixed.

    A decoder reads a bearing from the activity profile; then ``direction_noise``
    rad of Gaussian steering noise is added -- for *any* decoder -- standing in for
    downstream cytoskeletal imprecision. The two steps are independent: choose a
    decoder and a noise level separately.

    Decoders (activity -> bearing):
      weighted_mean : activity-weighted circular mean over all bins (the ML bearing
          for a von Mises-shaped signal). Continuous, spatially averaged.
      steepest : the bin with the largest front-minus-back contrast (argmax of the
          antipodal difference). Winner-take-all; immune to a uniform baseline.
          Requires even N.
      random : a uniformly random bin -- ignores the signal (chance-level control).
    """
    m = activity.size
    if method == "steepest" and m % 2 != 0:
        raise ValueError("number of membrane bins should be even for 'steepest'")

    phi = membrane_angles(m)

    if method == "weighted_mean":
        z1 = np.sum(np.cos(phi) * activity)
        z2 = np.sum(np.sin(phi) * activity)
        angle = np.arctan2(z2, z1)
    elif method == "steepest":
        idx = int(np.argmax(activity - np.roll(activity, m // 2)))
        angle = phi[idx]
    elif method == "random":
        angle = phi[rng.integers(0, m)]
    else:
        raise ValueError(
            f"unknown decoder {method!r}; expected weighted_mean, steepest or random"
        )

    # steering noise, applied after any decoder (0 = noiseless)
    if direction_noise:
        angle = angle + rng.standard_normal() * direction_noise

    return cellp + stepsz * np.array([np.cos(angle), np.sin(angle)])
