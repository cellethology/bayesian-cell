"""Simulation parameters.

Python port of the ``param`` struct threaded through ``legacy/src/*.m``. Defaults match
``legacy/src/default_param.mat`` as overridden by ``legacy/run.m``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


def conc2count(cell_radius: float, nbin: int) -> float:
    """nM -> molecules per membrane bin. Port of ``legacy/src/conc2count.m``.

    Sensing volume per bin is arclength x 1.5um x 1.5um; 1 nM = 0.602 molecules/um^3.
    (The MATLAB comments disagree about cell height -- line 3 says 3um, line 6 says
    1.5um. The code implements 1.5^2, which is reproduced here.)
    """
    bin_volume = (2 * np.pi * cell_radius / nbin) * 1.5**2
    return bin_volume * 0.602


def ellipse_perimeter(a: float, b: float) -> float:
    """Ramanujan's second approximation. Port of ``legacy/src/ellipsePerimeter.m``."""
    h = (a - b) ** 2 / (a + b) ** 2
    return np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))


@dataclass(frozen=True)
class Params:
    """Immutable parameter set for one simulation.

    Frozen so that per-timestep modifications (notably the coupled diffusivity)
    cannot leak across steps -- this reproduces MATLAB's pass-by-value semantics,
    which the original code relies on.
    """

    # --- cell geometry ---
    N: int = 100
    mean_cell_radius: float = 6.0
    stepsz: float = 1.0

    # --- movement ---
    direction_noise: float = 0.1  # std (rad) added to the decoded bearing, nextpos.m:18

    # --- time ---
    dt: float = 1.0
    T: float = 4 * 60 * 60.0

    # --- receptor kinetics ---
    d: float = 0.01          # membrane diffusivity, um^2/s
    h: float = 0.002         # feedback gain, 1/s
    koff: float = 0.0678     # endocytosis rate, 1/s
    rtot: float = 2000.0     # total receptor pool

    # --- sensing ---
    kd_nM: float = 10.0      # half-saturation, in nM (converted to counts below)
    noisy: bool = False
    nsamp: int = 30
    receptornoise: float = 0.1
    conc_scale: float = 1.0  # multiplies the whole ligand field; <1 = weaker,
                             # noisier signal (fewer molecules -> more shot noise)

    # --- signal-coupled diffusivity (optional) ---
    dcouple: bool = False
    couple_form: str = "hill"  # "hill": d0/(1+(z/z0)^n) on occupancy z;
                               # "invsqrt": d0/sqrt(eps + count) on raw count
                               # (direct analog of the EKF sigma_Q = base/sqrt(eps+z))
    z0: float = 0.2          # occupancy midpoint (hill form)
    dn: float = 2.0          # Hill exponent (hill form)
    couple_eps: float = 1.0  # offset for invsqrt form
    dmin: float | None = None  # floor on diffusivity

    # --- runtime ---
    seed: int | None = None

    # ------------------------------------------------------------------
    @property
    def conversion_factor(self) -> float:
        return conc2count(self.mean_cell_radius, self.N)

    @property
    def kd(self) -> float:
        """Half-saturation in molecule counts per bin."""
        return self.kd_nM * self.conversion_factor

    @property
    def dx(self) -> float:
        """Arclength per membrane bin (circular cell)."""
        return ellipse_perimeter(self.mean_cell_radius, self.mean_cell_radius) / self.N

    @property
    def n_steps(self) -> int:
        return int(np.floor(self.T / self.dt))

    @property
    def move_rate(self) -> int:
        """Cell moves every 30 s; receptors update every dt."""
        return int(np.floor(30.0 / self.dt))

    def replace(self, **kw) -> "Params":
        return replace(self, **kw)


def membrane_angles(N: int) -> np.ndarray:
    """Bin angles phi, matching the construction in ``legacy/src/nextpos.m`` exactly.

        phi = linspace(pi, -pi+2pi/m, m)';
        phi = circshift(flipud(phi), m/2+1);
        phi(1) = 0;
    """
    phi = np.linspace(np.pi, -np.pi + 2 * np.pi / N, N)
    phi = np.roll(np.flipud(phi), N // 2 + 1)
    phi[0] = 0.0  # correct minor numerical inaccuracy, as in the MATLAB
    return phi
