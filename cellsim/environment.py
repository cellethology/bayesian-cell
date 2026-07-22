"""Tissue ligand field.

Loads the steady-state fields produced by ``tissue_sim/`` and exposes a callable
that returns molecule counts per membrane bin at arbitrary positions.

Note on interpolation: ``legacy/src/racing_cells.m`` uses
``scatteredInterpolant(...,'natural','linear')``, but the underlying data is on a
regular grid (``pos`` is built from ``combvec(linspace, linspace)``). SciPy has no
natural-neighbour interpolator, so this uses ``RegularGridInterpolator`` with
linear interpolation -- the closest available match, and exact for gridded data
in the bilinear sense. Measured against the MATLAB reference on realistic
membrane rings, the decoded bearing differs by 0.1-0.95 degrees, well under the
5.7 degrees of decision noise ``nextpos`` already injects. Trajectories therefore
will not match MATLAB run-for-run; ensemble statistics should.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator

REPO_ROOT = Path(__file__).resolve().parent.parent
TISSUE_DIRS = [
    REPO_ROOT / "tissue_sim",
    REPO_ROOT / "tissue_sim" / "point_source",
    REPO_ROOT / "tissue_sim" / "koff_variants",
]


def _find_env_file(name: str) -> Path:
    stem = name if name.endswith(".mat") else f"{name}.mat"
    for d in TISSUE_DIRS:
        p = d / stem
        if p.exists():
            return p
    searched = "\n  ".join(str(d) for d in TISSUE_DIRS)
    raise FileNotFoundError(f"environment {stem!r} not found under:\n  {searched}")


class LigandField:
    """Static total-ligand field ``ctot = csol + cbound``, in molecule counts.

    Parameters
    ----------
    name : environment stem, e.g. "tissue_point_noflow" or "tissue_env"
    conversion_factor : nM -> counts/bin (see params.conc2count)
    source : "point" keeps absolute coordinates; "edge" shifts x,y to start at 1,
        reproducing the coordinate convention in ``racing_cells.m`` lines 58-71.
    conc_scale : multiplies the whole field; <1 lowers the ligand level, which in
        the shot-noise regime makes the signal noisier (fewer molecules per bin).
    """

    def __init__(self, name: str, conversion_factor: float, source: str = "point",
                 conc_scale: float = 1.0):
        path = _find_env_file(name)
        M = sio.loadmat(
            path,
            variable_names=["csol", "cbound", "xmin", "xmax", "ymin", "ymax"],
        )
        ctot = M["csol"] + M["cbound"]
        # scipy returns MATLAB scalars as (1,1) arrays; numpy 2.x refuses to
        # coerce those with float(), so unwrap explicitly
        xmin, xmax = M["xmin"].item(), M["xmax"].item()
        ymin, ymax = M["ymin"].item(), M["ymax"].item()

        if source == "point":
            gx = np.linspace(xmin, xmax, ctot.shape[0])
            gy = np.linspace(ymin, ymax, ctot.shape[1])
            self.bounds = np.array([[xmax, ymax], [xmin, ymin]])
        else:
            gx = np.linspace(1.0, xmax - xmin, ctot.shape[0])
            gy = np.linspace(1.0, ymax - ymin, ctot.shape[1])
            self.bounds = np.array(
                [[xmax - xmin - 5.0, ymax - ymin - 5.0], [5.0, 5.0]]
            )

        if np.min(ctot) < 0:
            raise ValueError(f"{path.name}: negative ligand concentration in field")

        self._interp = RegularGridInterpolator(
            (gx, gy), ctot * conversion_factor * conc_scale,
            method="linear", bounds_error=False, fill_value=None,
        )
        self.name = name
        self.source = source
        self.extent = (xmin, xmax, ymin, ymax)
        self.grid_shape = ctot.shape

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        """Sample the field. ``coords`` is (n, 2); returns (n,) molecule counts."""
        return self._interp(coords)

    def in_bounds(self, coords: np.ndarray) -> bool:
        """Port of ``legacy/src/inbound.m``: columnwise bounding-box test on the boundary."""
        return bool(
            np.all(coords.max(axis=0) < self.bounds[0])
            and np.all(coords.min(axis=0) > self.bounds[1])
        )


class RadialExpField:
    """Smooth analytic point source: ``c(r) = c0 * exp(-lambda * r)`` in nM.

    The clean, heterogeneity-free counterpart to LigandField, and the direct
    analogue of the robotics signal model (robotic/ uses c0*exp(-lambda*dist)).
    Same interface as LigandField so it drops into simulate_cell/race unchanged.

    Amplitude and decay (c0=24.5 nM, lambda=0.0315/um) are fitted to the radial
    profile of tissue_point_noflow, so a run on this field has the same *average*
    gradient as the tissue -- isolating spatial heterogeneity as the only
    difference. The source sits at the origin, coincident with the success-zone
    centre, so every cell starts exactly one start-radius away (the tissue's own
    peak is at (0,12), but this clean control centres it).
    """

    def __init__(self, conversion_factor: float, source: str = "point",
                 conc_scale: float = 1.0, c0_nM: float = 24.5111,
                 decay: float = 0.03149, source_pos=(0.0, 0.0),
                 bound: float = 120.0):
        self.c0 = c0_nM * conversion_factor * conc_scale
        self.decay = decay
        self.source_pos = np.asarray(source_pos, dtype=float)
        self.source = source
        self.bounds = np.array([[bound, bound], [-bound, -bound]])

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        r = np.hypot(coords[:, 0] - self.source_pos[0],
                     coords[:, 1] - self.source_pos[1])
        return self.c0 * np.exp(-self.decay * r)

    def in_bounds(self, coords: np.ndarray) -> bool:
        return bool(
            np.all(coords.max(axis=0) < self.bounds[0])
            and np.all(coords.min(axis=0) > self.bounds[1])
        )


def make_field(env_name: str, conversion_factor: float, source: str = "point",
               conc_scale: float = 1.0):
    """Build the ligand field for ``env_name``.

    ``"radial_exp"`` (optionally ``"radial_exp:c0=..,decay=.."``) gives the smooth
    analytic gradient; anything else is a tissue .mat loaded via LigandField.
    """
    if env_name == "radial_exp" or env_name.startswith("radial_exp:"):
        kw = {}
        if ":" in env_name:
            for tok in env_name.split(":", 1)[1].split(","):
                k, v = tok.split("=")
                kw[{"c0": "c0_nM", "decay": "decay"}.get(k, k)] = float(v)
        return RadialExpField(conversion_factor, source, conc_scale, **kw)
    return LigandField(env_name, conversion_factor, source, conc_scale)
