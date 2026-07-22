"""Ensemble statistics vs MATLAB. SLOW (~3 min).

Trajectories cannot be compared: the interpolator differs (SciPy has no
natural-neighbour equivalent of MATLAB's scatteredInterpolant), RNG streams
differ, and the system diverges chaotically from perturbations at machine
precision. So the bar here is distributional -- success rate over an ensemble.

MATLAB reference values were measured on 2026-07-19 with 5 reps x 100 cells on
tissue_point_noflow. Two independent runs are quoted per regime; the port is
accepted if it lands within 3 combined SE of either.

Requires a __main__ guard when run standalone: cellsim.race uses multiprocessing
and macOS spawns rather than forks.
"""

from __future__ import annotations

import numpy as np

from cellsim import Params, race

SLOW = True          # for pytest: collected via -k / -m if you add a marker
NREP = 5
ENV = "tissue_point_noflow"

# (label, param overrides, [(matlab mean, matlab SE, source run), ...])
REGIMES = [
    ("deterministic, uncoupled",
     dict(noisy=False, dcouple=False),
     [(83.4, 0.2, "coupling_sweep"), (82.2, 0.6, "floor_sweep")]),
    ("poisson nsamp=30, uncoupled",
     dict(noisy=True, dcouple=False),
     [(82.8, 0.7, "floor_sweep"), (82.6, 0.9, "d0_sweep")]),
    ("poisson nsamp=30, coupled z0=0.2 n=2",
     dict(noisy=True, dcouple=True, z0=0.2, dn=2.0),
     [(85.2, 0.6, "floor_sweep"), (85.4, 0.9, "d0_sweep")]),
]


def _measure(overrides, nrep=NREP):
    p = Params(N=100, mean_cell_radius=6.0, T=4 * 3600.0, rtot=2000.0,
               kd_nM=10.0, d=0.01, nsamp=30, **overrides)
    rates = [race(ENV, p, source="point", receptor="feedback",
                  decoder_method="weighted_mean", seed=100 + r).success_rate
             for r in range(nrep)]
    a = np.array(rates)
    return a.mean(), a.std(ddof=1) / np.sqrt(nrep)


def _check_regime(label, overrides, refs):
    m, se = _measure(overrides)
    ok = any(abs(m - rm) <= 3 * np.hypot(se, rse) for rm, rse, _ in refs)
    detail = ", ".join(f"{rm:.1f}+-{rse:.1f} ({src})" for rm, rse, src in refs)
    assert ok, f"{label}: python {m:.1f}+-{se:.1f} disagrees with MATLAB {detail}"
    return m, se


def test_deterministic_uncoupled_matches_matlab():
    _check_regime(*REGIMES[0])


def test_poisson_uncoupled_matches_matlab():
    _check_regime(*REGIMES[1])


def test_poisson_coupled_matches_matlab():
    _check_regime(*REGIMES[2])


def test_coupling_reproduces_the_benefit():
    """The port must reproduce the *finding*, not just the baseline.

    MATLAB measured coupling worth about +2.5 points under Poisson sensing
    (+2.4, p=0.026 in floor_sweep; +2.8, p=0.059 in d0_sweep). The effect is
    small and n=5 is thin, so this only asserts the sign and rough size.
    """
    base, base_se = _measure(dict(noisy=True, dcouple=False))
    coup, coup_se = _measure(dict(noisy=True, dcouple=True, z0=0.2, dn=2.0))
    delta = coup - base
    assert delta > 0, f"coupling should help under noise, got {delta:+.1f} pts"
    assert delta < 10, f"implausibly large coupling effect: {delta:+.1f} pts"


def test_reproducible_across_runs():
    """Same seed must give the same answer -- MATLAB's parfor cannot do this."""
    p = Params(N=100, mean_cell_radius=6.0, T=1800.0, rtot=2000.0,
               kd_nM=10.0, d=0.01, noisy=False)
    a = race(ENV, p, source="point", seed=7)
    b = race(ENV, p, source="point", seed=7)
    assert a.success_rate == b.success_rate
    np.testing.assert_array_equal(
        np.nan_to_num(a.arrival_min, nan=-1.0),
        np.nan_to_num(b.arrival_min, nan=-1.0),
    )


if __name__ == "__main__":
    import sys
    import time

    from cellsim.tests import run_module

    print("ensemble statistics vs MATLAB (slow)")
    t0 = time.perf_counter()
    code = run_module(sys.modules[__name__])
    print(f"elapsed {time.perf_counter() - t0:.0f}s")
    raise SystemExit(code)
