"""Invariants of the model that hold regardless of any MATLAB reference.

These guard the properties most likely to be broken silently by future edits --
in particular receptor conservation, which the augmented row of the
Crank-Nicolson operator provides and which a careless variable-coefficient
discretisation would destroy without any visible error.
"""

from __future__ import annotations

import numpy as np

from cellsim import (LigandField, Params, cell_boundary, cn_matrices,
                     coupled_diffusivity, membrane_angles, simulate_cell)

P = Params(N=100, mean_cell_radius=6.0, rtot=2000.0, kd_nM=10.0,
           dt=1.0, d=0.01, koff=0.0678, h=0.002, noisy=False)


def test_vector_d_conserves_receptors():
    """Column sums of both blocks must telescope to 1/dt on the periodic ring.

    This is what makes sum(f) conserved for a spatially varying diffusivity.
    """
    rng = np.random.default_rng(7)
    d = P.d * (0.1 + rng.random(P.N))          # strongly heterogeneous
    L, R = cn_matrices(P, d)
    assert np.allclose(L[:-1, :-1].sum(axis=0), 1 / P.dt, atol=1e-12)
    assert np.allclose(R[:-1, :-1].sum(axis=0), 1 / P.dt, atol=1e-12)


def test_diffusion_does_not_leak_mass():
    """Integrate pure diffusion and check the total is preserved."""
    rng = np.random.default_rng(7)
    d = P.d * (0.1 + rng.random(P.N))
    L, R = cn_matrices(P, d)
    w = np.concatenate([np.where(np.arange(P.N) // 20 == 2, 1000.0, 0.0), [500.0]])
    total0 = w.sum()
    C = np.linalg.solve(L, R)
    for _ in range(500):
        w = C @ w
    assert abs(w.sum() - total0) / total0 < 1e-12, f"drift {w.sum()-total0:.3e}"


def test_scalar_d_matches_uniform_vector_d():
    """A scalar diffusivity and a constant vector must give identical operators."""
    Ls, Rs = cn_matrices(P)
    Lv, Rv = cn_matrices(P, np.full(P.N, P.d))
    assert np.array_equal(Ls, Lv)
    assert np.array_equal(Rs, Rv)


def test_coupled_diffusivity_is_monotonically_decreasing():
    """The whole point of the coupling: more signal -> less lateral mobility."""
    p = P.replace(dcouple=True, z0=0.5, dn=2.0)
    env = np.linspace(0.0, 500.0, 400)          # increasing ligand
    d = coupled_diffusivity(env, p)
    assert np.all(np.diff(d) < 0), "d(z) must strictly decrease in signal"
    assert d[0] <= p.d + 1e-15, "d must never exceed the baseline"
    assert np.all(d > 0), "d must stay positive"


def test_diffusivity_floor_is_respected():
    p = P.replace(dcouple=True, z0=0.05, dn=4.0, dmin=0.25 * P.d)
    env = np.linspace(0.0, 5000.0, 200)
    d = coupled_diffusivity(env, p)
    assert d.min() >= 0.25 * P.d - 1e-15, "dmin floor violated"


def test_membrane_angles_span_the_circle():
    phi = membrane_angles(100)
    assert phi.size == 100
    assert phi[0] == 0.0
    assert np.isclose(np.abs(np.sum(np.exp(1j * phi))), 0.0, atol=1e-10), \
        "bin angles should be uniformly distributed around the circle"


def test_uniform_scheme_holds_receptors_fixed():
    """The 'uniform' control must never repolarise -- it is the null model."""
    field = LigandField("tissue_point_noflow", P.conversion_factor, "point")
    p = P.replace(T=300.0)
    res = simulate_cell(p, field, np.array([90.0, 0.0]), receptor="uniform",
                        mode="localization", decoder_method="steepest",
                        rng=np.random.default_rng(0))
    n = res.n_used
    assert np.allclose(res.f[:n], res.f[0]), "uniform receptors drifted"


def test_feedback_conserves_total_receptors_in_a_real_run():
    """sum(f) + F_cyto must stay at rtot for the whole trajectory."""
    field = LigandField("tissue_point_noflow", P.conversion_factor, "point")
    p = P.replace(T=600.0)
    res = simulate_cell(p, field, np.array([90.0, 0.0]), receptor="feedback",
                        mode="localization", decoder_method="steepest",
                        rng=np.random.default_rng(0))
    n = res.n_used
    total = res.f[:n].sum(axis=1) + res.FC[:n]
    assert np.allclose(total, p.rtot, rtol=1e-10), \
        f"receptor pool drifted: {total.min():.6f}..{total.max():.6f} vs {p.rtot}"


def test_cell_boundary_is_a_circle_of_the_right_radius():
    b = cell_boundary(P)
    assert b.shape == (P.N, 2)
    assert np.allclose(np.hypot(b[:, 0], b[:, 1]), P.mean_cell_radius)


if __name__ == "__main__":
    import sys

    from cellsim.tests import run_module

    print("model invariants")
    raise SystemExit(run_module(sys.modules[__name__]))
