"""Deterministic building blocks vs values exported from MATLAB.

No RNG and no interpolation here, so these must agree to floating point.
A failure is a genuine porting bug, not chaotic divergence.
"""

from __future__ import annotations

import numpy as np

from cellsim import (Params, cn_matrices, conc2count, ellipse_perimeter, hill,
                     membrane_angles, next_position, receptor_activity)
from cellsim.tests import assert_close, load_reference

REF = load_reference()

# the parameter set used when the reference was exported (see data/export_unit_ref.m)
P = Params(N=100, mean_cell_radius=6.0, rtot=2000.0, kd_nM=10.0,
           dt=1.0, d=0.01, koff=0.0678, h=0.002, noisy=False)
ENV = REF["env"].ravel()
RVEC = REF["rvec"].ravel()


def test_conc2count():
    assert_close(conc2count(6, 100), REF["cf_6_100"], 0.0, "conc2count(6,100)")
    assert_close(conc2count(10, 50), REF["cf_10_50"], 0.0, "conc2count(10,50)")


def test_ellipse_perimeter():
    assert_close(ellipse_perimeter(6, 6), REF["ellipse_6_6"], 0.0, "ellipse(6,6)")
    assert_close(ellipse_perimeter(2, 5), REF["ellipse_2_5"], 0.0, "ellipse(2,5)")


def test_membrane_angles():
    # circshift/flipud construction from nextpos.m; exact up to one ULP
    assert_close(membrane_angles(100), REF["phi"], 1e-15, "phi")


def test_hillfun():
    assert_close(hill(ENV, P), REF["hill"], 0.0, "hillfun")


def test_receptor_activity_deterministic():
    assert_close(receptor_activity(ENV, RVEC, P), REF["act"], 0.0, "receptor_output")


def test_cn_matrices_scalar_are_bit_identical():
    """Scalar d must reproduce the original constant-coefficient stencil exactly.

    This is why (am + ap) is parenthesised in cn_matrices: for uniform d it then
    evaluates bit-identically to MATLAB's 1/dt + 2*alpha, keeping pre-existing
    results reproducible.
    """
    L, R = cn_matrices(P)
    assert_close(L, REF["L_scalar"], 0.0, "L scalar")
    assert_close(R, REF["R_scalar"], 0.0, "R scalar")


def test_cn_matrices_vector_are_bit_identical():
    L, R = cn_matrices(P, REF["dvec"].ravel())
    assert_close(L, REF["L_vector"], 0.0, "L vector")
    assert_close(R, REF["R_vector"], 0.0, "R vector")


def test_decoder_perfect():
    act = REF["act_test"].ravel()
    rng = np.random.default_rng(0)  # unused by the 'steepest' branch (noiseless)
    got = next_position(np.array([10.0, 5.0]), act, 1.0, "steepest", rng)
    assert_close(got, REF["next_perfect"], 0.0, "nextpos perfect")


def test_full_receptor_step():
    """One Crank-Nicolson update, matching update_receptor.m."""
    hprop = float(REF["hprop"].ravel()[0])
    assert_close(P.h / np.mean(receptor_activity(ENV, RVEC, P)), REF["hprop"], 1e-15, "hprop")

    a = receptor_activity(ENV, RVEC, P)
    L0, R0 = cn_matrices(P)
    N = P.N
    L, R = L0.copy(), R0.copy()
    d = np.arange(N)
    L[d, d] += P.koff / 2
    R[d, d] -= P.koff / 2
    L[:N, N] = -(hprop / 2) * a
    R[:N, N] = +(hprop / 2) * a
    v = np.concatenate([RVEC, [P.rtot - RVEC.sum()]])
    w = np.linalg.solve(L, R @ v)

    assert_close(w[:N], REF["step_f"], 1e-14, "step f")
    assert_close(w[N], REF["step_FC"], 1e-14, "step FC")
    assert_close(a, REF["step_a"], 0.0, "step a")
    assert_close(np.mean(hprop * a), REF["step_kfb"], 1e-15, "step kfb")


if __name__ == "__main__":
    import sys

    from cellsim.tests import run_module

    print("MATLAB parity (deterministic blocks)")
    raise SystemExit(run_module(sys.modules[__name__]))
