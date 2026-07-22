"""Regression tests for the MATLAB -> Python port.

Two levels, because only one of them can demand exactness:

* **Unit parity** (``test_matlab_parity.py``) -- deterministic building blocks
  checked against values exported from MATLAB. No RNG, no interpolation, so
  these agree to floating point and most are bit-identical. A failure here is a
  real porting bug.
* **Ensemble statistics** (``test_ensemble.py``) -- success rates compared with
  MATLAB runs. Trajectories *cannot* match: the interpolator differs (SciPy has
  no natural-neighbour), RNG streams differ, and the system diverges chaotically
  from perturbations at machine precision. Only distributions are comparable.

Property tests (``test_properties.py``) need no MATLAB reference at all.

Run with pytest if available::

    pytest cellsim/tests -q
    pytest cellsim/tests -q -k "not slow"

or standalone, with no pytest dependency::

    python -m cellsim.tests.test_matlab_parity
    python -m cellsim.tests.test_properties
    python -m cellsim.tests.test_ensemble      # slow, ~3 min

To regenerate the reference data after changing the MATLAB, run
``data/export_unit_ref.m`` in MATLAB and copy ``unit_reference.mat`` here.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import scipy.io as sio

DATA = Path(__file__).parent / "data"


def load_reference() -> dict:
    """MATLAB reference values for the deterministic building blocks."""
    path = DATA / "unit_reference.mat"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- regenerate it by running {DATA/'export_unit_ref.m'} in MATLAB"
        )
    return sio.loadmat(path)


def assert_close(got, want, tol=1e-12, name="value"):
    """Compare against a MATLAB reference on max relative error."""
    got = np.asarray(got, dtype=float).ravel()
    want = np.asarray(want, dtype=float).ravel()
    assert got.shape == want.shape, f"{name}: shape {got.shape} != {want.shape}"
    err = np.max(np.abs(got - want))
    scale = max(np.max(np.abs(want)), 1e-30)
    assert err / scale <= tol, f"{name}: max|err| {err:.3e}, rel {err/scale:.3e} > {tol:.1e}"
    return err / scale


def run_module(module) -> int:
    """Run every ``test_*`` in a module without pytest. Returns an exit code."""
    tests = sorted(
        (n, f) for n, f in vars(module).items()
        if n.startswith("test_") and callable(f)
    )
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as exc:  # noqa: BLE001 -- this is a test runner
            failed.append(name)
            print(f"  FAIL  {name}: {exc}")
            traceback.print_exc()
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0
