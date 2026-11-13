# Development guide

## Components and boundaries

| Location | Purpose | Put new work here when… |
|---|---|---|
| `cellsim/` | Cell-navigation model and ensemble simulator | changing receptor dynamics, sensing, movement, or arrival statistics |
| `cellsim/cli.py` | Simulator command-line interface | exposing a model parameter or changing run reporting |
| `cellsim/tests/` | Model invariants and MATLAB regression checks | validating model behavior or parity |
| `source_kf/` | Source-localization demonstration | studying receptor readouts and Kalman belief updates in this demo |
| `robotic/core/` | Robotics environments and configuration | changing the simulated world or study defaults |
| `robotic/filters/` | Filter implementations | adding or changing a Kalman filter |
| `robotic/analysis/` | Comparisons, sweeps, and result plots | comparing filters or analyzing runs |
| `robotic/histogram_filter/` | Grid-based Bayesian navigation | working on histogram-filter experiments |
| `robotic/shared/` | Utilities used across robotics components | sharing signal models, statistics, or file handling |
| `tissue_sim/` | Tissue-field inputs and their MATLAB generators | generating or adding ligand fields |
| `legacy/` (local, Git-ignored) | Original MATLAB navigation implementation | regenerating MATLAB parity fixtures |
| `LEGI/` | Tracked MATLAB reference models | inspecting or maintaining LEGI models |

Keep experiment orchestration separate from reusable model functions. Use relative
imports within each Python package, and qualified imports such as
`from robotic.core import EKFEnvironment` across packages or in external scripts.
Run Python modules from the repository root; no manual `PYTHONPATH` is needed.
The direct-script guards in older robotics entry points preserve those commands.

## Environment

Use Python 3.10 or newer. From the repository root:

```bash
uv sync --locked
uv run python -m cellsim --help
uv run python -m source_kf --help
```

For robotics:

```bash
uv sync --locked --extra robotic
uv run --extra robotic python -m robotic --help
```

All three Python packages belong to this project. Robotics dependencies are an
optional extra; importing or running `cellsim` does not require them. For pip,
use `python -m pip install -e .` or `python -m pip install -e '.[robotic]'`.

A copied virtual environment is machine-specific. If its Python executable no
longer works, recreate it on the current machine with `uv sync`; keep research
files outside `.venv/`.

## Inputs, fixtures, and outputs

- Keep `.mat` tissue inputs in `tissue_sim/`. `cellsim.environment` resolves them
  relative to the checkout, including `point_source/` and `koff_variants/`.
- Keep MATLAB regression fixtures in `cellsim/tests/data/` and retain their
  generator `export_unit_ref.m` for provenance. Regenerating the fixtures requires
  MATLAB and a local copy of `legacy/src/`; `legacy/` is ignored by Git and is not
  distributed in new checkouts. Existing tests use the committed fixtures.
- Use root `figures/` for scratch figures and root `output/` for scratch run
  results. These are ignored by Git. Relative output paths use the working
  directory; run documented commands from the repository root.
- Existing `robotic/output/` files are research results. Preserve their names and
  paths when reproducing existing figures. Deliberately add reproducible new
  reference results there with their generating command, seed, and parameters.

The wheel contains the Python packages; the repository holds the tissue inputs
and MATLAB source. Use an editable install in this checkout for tissue-backed
simulations. The analytic `radial_exp` environment needs no tissue files.

## Validation

Fast model checks need no pytest dependency:

```bash
uv run python -m cellsim.tests.test_matlab_parity
uv run python -m cellsim.tests.test_properties
```

Check command-line wiring with a short, single-process run:

```bash
uv run python -m cellsim --env radial_exp --hours 0.001 --cells 2 --processes 1
uv run python -m source_kf --max-steps 2 --save ''
uv run --extra robotic python -m robotic --max-steps 2 --no-show-plots --quiet
```

For changes to scientific behavior, also run the slower statistical regression:

```bash
uv run python -m cellsim.tests.test_ensemble
```

A reorganization should preserve numerical behavior, field lookup, random seeds,
and multiprocessing. When changing import paths, check worker execution as well
as the main process. See [the robotics guide](../robotic/README.md) for its studies.
