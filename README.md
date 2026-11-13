# Cell Navigation Simulator

Simulates cells chemotaxing up a ligand gradient by Bayesian inference implemented
in receptor biochemistry: the receptor-density profile around the membrane **is** a
belief over the direction to the source, activity-driven recruitment is the
likelihood update, lateral diffusion is the motion model, and the finite receptor
pool enforces normalisation. An ensemble of cells races from a start ring toward a
source; the simulator reports how many arrive and how fast.

The active codebase is **`cellsim/`** (Python). The original MATLAB implementation
it was ported from lives in `legacy/` for reference.

## Quick start

Install [uv](https://docs.astral.sh/uv/), then from the repo root:

```bash
uv sync                                              # create the env (numpy, scipy, matplotlib)
uv run python -m cellsim                             # 100 cells, tissue point source, 4 h
uv run python -m cellsim --env radial_exp            # smooth analytic gradient instead
uv run python -m cellsim --no-noisy                  # deterministic sensing (MATLAB-style)
uv run python -m cellsim --d 0.02 --reps 5           # 5 repeats with different seeds
uv run python -m cellsim --dcouple                   # signal-coupled diffusivity
uv run python -m cellsim --help                      # every knob, grouped
```

Each run prints the success rate and arrival-time distribution per repeat.

Prefer pip? `pip install -e .` then `python -m cellsim ...` (or `cellsim ...`).
The existing `python run.py ...` command still works.

The other Python components use the same environment and package convention:

```bash
uv run python -m source_kf --help                    # source-localization demo
uv sync --extra robotic                              # add robotics dependencies
uv run --extra robotic python -m robotic --help      # EKF / UKF tracking demo
```

See [the development guide](docs/development.md) for where code belongs, and
[the robotics guide](robotic/README.md) for studies and plotting commands.

## What you can vary

| Group | Flags | What it controls |
|---|---|---|
| Environment | `--env`, `--source`, `--conc-scale` | which field, point/edge source, overall ligand level |
| Cell | `--receptor`, `--decoder`, `--stepsz`, `--direction-noise` | sensing scheme, how the belief becomes motion, speed, steering noise |
| Belief | `--d`, `--rtot`, `--kd-nM` | membrane diffusivity (belief blur), receptor pool, sensing affinity |
| Sensing noise | `--noisy`, `--nsamp` | Poisson shot noise, samples averaged per step |
| Coupling | `--dcouple`, `--couple-form`, `--z0`, `--dn`, `--couple-eps` | make diffusivity depend on local signal |
| Run | `--hours`, `--cells`, `--reps`, `--seed` | time budget, ensemble size, repeats |

**Environments** (`--env`): `tissue_point_noflow` (default; point source in
heterogeneous tissue), `tissue_env` (edge source), `radial_exp` (smooth analytic
exponential gradient — the clean, heterogeneity-free control), or any tissue-field
`.mat` stem under `tissue_sim/` (e.g. `tissue_env_koff=1e-2`).

**Decoders** (`--decoder`): how the activity profile becomes a bearing —
`weighted_mean` (activity-weighted circular mean over all bins; default),
`steepest` (the single bin of largest front-to-back contrast), or `random`
(ignores the signal; control). `--direction-noise` then adds Gaussian steering
noise on top of *any* decoder (set it to `0` for noiseless movement).

## Using it as a library

```python
from cellsim import Params, race

p = Params(noisy=True, nsamp=1, d=0.02, conc_scale=0.5)
result = race("radial_exp", p, seed=0)
print(result.summary())          # success rate + arrival quartiles
```

`simulate_cell` runs a single cell and returns its full trajectory; see
`cellsim/__init__.py` for the public API.

## Directory structure

```
.
├── pyproject.toml       # shared Python dependencies, extras, and package config
├── uv.lock              # pinned dependency versions
├── run.py               # compatibility wrapper for cellsim.cli
├── cellsim/             # cell-navigation simulator
│   ├── __main__.py      # python -m cellsim
│   ├── cli.py           # argument parsing and run orchestration
│   ├── params.py        # parameters
│   ├── environment.py   # tissue .mat loader and analytic gradients
│   ├── receptor.py      # sensing and membrane dynamics
│   ├── decoder.py       # activity-to-bearing readouts
│   ├── cell.py          # single-cell simulation
│   ├── ensemble.py      # ensembles and arrival statistics
│   └── tests/           # regression tests and MATLAB parity fixtures
├── source_kf/           # receptor-based source-localization demo
├── robotic/             # EKF, UKF, and histogram-filter studies
│   ├── core/            # environments, configuration, visualization
│   ├── filters/         # Kalman-filter implementations
│   ├── shared/          # signal models and common utilities
│   ├── analysis/        # comparisons and plotting
│   ├── histogram_filter/ # grid-based Bayesian navigation study
│   └── output/          # existing study results and figures
├── tissue_sim/          # tissue input data and MATLAB field generators
│   ├── point_source/
│   └── koff_variants/
├── legacy/              # original MATLAB navigation code
├── LEGI/                # reference LEGI models
└── docs/                # development and organization guide
```

Generated scratch figures belong in `figures/`; new scratch run results can go in
`output/`. Both root folders are ignored by Git. Existing results in
`robotic/output/` retain their paths. Tissue data and parity fixtures remain beside
the code that generates or validates them.

## Tests

The port is validated against the MATLAB, at two levels:

```bash
uv run python -m cellsim.tests.test_matlab_parity    # deterministic blocks, bit-identical to MATLAB
uv run python -m cellsim.tests.test_properties       # invariants (conservation, monotonicity, ...)
uv run python -m cellsim.tests.test_ensemble         # slow (~4 min): success rates vs MATLAB
```

Deterministic building blocks match MATLAB to floating point; ensemble success
rates agree statistically. Individual trajectories do **not** match run-for-run —
the interpolator and RNG differ and the dynamics are chaotic — so compare
distributions, not paths.

## Legacy MATLAB (`legacy/src/`)

The original navigation code. `cellsim/` is a validated port of it and is the code
you should run; `legacy/` is kept for provenance and for regenerating the test
reference (`cellsim/tests/data/export_unit_ref.m`). The tissue-field `.mat` files
stay in `tissue_sim/` and are loaded directly by `cellsim` — that data is **not**
legacy, even though the `.m` scripts beside it (which generated the fields) are.

## LEGI models

`LEGI/` contains Local Excitation, Global Inhibition models (code from
[Shi et al. 2013](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003122)),
kept as reference. Not part of the `cellsim` simulation path.

## License

See [LICENSE](LICENSE).
