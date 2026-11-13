# Robotic navigation studies

Run these commands from the repository root. Install the optional dependencies:

```bash
uv sync --locked --extra robotic
```

## Tracking demo

```bash
uv run --extra robotic python -m robotic --help
uv run --extra robotic python -m robotic --filter EKF --max-steps 100 --no-show-plots
uv run --extra robotic python -m robotic --filter UKF --max-steps 100 --no-show-plots
```

`python robotic/main.py ...` remains supported. The default output directory is
`output/` relative to the working directory; `--output-dir` changes it and
`--save-plots` enables figure saving.

## Comparisons and plots

```bash
mkdir -p output
uv run --extra robotic python -m robotic.analysis.linear_gaussian_comparison --help
uv run --extra robotic python -m robotic.analysis.signal_max_comparison
```

Comparison studies use the run counts and step budgets in `core/base_config.py`
and `core/linear_config.py`; defaults are full research sweeps. For the linear
study, `--n-runs` controls repeats and `--out` controls the CSV destination.

Plot an existing result without rerunning a study:

```python
from robotic.analysis.plot_signal_max_results import (
    load_and_prepare_data,
    create_mean_plot_with_sem,
)

results = load_and_prepare_data("robotic/output/signal_max_comparison_results.csv")
create_mean_plot_with_sem(
    results,
    save_path="output/signal_max_comparison_plot.pdf",
    show_plot=False,
)
```

The histogram-filter models and comparisons are available through package imports:

```python
from robotic.histogram_filter.navigation_env import run_navigation_simulation
from robotic.histogram_filter.multi_config_comparison import MultiConfigComparison

result = run_navigation_simulation(steps=2)
```

Its sweep entry point is
`python -m robotic.histogram_filter.signal_decay_comparison`; review the experiment
budget in that module before a full run.

## Layout

| Directory | Responsibility |
|---|---|
| `core/` | Tracking environments, configurations, and visualizers |
| `filters/` | EKF, UKF, and linear Kalman-filter implementations |
| `shared/` | Field models, statistics, configuration, and output utilities |
| `analysis/` | Filter comparisons, sweeps, and result plots |
| `histogram_filter/` | Grid-based navigation models and their experiments |
| `output/` | Existing study CSVs and figures |

Scratch runs from the repository root write to root `output/`, which is ignored
by Git. Existing reference results in `robotic/output/` remain available at their
original paths. See [the development guide](../docs/development.md) for conventions.
