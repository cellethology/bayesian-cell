"""
Linear-Gaussian counterpart of analysis/signal_max_comparison.py.

Reuses analysis.comparison.FilterComparison ENTIRELY (parallelism, summary stats,
paired t-test, plots).  The only hook needed is that FilterComparison.run_comparison
looks up `run_single_ekf_simulation` in its own module globals at call time, so
rebinding that global routes every task to the linear environment.  This survives the
`spawn` start method because the replacement function still pickles by reference to
ITS OWN module, which the child re-imports.

Run:  uv run python -m robotic.analysis.linear_gaussian_comparison
"""

# Support the historical direct-script command as well as package execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "robotic.analysis"

import contextlib

import numpy as np
import pandas as pd

from . import comparison as _cmp
from .comparison import FilterComparison
from ..core.linear_config import (
    get_linear_base_config,
    get_linear_method_configs,
    get_linear_study_config,
)
from ..core.linear_environment import LinearGaussianEnvironment


def run_single_linear_simulation(args):
    """Byte-for-byte the body of analysis.comparison.run_single_ekf_simulation with
    EKFEnvironment -> LinearGaussianEnvironment."""
    seed, config_name, config, verbose, collect_trajectories = args

    config_copy = config.copy()
    config_copy.pop("random_seed", None)

    env = LinearGaussianEnvironment(config_copy, verbose=verbose, random_seed=seed)
    results = env.run_simulation()

    sigma_mean = (
        np.mean(results["sigma_history"])
        if len(results["sigma_history"]) > 0
        else config.get("baseline_process_noise")
    )
    sigma_std = (
        np.std(results["sigma_history"]) if len(results["sigma_history"]) > 0 else 0.0
    )

    performance_metrics = {
        "config_name": config_name,
        "seed": seed,
        "steps_to_target": results["steps_completed"],
        "target_reached": results["target_reached"],
        "sigma_mean": sigma_mean,
        "sigma_std": sigma_std,
        "adaptive_process_noise": config.get("adaptive_process_noise"),
        "adaptive_measurement_noise": config.get("adaptive_measurement_noise"),
    }

    if collect_trajectories:
        return {
            "performance_metrics": performance_metrics,
            "results": results,
            "seed": seed,
            "robot_trajectory": results["robot_trajectory"],
            "target_trajectory": results["target_trajectory"],
            "config": config_copy,
        }
    return performance_metrics


@contextlib.contextmanager
def linear_environment():
    """Route FilterComparison's workers at the linear environment."""
    original = _cmp.run_single_ekf_simulation
    _cmp.run_single_ekf_simulation = run_single_linear_simulation
    try:
        yield
    finally:
        _cmp.run_single_ekf_simulation = original


def run_linear_signal_max_comparison(
    preset="robotic",
    overrides=None,
    n_runs=None,
    out_csv="output/linear_signal_max_results.csv",
    verbose=True,
):
    base = get_linear_base_config(preset)
    base.update(overrides or {})
    study = get_linear_study_config()
    methods = get_linear_method_configs()
    n = n_runs if n_runs is not None else study["n_runs"]

    all_results = []
    for signal_max in study["signal_max_values"]:
        if verbose:
            print(f"linear signal_max = {signal_max} (preset={preset}, n={n})")
        cfg = base.copy()
        cfg["signal_max"] = float(signal_max)
        comparison = FilterComparison(cfg)
        for name, mcfg in methods.items():
            comparison.add_config(name, mcfg)
        with linear_environment():
            res = comparison.run_comparison(n_runs=n, verbose=False)
        res["signal_max"] = signal_max
        res["method"] = res["config_name"]
        all_results.append(res)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(out_csv, index=False)
    if verbose:
        print(f"saved {out_csv}")
    return combined


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Linear-Gaussian signal_max sweep")
    ap.add_argument("--preset", default="robotic", choices=["robotic", "source_kf"])
    ap.add_argument("--n-runs", type=int, default=None)
    ap.add_argument("--out", default="output/linear_signal_max_results.csv")
    args = ap.parse_args()
    run_linear_signal_max_comparison(
        preset=args.preset, n_runs=args.n_runs, out_csv=args.out
    )
