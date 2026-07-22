"""Ensemble driver: race N cells from a start line and score how many arrive.

Port of ``legacy/src/racing_cells.m``. Unlike the MATLAB, each cell gets an explicitly
seeded generator, so runs are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool

import numpy as np

from .cell import simulate_cell
from .environment import make_field
from .params import Params

N_RUN = 100  # cells per ensemble, matching racing_cells.m:79


def start_positions(p: Params, source: str, task: str = "localization",
                    n_run: int = N_RUN) -> np.ndarray:
    """Where cells are released. Port of ``racing_cells.m`` lines 79-96."""
    if source == "point":
        theta = np.linspace(0, 2 * np.pi * (1 - 1 / n_run), n_run)
        return np.column_stack([90 * np.cos(theta), 90 * np.sin(theta)])
    xc = 50.0 if task == "localization" else 5.0
    yc = np.linspace(p.mean_cell_radius * 10, 900 - p.mean_cell_radius * 10, n_run)
    return np.column_stack([np.full(n_run, xc), yc])


@dataclass
class EnsembleResult:
    success_rate: float          # percent of cells reaching the source
    arrival_min: np.ndarray      # (n_run,) minutes; NaN if never arrived
    stat_summary: np.ndarray     # mean of per-cell record_stat
    total_time_min: float

    @property
    def n_arrived(self) -> int:
        return int(np.sum(~np.isnan(self.arrival_min)))

    def summary(self) -> str:
        t = self.arrival_min[~np.isnan(self.arrival_min)]
        if t.size == 0:
            return f"success 0.0% (0/{self.arrival_min.size})"
        return (
            f"success {self.success_rate:.1f}% "
            f"({self.n_arrived}/{self.arrival_min.size}), "
            f"arrival median {np.median(t):.1f} min "
            f"[Q1 {np.percentile(t, 25):.1f}, Q3 {np.percentile(t, 75):.1f}]"
        )


_WORKER: dict = {}


def _init_worker(env_name, conversion_factor, source, conc_scale):
    """Build the ligand field once per worker process.

    Without this the field is reloaded from disk for every cell (~16 ms x
    n_run per race), which is a large fraction of a short run.
    """
    _WORKER["field"] = make_field(env_name, conversion_factor, source, conc_scale)


def _run_one(args, p, env_name, source, receptor, decoder_method):
    idx, start, seed = args
    field = _WORKER.get("field")
    if field is None:  # serial path, or a worker that skipped the initializer
        field = _WORKER["field"] = make_field(
            env_name, p.conversion_factor, source, p.conc_scale)
    rng = np.random.default_rng(seed)
    res = simulate_cell(p, field, start, receptor=receptor,
                        mode="localization", decoder_method=decoder_method, rng=rng)
    # arrival time from the move-subsampled trace, as racing_cells.m does:
    # count nonzero samples / 2 -> minutes (one sample per 30 s)
    sub = res.positions_at_moves(p.move_rate)
    minutes = np.count_nonzero(sub[:, 0]) / 2.0
    return idx, minutes, res.stat


def race(
    env_name: str,
    p: Params,
    source: str = "point",
    receptor: str = "feedback",
    decoder_method: str = "optimal_noise",
    n_run: int = N_RUN,
    processes: int | None = None,
    seed: int = 0,
) -> EnsembleResult:
    """Simulate ``n_run`` cells and report the fraction reaching the source."""
    starts = start_positions(p, source, n_run=n_run)
    seeds = np.random.SeedSequence(seed).spawn(n_run)
    tasks = [(i, starts[i], seeds[i]) for i in range(n_run)]

    worker = partial(_run_one, p=p, env_name=env_name, source=source,
                     receptor=receptor, decoder_method=decoder_method)

    if processes == 1:
        _init_worker(env_name, p.conversion_factor, source, p.conc_scale)
        out = [worker(t) for t in tasks]
    else:
        # NOTE: callers must be under an `if __name__ == "__main__"` guard --
        # macOS spawns rather than forks, so an unguarded script makes every
        # worker re-execute it and spawn its own pool.
        with Pool(processes,
                  initializer=_init_worker,
                  initargs=(env_name, p.conversion_factor, source, p.conc_scale)) as pool:
            out = pool.map(worker, tasks, chunksize=max(1, len(tasks) // (4 * (processes or 8))))

    total_time_min = p.n_steps / p.move_rate / 2.0
    arrival = np.full(n_run, np.nan)
    stats = np.zeros((n_run, 4))
    for idx, minutes, stat in out:
        if minutes < total_time_min:
            arrival[idx] = minutes
        stats[idx] = stat

    return EnsembleResult(
        success_rate=100.0 * np.mean(~np.isnan(arrival)),
        arrival_min=arrival,
        stat_summary=stats.mean(axis=0),
        total_time_min=total_time_min,
    )
