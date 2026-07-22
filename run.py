"""Run a cell-navigation simulation.

Simulates an ensemble of cells chemotaxing up a ligand field and reports how many
reach the source. Every knob below has a sensible default, so:

    python run.py                                  # defaults: shot-noise sensing (nsamp=1)
    python run.py --env radial_exp                 # smooth analytic gradient instead
    python run.py --no-noisy                       # deterministic sensing (MATLAB-style)
    python run.py --d 0.02 --stepsz 1.0 --reps 5   # sweep-style: 5 repeats
    python run.py --dcouple                        # signal-coupled diffusivity (invsqrt)

Environments (--env):
    tissue_point_noflow   point source in heterogeneous tissue (default)
    tissue_env            edge source in tissue
    radial_exp            smooth analytic exponential gradient (clean control)
    tissue_env_koff=1e-2  ... any tissue-field file under tissue_sim/ (by stem)
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from cellsim.ensemble import race
from cellsim.params import Params


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    env = ap.add_argument_group("environment")
    env.add_argument("--env", default="tissue_point_noflow",
                     help="field: a .mat stem under environments/, or 'radial_exp'")
    env.add_argument("--source", default="point", choices=["point", "edge"])
    env.add_argument("--conc-scale", type=float, default=1.0,
                     help="multiply the whole field; <1 = weaker, noisier signal")

    cell = ap.add_argument_group("cell / navigation")
    cell.add_argument("--receptor", default="feedback", choices=["feedback", "uniform"])
    cell.add_argument("--decoder", default="weighted_mean",
                      choices=["weighted_mean", "steepest", "random"])
    cell.add_argument("--bins", type=int, default=100, help="membrane bins N")
    cell.add_argument("--radius", type=float, default=6.0, help="cell radius (um)")
    cell.add_argument("--stepsz", type=float, default=1.0, help="cell step size (um/move)")
    cell.add_argument("--direction-noise", type=float, default=0.1,
                      help="std (rad) of steering noise added after any decoder")

    rec = ap.add_argument_group("receptor / belief")
    rec.add_argument("--d", type=float, default=0.01, help="membrane diffusivity (belief blur)")
    rec.add_argument("--rtot", type=float, default=2000.0, help="total receptor pool")
    rec.add_argument("--kd-nM", type=float, default=10.0, help="half-saturation (nM)")

    sense = ap.add_argument_group("sensing noise")
    sense.add_argument("--noisy", action=argparse.BooleanOptionalAction, default=True,
                       help="Poisson sensing noise (on by default; --no-noisy for deterministic)")
    sense.add_argument("--nsamp", type=int, default=1,
                       help="Poisson samples averaged per step (1 = fully shot-noise-limited)")

    couple = ap.add_argument_group("signal-coupled diffusivity (optional)")
    couple.add_argument("--dcouple", action="store_true", help="make d depend on local signal")
    couple.add_argument("--couple-form", default="invsqrt", choices=["hill", "invsqrt"],
                        help="invsqrt: d0/sqrt(eps+count) (default); hill: d0/(1+(z/z0)^n)")
    couple.add_argument("--z0", type=float, default=0.2)
    couple.add_argument("--dn", type=float, default=2.0)
    couple.add_argument("--couple-eps", type=float, default=1.0)
    couple.add_argument("--dmin", type=float, default=None, help="floor on diffusivity")

    run = ap.add_argument_group("run")
    run.add_argument("--hours", type=float, default=4.0, help="simulated time budget")
    run.add_argument("--cells", type=int, default=100)
    run.add_argument("--reps", type=int, default=1, help="repeats with different seeds")
    run.add_argument("--processes", type=int, default=None, help="worker processes (default: all cores)")
    run.add_argument("--seed", type=int, default=0)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    p = Params(
        N=args.bins, mean_cell_radius=args.radius, T=args.hours * 3600.0,
        stepsz=args.stepsz, rtot=args.rtot, kd_nM=args.kd_nM, d=args.d,
        noisy=args.noisy, nsamp=args.nsamp, direction_noise=args.direction_noise,
        conc_scale=args.conc_scale, dcouple=args.dcouple, couple_form=args.couple_form,
        z0=args.z0, dn=args.dn, couple_eps=args.couple_eps, dmin=args.dmin,
    )
    print(f"env={args.env} source={args.source} receptor={args.receptor} decoder={args.decoder}")
    print(f"T={p.T:g}s ({args.hours:g}h) N={p.N} d={p.d:g} stepsz={p.stepsz:g} "
          f"conc_scale={p.conc_scale:g} noisy={p.noisy} dcouple={p.dcouple}")

    rates = []
    for r in range(args.reps):
        t0 = time.perf_counter()
        res = race(args.env, p, source=args.source, receptor=args.receptor,
                   decoder_method=args.decoder, n_run=args.cells,
                   processes=args.processes, seed=args.seed + r)
        rates.append(res.success_rate)
        print(f"  rep {r + 1}/{args.reps}: {res.summary()}  ({time.perf_counter() - t0:.1f}s)")

    if args.reps > 1:
        a = np.array(rates)
        print(f"\nmean success {a.mean():.1f} +- {a.std(ddof=1) / np.sqrt(a.size):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
