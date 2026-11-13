"""Demo entry point: ``python -m source_kf``."""

from __future__ import annotations

import argparse

import numpy as np

from .plot import plot_run
from .sim import Config, effective_sigma, kappa, run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Linear-Gaussian source-localization demo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-receptors", type=int, default=8)
    ap.add_argument("--cell-radius", type=float, default=5.0)
    ap.add_argument("--peak", type=float, default=20.0, help="ligand concentration at the source")
    ap.add_argument("--sigma-receptor", type=float, default=1.0, help="receptor noise scale")
    ap.add_argument("--gaussian", action="store_true", help="constant Gaussian noise (not Poisson)")
    ap.add_argument("--adapt-r", action="store_true", help="R tracks the local Poisson variance")
    ap.add_argument("--sigma-source", type=float, default=0.5, help="per-step source diffusion")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--blind", action="store_true", help="disable sensing (control)")
    ap.add_argument("--save", default="figures/source_kf.png", help="figure path ('' to skip)")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args(argv)

    cfg = Config(speed=args.speed, n_receptors=args.n_receptors, cell_radius=args.cell_radius,
                 peak=args.peak, sigma_receptor=args.sigma_receptor, poisson=not args.gaussian,
                 adapt_R=args.adapt_r, sigma_source=args.sigma_source)
    r = run(cfg, seed=args.seed, max_steps=args.max_steps, sense=not args.blind)

    sg = effective_sigma(cfg)
    print(f"receptors: N={cfg.n_receptors}, radius={cfg.cell_radius:g}, sigma_receptor={cfg.sigma_receptor:g}, "
          f"{'POISSON (var=mean conc)' if cfg.poisson else 'Gaussian'}; source diffusion={cfg.sigma_source:g}")
    print(f"  filter assumes sigma_g = {sg:.0f}")
    if cfg.poisson:
        c_start = cfg.peak - 0.5 * kappa(cfg) * float(np.sum((np.asarray(cfg.source) - np.asarray(cfg.start)) ** 2))
        print(f"  true sigma_g ~ sqrt(conc)*{sg:.0f} = {np.sqrt(max(c_start,0))*sg:.0f} (start)"
              f" .. {np.sqrt(cfg.peak)*sg:.0f} (near source)  -> filter is overconfident")
    print(f"{'BLIND (no sensing)' if args.blind else 'sensing'}: "
          f"arrived={r.arrived} in {r.n_steps} steps")
    print(f"  source drifted {tuple(map(int, r.source_path[0]))} -> {tuple(map(int, r.source_path[-1]))}, "
          f"final belief mean ({r.est[-1][0]:.1f}, {r.est[-1][1]:.1f}), error {r.final_error:.1f}")
    final_std = np.sqrt(np.trace(r.snaps[-1][2]) / 2.0)
    print(f"  final belief std over source: {final_std:.1f} (started at {cfg.prior_std:.0f})")

    if args.save:
        plot_run(r, cfg, save_path=args.save, show=args.show)
        print(f"  figure saved to {args.save}")
    elif args.show:
        plot_run(r, cfg, show=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
