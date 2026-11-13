"""Plotting for linear-Gaussian source localization: the belief ellipse shrinking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from .sim import Config, Result, effective_sigma, kappa


def _ellipse(ax, mean, cov, n_std=2.0, **kw):
    w, V = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    ax.add_patch(Ellipse(mean, 2 * n_std * np.sqrt(w[1]), 2 * n_std * np.sqrt(w[0]),
                         angle=ang, **kw))


def plot_run(result: Result, cfg: Config | None = None, save_path: str | None = None,
             show: bool = False):
    cfg = cfg or Config()
    path, est, snaps = result.path, result.est, result.snaps
    spath = result.source_path
    src = spath[-1]                                         # final source position (field drawn here)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6.3))

    # --- left: arena, field, trajectory, shrinking belief ellipses ---
    g = np.linspace(0, cfg.arena, 240)
    X, Y = np.meshgrid(g, g)
    field = np.maximum(cfg.peak - 0.5 * kappa(cfg) * ((X - src[0]) ** 2 + (Y - src[1]) ** 2), 0.0)
    cf = ax[0].contourf(X, Y, field, levels=25, cmap="viridis", alpha=0.65)
    cb = fig.colorbar(cf, ax=ax[0], fraction=0.046, pad=0.04)
    cb.set_label(f"ligand concentration (peak {cfg.peak:.0f} at source, ≥0)", fontsize=9)

    idx = np.linspace(0, len(snaps) - 1, min(6, len(snaps))).astype(int)
    for k in idx:
        step, m, P = snaps[k]
        _ellipse(ax[0], m, P, fill=False, ec="white", lw=1.1, alpha=0.85)
    n = min(len(path), len(est))
    ax[0].plot(path[:, 0], path[:, 1], "-", color="orangered", lw=2.2, label="agent path")
    ax[0].plot(est[:n, 0], est[:n, 1], "--", color="deepskyblue", lw=1.5,
               label="belief mean (source estimate)")
    ax[0].plot(*cfg.start, "wo", ms=10, mec="k", label=f"start {tuple(map(int, cfg.start))}")
    ax[0].plot(spath[:, 0], spath[:, 1], "-", color="gold", lw=1.3, alpha=0.9, label="source path (drifting)")
    ax[0].plot(*spath[0], "*", color="gold", ms=13, mec="k", alpha=0.45)
    ax[0].plot(*src, "*", color="gold", ms=22, mec="k", label="source final")
    ax[0].plot(*path[-1], "P", color="orangered", ms=13, mec="k", label="agent final")
    ax[0].set_xlim(0, cfg.arena); ax[0].set_ylim(0, cfg.arena); ax[0].set_aspect("equal")
    ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
    status = "arrived" if result.arrived else "did not arrive"
    ax[0].set_title(f"Linear-Gaussian source localization — {status} in {result.n_steps} steps\n"
                    "white ellipses = 2σ belief over the source (shrinking)", fontsize=11)
    ax[0].legend(fontsize=8, loc="lower right", framealpha=0.9)

    # --- right: belief uncertainty over time ---
    steps = [s_[0] for s_ in snaps]
    stds = [np.sqrt(np.trace(s_[2]) / 2.0) for s_ in snaps]      # RMS belief std over the source
    sig = effective_sigma(cfg)
    ax[1].plot(steps, stds, "-o", color="crimson", ms=3, lw=1.8)
    ax[1].axhline(sig, ls=":", color="0.5",
                  label=f"filter's assumed σ_g={sig:.0f}")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("belief std over source position")
    ax[1].set_title("Uncertainty drops, then plateaus\n"
                    "(motion noise keeps injecting Q → steady-state floor, not σ/√N→0)",
                    fontsize=11)
    ax[1].legend(fontsize=9); ax[1].set_ylim(bottom=0)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    return fig
