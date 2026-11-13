# source_kf

Linear-Gaussian **source localization as an exact Kalman filter** — the agent finds a
source it cannot see, its belief over the source stays Gaussian, and (unlike
`kalman_nav`) sensing is what drives the search.

**Setup.** A source sits at an unknown point in `[0,200]²`. The concentration field is
a smooth bump `c(x) = peak − ½κ|s − x|²`, with `κ` set so it peaks at `peak` (=20) at
the source and stays ≥0 across the whole arena. Its gradient `κ(s − p)` is linear in the
unknown source position `s` (the `1/κ` is folded into the readout, recovering `s − p`).

**State = the vector to the source** `d = s − p` (what the gradient measures). This is
what lets movement error enter the filter exactly: a commanded move is a known control
(`d ← d − u`) and the landing error `w ~ N(0, σ_move²)` is process noise (`P ← P + Q`),
since `d_{t+1} = d_t − u_t − w_t` is linear-Gaussian. So the predict step is exact, like
the update. Motion noise keeps injecting `Q`, so the belief settles at a **steady-state
floor** instead of collapsing to zero.

**Sensing is mechanistic** (as a cell would): `N` receptors sit evenly around the
membrane (radius `r`), each reading the local concentration with Gaussian noise. The
cell combines them with a **linear vector-sum readout** `ĝ = (2/Nr) Σ û(φ_i) m(φ_i)`.
The offset common to all receptors cancels (unit vectors around a circle sum to zero),
leaving `ĝ = (s − p) + noise = d + noise` (`H = I`) — a direct linear observation, so the
belief stays Gaussian and the Kalman filter is **exact**. Per-receptor noise `σ_rec`
becomes an effective source-fix noise `σ_g = σ_rec·√(2/N)/r`.

**Why not the flat `c = x+y` ramp?** A flat ramp has the same gradient everywhere, so
it reveals nothing about where a source is — you can't localize from it. A *peak* is
the minimal field that makes the gradient point at the source, and a *quadratic* peak
keeps that pointing exactly linear, so the filter stays an exact Kalman filter.

**Sensing is essential here.** Turn it off (`--blind`) and the belief never updates —
its mean sits at the prior center and the agent walks there and is lost. You cannot
dead-reckon to a source that only exists in the measurements.

| | with sensing | blind |
|---|---:|---:|
| arrives? | yes, ~31 steps | never |
| belief std over source | 80 → ~5 | stays 80 |
| final estimate error | ~5 | ~70 |

## Related code in this repository

- `source_kf/`: receptor-based source-localization demonstration.
- `cellsim/`: biochemical cell-navigation simulator.
- `robotic/`: Kalman and histogram-filter navigation studies.

See [the development guide](../docs/development.md) for the shared environment and
[the main README](../README.md) for the repository map.

## Run

```bash
uv run python -m source_kf                    # seek + save figures/source_kf.png
uv run python -m source_kf --blind            # no-sensing control
uv run python -m source_kf --sigma-receptor 2 --seed 3 --show
```

Run from the repository root. Use `uv run python -m source_kf --help` for current
options; `--save ''` skips figure output.
