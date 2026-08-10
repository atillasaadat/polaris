# `analysis/detumble` — the B-dot detumble Monte Carlo campaign

Characterises the **residual-spin tail**: how long a B-dot detumble takes to
reach the `DetumbleExitRadps` completion predicate, across the dispersions that
actually drive it. This is the item REQ-ACTL-001 records as owed — the
requirement is verified today on the fast phase only (below 3.8 deg/s within
200 s of engaging), and carries no bound on the time to completion because no
campaign had been flown.

Design doc §13 (analysis), §23.2 (Monte Carlo). The requirement is in
`docs/requirements/adcs_control.rst`.

## The split, and why

| Piece | Where | Why |
|---|---|---|
| Flying the runs | `tests/mc/detumble_mc.cpp` → `polaris_detumble_mc` | C++. Each run forks the real deployment and drives the real `ClosedLoop` over the SITL wire, exactly as the SITL rows do. No GNC math is reimplemented. |
| Statistics, figures, report | this package | Python. Sampling statistics, plotting and report generation are the named exceptions in `analysis/CLAUDE.md`. |
| The interchange | JSONL under `build-artifacts/` | A derived artifact. Never committed. |

B-dot's input is not the magnetometer — it is the voted, plausibility-gated,
quiet-window field the `AttitudeEstimator` publishes under the §7 MTQ/MAG duty
cycle. Rebuilding that chain sim-side would be a transcription of the flight
path, so the driver flies the flight binary. That costs about 8x real time per
run with the reference vehicle's full environment, which is what makes this a
campaign and not a CI gate.

## Running the campaign

Build the deployment and the driver first (the driver skips nothing — it exits
non-zero if either the flight binary or the `configc` interpreter is missing):

```bash
uv run fprime-util build                                     # the deployment
uv run cmake --build build-fprime-automatic-native \
    --target polaris_detumble_mc -j8
```

Build it in `build-fprime-automatic-native`, not the `-ut` tree: F´ compiles the
unit-test tree with ASan/UBSan, which the campaign does not need and pays about
3x for.

**Smoke case — two short runs, a couple of minutes.** This is what proves the
harness works; it is deliberately *not* registered with ctest, for the same
reason `tests/benchmark/` is not (see `tests/mc/CMakeLists.txt`).

```bash
./build-fprime-automatic-native/bin/Linux/polaris_detumble_mc \
    --runs 2 --duration-s 600 --out build-artifacts/detumble-mc/smoke.jsonl
```

**The full campaign.** Runs are sequential within one process; parallelise by
sharding across processes, one shard per core. Shards are independently
restartable, and each needs its own `--work` directory (the deployment's
parameter file and event logs live there).

```bash
N=59            # see "How many runs" below
JOBS=8
mkdir -p build-artifacts/detumble-mc/campaign
seq 0 $((N-1)) | xargs -P $JOBS -I{} \
  ./build-fprime-automatic-native/bin/Linux/polaris_detumble_mc \
      --first-run {} --runs 1 --duration-s 45416 \
      --work build-artifacts/detumble-mc/campaign/{} \
      --out  build-artifacts/detumble-mc/campaign/shard-{}.jsonl
```

`--duration-s 45416` is eight orbits at the reference vehicle's 5677 s period.
Size it above the longest tail you expect: a run whose arc ends first is
**right-censored**, and the analysis fails the campaign rather than quoting a
bound that points the wrong way.

**Reading the result.** The analysis takes the shard directory directly.

```bash
uv run --group analysis python -m analysis.detumble \
    build-artifacts/detumble-mc/campaign
```

It prints the report, writes figures and the rendered report under
`build-artifacts/analysis/detumble/`, and exits non-zero on any FAIL.

## Reproducing one run

Every record carries its dispersion draw, its derived seed and the
`config_hash` of the vehicle it flew. Re-fly a single run with:

```bash
./build-fprime-automatic-native/bin/Linux/polaris_detumble_mc \
    --first-run <run_index> --runs 1 --seed <master seed> --duration-s <same>
```

The draw is derived from `{master seed, run index}`, not walked from a single
generator, so adding runs to a campaign never moves an existing run's geometry.

## The open-loop ablation

`--ctrl-mode 0` flies the identical dispersed scenario with the control law in
IDLE. That is the comparison that separates what B-dot did from what the
environment did to the same vehicle — use it before attributing anything in a
rate history to the control law.

```bash
./build-fprime-automatic-native/bin/Linux/polaris_detumble_mc \
    --runs 1 --duration-s 11354 --ctrl-mode 0 \
    --out build-artifacts/detumble-mc/idle.jsonl
```

## What is dispersed

| Draw | Distribution | Why |
|---|---|---|
| Tip-off rate magnitude | uniform 2–5 deg/s | REQ-ACTL-001's 5 deg/s is the demanding end of a dispenser's spec; the spread is what lets the campaign *test* whether magnitude drives the tail. |
| Tip-off direction | uniform on the sphere | The whole point. B-dot is blind to the rate component along **B**, so the spin's orientation relative to the field decides how much residual there is. |
| Initial attitude | uniform on SO(3) | The rod triad is body-fixed and the dipole clamp is per rod, so orientation decides when the law saturates. |
| RAAN, argument of latitude | uniform 0–360° | Where the orbit plane sits relative to the geomagnetic dipole, and where in it the run starts. |
| Epoch | uniform over a sidereal day | The dipole is tilted ~11°, so Earth rotation moves it against a fixed plane on a daily cycle. |

The orbit dispersion is applied as **rotations of the compiled Cartesian state**
(about ECI +Z for the node, about the orbit normal for the argument of latitude),
which is exact for the reference vehicle's circular orbit and keeps
Keplerian-to-Cartesian conversion in the config compiler where §19.3 puts it.
The driver checks the compiled orbit really is circular before rotating.

The campaign clears the committed scenario's scheduled GNSS outage and spoof and
disables the geographic jamming map. Either one starves B-dot of its only input,
which is a fault-response duration and not a convergence measurement; detumble
through a GNSS outage belongs in the FDIR suite.

## How many runs

The handover time is an upper tolerance bound on a high quantile, and the tail
distribution has no reason to be normal — so the bound is distribution-free
(Wilks order statistics; see `statistics.py`). For a 95th percentile at 95 %
confidence:

- **59 runs** support a bound read from the sample maximum (first order).
- **93 runs** support one read from the second largest, which survives a single
  outlier run — the preferred size, because a first-order bound *is* the worst
  run and cannot tell "the physics does this" from "one run misbehaved".

`analysis.detumble.statistics.wilks_sample_size` computes these; the report
fails the campaign if the sample is too small for the bound it quotes.

## Output schema

One JSON object per line. The fields are documented on
`analysis.detumble.records.RunRecord`; the ones that carry the argument are
`t_engage_s` (B-dot's first commanded cycle, read off the wire, which is the
origin every other time is measured from), `t_exit_s`, the truth-side rate
milestones, the spin/field angle at engagement and at the end of the fast phase,
and — as diagnostics that make a rate history interpretable — the peak commanded
wheel torque (must be zero in DETUMBLE) and the rotational kinetic energy at
engagement, at its minimum and at the end. B-dot's guarantee is dissipativity in
**energy**; with a near-isotropic inertia |ω| can only differ from it by
√(J_max/J_min), so energy is what tells a redistribution from a defect.

The deployment's event log is deleted for a healthy run that engaged, and kept
otherwise — a run in which B-dot never commanded a rod is a run whose event
stream is the only thing that says why.
