# `tools/gmat/` — GMAT golden-data harness

GMAT (NASA GSFC) is the V&V reference tool (design doc §23.1, REQ-SYS-010 /
REQ-VV-002): it generates the versioned reference ("golden") datasets in
[`tests/golden/`](../../tests/golden/) that Polaris numerical functions are
checked against, within documented per-quantity tolerance bands.

**GMAT is not a build dependency.** It is a ~300 MB NASA desktop tool shipped as
a per-release Linux/Windows tarball (no apt/pip/conda package), so the inner
build and every PR stay GMAT-free: the fixtures are committed data, and CI's
per-PR lane only compares Polaris output against them. GMAT is used two ways,
both outside that loop:

- **Regenerate** a fixture when the reference case changes (below).
- **Drift-check** — a scheduled job (and an opt-in local test) that reruns GMAT
  and fails if the committed fixtures no longer agree with it.

## Install GMAT (only for the two uses above)

```bash
GMAT_CONSOLE="$(bash tools/gmat/install_gmat.sh)"   # downloads a pinned tarball, prints the GmatConsole path
```

Override `GMAT_VERSION` / `GMAT_URL` / `GMAT_DIR` via env; verify the version
against <https://sourceforge.net/projects/gmat/files/GMAT/> before trusting a run.

## Regenerating a fixture (requires GMAT)

```bash
GMAT_CONSOLE=… PYTHONPATH=tools uv run python -m gmat regenerate --out tests/golden/time_scales.json
```

A GMAT-regenerated fixture uses the harness default tolerance (`GMAT_MJD_TOL_S`,
~1 µs) — the floor of differencing float64 ModJulian columns. The committed
`time_scales.json` is instead seeded from exact published constants, so it can
declare (and lib/time meets) a far tighter 1 ns band; after a regenerate, keep
the tighter tolerances if the epochs are constant-derived.

## Drift-checking (cross-validate the committed fixture against GMAT)

```bash
# One-shot, from the CLI:
GMAT_CONSOLE=… PYTHONPATH=tools uv run python -m gmat drift-check --fixture tests/golden/time_scales.json

# Or via the skip-gated pytest test (skips cleanly when GMAT is absent):
GMAT_CONSOLE=… uv run pytest tests/tools/test_gmat_drift.py
```

The drift check compares within the *looser* of the two tolerance bands, so
GMAT's own ~µs precision — not the fixture's 1 ns claim — sets the bar. CI runs
this weekly and on demand via [`.github/workflows/golden.yml`](../../.github/workflows/golden.yml);
locally, CMake registers the same test as `golden_regen_drift` when it finds
`GmatConsole` on `PATH` (otherwise it is not registered).

`golden.py` is split so the parser, offset math, and drift comparison are
unit-tested without GMAT (`tests/tools/test_gmat_golden.py`); only the GMAT run
itself needs the binary. `propagation.py` follows the same split
(`tests/tools/test_gmat_propagation.py`).

## Propagation fixture (`tests/golden/gmat_propagation.json`)

Cross-validates the Polaris orbit propagator against GMAT's RungeKutta89 over
three force models — `two_body`, `zonal_j2` (degree 2, order 0), and
`third_body` (Sun + Moon point masses).

```bash
GMAT_CONSOLE=… PYTHONPATH=tools uv run python -m gmat regenerate-propagation \
  --out tests/golden/gmat_propagation.json --scripts-dir tools/gmat/scripts

GMAT_CONSOLE=… PYTHONPATH=tools uv run python -m gmat drift-check-propagation \
  --fixture tests/golden/gmat_propagation.json
```

`--scripts-dir` re-emits the committed `prop_*.script` files alongside the
fixture, so the provenance record cannot drift from the generator; a unit test
asserts they agree.

Three things about GMAT this harness works around:

- **Sample times overshoot.** `Propagate prop(sat) {sat.ElapsedSecs = 600}` stops
  a few hundred nanoseconds late. The fixture records GMAT's actual
  `ElapsedSecs`, so propagating to those exact times cancels the artifact.
- **Report precision.** The default `rf.Precision` of 10 digits quantises
  position to ~1 m; the scripts set 16.
- **Ephemeris source is silent when wrong.** `SolarSystem.EphemerisSource = 'DE424'`
  is required for the third-body case — GMAT falls back to DE405 without it, and
  logs nothing at the point of failure. Confirm the run log contains
  `Successfully set Planetary Source to use: DE424` (the earlier DE405 lines are
  startup noise, emitted before the script is read).

## Provenance

The current `tests/golden/time_scales.json` was seeded from **published
constants** (IERS leap seconds; the IAU definitional TT−TAI = 32.184 s and
TAI−GPS = 19 s), which are independent of the Polaris code they verify and which
GMAT reproduces exactly. It is marked `gmat_regeneratable: true`; a GMAT run
replaces the seed values in place (they will agree to the fixture tolerance).
