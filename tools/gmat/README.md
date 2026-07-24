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

Cross-validates the Polaris orbit propagator, force composite, and attitude
kinematics against GMAT's RungeKutta89. Eight cases:

| Case | What it stresses | Band (achieved) |
|---|---|---|
| `two_body` | integrator truncation only (Earth.Mu pinned to wgs84::kGM) | 5 cm (3.5 mm) |
| `zonal_j2` | degree-2 field, EGM96↔EGM2008 | 0.5 m (4.7 cm) |
| `third_body` | Sun+Moon point mass, DE424↔DE440 | 0.1 m (3.9 mm) |
| `iss_leo` | 417 km / 51.6°, degree-8 field | 2 m (0.88 m) |
| `sso_leo` | 700 km / 98.2° sun-sync, degree-8 field | 2 m (0.76 m) |
| `geo` | 42,164 km, degree-4 + Sun/Moon + SRP, 1 day | 150 m (120 m) |
| `molniya_heo` | 600×39,800 km, e=0.74, 63.4° — step-control stress | 25 m (6.3 m) |
| `attitude_spinner` | torque-free spinner vs GMAT Spinner (quaternion kinematics) | 1e-3° |

The bands are model-difference budgets (EGM96↔EGM2008, DE424↔DE440, cannonball↔
spherical SRP) or, for `molniya_heo`, adaptive-step divergence on the
high-eccentricity arc — **not** propagator error, which sits well below.
`attitude_spinner` compares the net rotation angle (a quaternion-convention-
independent invariant) between Polaris, GMAT, and the analytic `|ω|·t`.

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
