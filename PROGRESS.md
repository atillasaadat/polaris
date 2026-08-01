# Polaris — Progress

Living status of the build-out, tracked against the design doc's development
phasing (`docs/design/Polaris_Design_Document.md` §24). This file summarizes
**what exists and is verified**; the authoritative spec and the requirements
baseline (`docs/requirements/`) remain the sources of truth. Per-push detail
lives in the merged PR descriptions and the design doc's "Implemented (Push N)"
notes — this tracker stays a rollup so it cannot rot the way a narrative does.

**Current phase:** Phase 4 — attitude determination (the `lib/gnc` coarse chain is in; the F´ estimator component that wires it to the SITL rate group is next)
**Last updated:** Push 39 (coarse attitude chain in `lib/gnc`: TRIAD with Shuster covariance + the SS+MAG+IMU coarse estimator that stands behind the Safe-mode sun-pointing floor)

---

## Status at a glance

| Area | State |
|---|---|
| Repo skeleton, licensing, tooling, CI (lint + build + F´ tests + docs gates) | ✅ done; branch protection + squash auto-merge on `main` |
| Requirements ICD (sphinx-needs, 71 reqs) + traceability gate (`-W`) | ✅ builds clean; uncovered baselined reqs fail CI |
| Docs site (Sphinx + Doxygen/Breathe + bibtex) | ✅ published: <https://atillasaadat.github.io/polaris/> |
| F´ v4.2.2 barebones deployment (buildable/runnable) | ✅ done |
| `lib/` foundations: constants, typed vectors/quaternions, frames (LVLH/RIC), canonical state structs | ✅ done + tested |
| `lib/` time: TAI/GPS/TT/UTC/TDB, leap seconds | ✅ done + tested |
| `lib/` ECI↔ECEF (IAU 2006/2000A via ERFA) + IERS EOP | ✅ done + GMAT-validated |
| `lib/` onboard Chebyshev ephemeris + IGRF-14 | ✅ done + tested |
| Config pipeline: pydantic schema → compiler → `sim_setup.json` → C++ loader | ✅ done; single source of truth, provenance-hashed |
| Hardware model library (`config/hardware/`, 17 entries, datasheet-pinned) | ✅ done; no in-code catalogs (§19.4) |
| Truth dynamics: 6DOF rigid body + RK8(9) | ✅ done; conservation + Kepler-closure tested |
| Environment: EGM2008 (to 200×200), third-body (DE440), drag (exponential + NRLMSIS 2.1 + space weather), SRP + conical eclipse, IGRF-14 + dipole torque | ✅ done + GMAT golden cross-validation |
| Truth-sim executable (`sim/main.cpp`) | ✅ done |
| Sensors: IMU, magnetometer, star tracker, sun sensor (analogue + digital), GNSS | ✅ done + tested (full error stacks, occlusion, fault hooks) |
| Actuators: reaction wheel (torque + speed modes, W-matrix assembly), magnetorquer | ✅ done + tested |
| Scenario controls: seed, sensor-noise switches (global + per-unit), GNSS jamming KML + fault schedule | ✅ done + tested |
| CMGs, thrusters (truth models) | ⬜ remaining in Phase 2 scope (§7) |
| §2.4 closed loop (`sim/io`): sensors sampled at native rates, actuator feedback into the plant, fault bindings live | ✅ done + tested |
| §2.2 SITL two-process transport: `lib/sitl` wire + `SitlBridge` F´ component + sim TCP barrier | ✅ done + tested (bit-identical two-process trace) |
| §2.4 barrier-driven FSW cycle: SITL `PassiveRateGroup` + `SitlTime` sim-time source + real command path (placeholder `ScriptedCmdSource`) | ✅ done + tested (scripted profile crosses the wire → bit-identical to in-process) |
| §2.2 SITL packaged as the `PolarisSitl` subtopology (excludable for a flight build; dictionary byte-identical) | ✅ done + tested (both two-process gates stay bitwise green) |
| Onboard tables (`OnboardTables` flight component): leap/EOP/Chebyshev loaded + validated + served via typed ports; `RELOAD_TABLES` upload→activate; coverage-expiry check | ✅ done + tested (`lib/onboard` double-buffer swap; port answers vs lib evaluators) |
| `lib/gnc` coarse attitude chain: TRIAD + Shuster covariance, SS+MAG+IMU coarse estimator (gyro propagation, eclipse coasting, coast-horizon invalidation, systematic covariance floor, canonical-state output) | ✅ done + tested (exact recovery, Monte-Carlo NEES, degeneracy rejection, floor + clock-fault refusal) |
| FSW GNC components (estimator/control wiring), MEKF fine mode, sensor fusion | ⬜ Phase 4 (replace `ScriptedCmdSource`) |

**Test gates (all green):** 451 C++ unit (ASan/UBSan) · 22 integration ·
4 GMAT golden · 84 Python (config compiler, GMAT harness, space weather, orbit) ·
docs `-W` (bibliography + requirements traceability) · pre-commit
(clang-format + ruff) · F´ flight build.

---

## Push log

Phase 0 — Foundations
| Push | PR | Delivered |
|---|---|---|
| 1–2a | #1–2 | Repo/CI/licensing skeleton; requirements ICD (sphinx-needs); docs site + Pages gating; `lib/` constants + typed vectors + JPL quaternion |
| 2b | #3 | `lib/time` (TAI/GPS/TT/UTC, leap seconds); F´ v4.2.2 barebones deployment (Push 4 pulled forward) |
| 2c–2d | #4 | Geometric frames (LVLH/RIC) + canonical `EstimatedState`/`TruthState`; onboard Chebyshev ephemeris + TT↔TDB |
| 3 | #5 | Config schema + hardware library + config compiler (§19.3) |
| 5 | #6 | GMAT golden-data harness + first fixture |

Phase 1 — Truth sim core
| Push | PR | Delivered |
|---|---|---|
| 6 | #7 | 6DOF rigid-body dynamics + RK8(9) integrator |
| 7 | #8 | Spherical-harmonic gravity + gravity-gradient torque |
| 8 | #9 | Fully-normalized singularity-free gravity (stable to 200×200) |
| 9 | #10 | ECI↔ECEF reduction (IAU 2006/2000A) + IERS EOP tables |
| 10–11 | #11–13 | EGM2008 `.gfc` loader (tesseral, ECEF-frame); third-body gravity; degree-200 golden fixture |
| 12 | #14 | Solar radiation pressure + conical eclipse |
| 13 | #15 | Atmospheric drag + exponential atmosphere |
| 14 | #16 | DE440 Chebyshev ephemeris layer (Sun/Moon) |
| 15 | #17 | IGRF-14 geomagnetic field + residual-dipole torque |
| 16 | #18 | Truth-sim executable |
| 17 | #19 | GMAT force-model golden cross-validation |
| 18 | #20 | CelesTrak space-weather layer for NRLMSIS |

Phase 2 — Sensor & actuator models
| Push | PR | Delivered |
|---|---|---|
| 19 | #21 | Sensor truth-model foundation (§6.1 error stack) + magnetometer |
| 20 | #22 | IMU truth model + datasheet catalog entries (STIM300/377H) |
| 21 | #23 | Reaction wheel (RW-0.4 rundown model) + magnetorquer (hysteresis) |
| 22 | #24 | Config-driven hardware — in-code catalogs deleted (§19.4) |
| 23 | #25 | Shared line-of-sight occlusion (+ configurable atmosphere limb) + Sodern AURIGA star tracker (acquisition/tracking state machine) |
| 24 | #26 | Sun sensor (analogue counts + GomSpace FSS digital vector) + magnetometer config wiring |
| 25 | #27 | GNSS PVT-fix receiver (NovAtel OEM7600) + geographic jamming KML + scheduled fault events |
| 26 | #28 | RW assembly (W-matrix, `spin_axis` config) + wheel-local speed command mode |
| 27 | #29 | Sensor-noise master switch + per-unit overrides |
| 28 | #30 | Holistic audit: config-pipeline fixes (`com_m`, `gravity_order`, defaults, validation), §-reference sweep, refs.bib completion, documentation standard (§21.3) + per-folder READMEs |
| 29 | #31 | Planetary third-body perturbers config-selectable (mercury/venus/mars/jupiter/saturn/uranus/neptune, case-insensitive; DE440 barycenters/system GMs); ephemeris fixture regenerated with all nine bodies |
| 30 | #32 | §2.4 closed loop (`sim/io`): native-rate sensor sampling, IMU delta-accumulation, actuator wrench feedback into the plant, live fault bindings; full-stack 6DOF orbit+attitude integration test |
| 31 | #33 | GMAT orbit-regime matrix (ISS/SSO/GEO/Molniya) + attitude-spinner cross-validation; model-difference tolerance budgets |
| 32 | #34 | Docs overhaul: sectioned API reference (per-namespace pages grouped `lib`/`sim`/tools) + MyST user guides reusing per-folder READMEs; frame-safe quaternion + closed-loop docstring examples |
| 33 | #35 | Math documentation standard (§21.3): implementation-exact LaTeX equation blocks on every model class (sensors, actuators, dynamics, environment, quaternion) rendered via Doxygen→Breathe→MathJax; source-controlled Mermaid diagrams in guides (§2.4 sequence, config pipeline, frame graph, architecture) |
| 34 | — | Phase 3 kickoff — §2.2 SITL transport: shared `lib/sitl` wire format (frozen v1, measurements-only §2.3 boundary), `SitlBridge` passive F´ component on a dedicated comm stack (`-s <port>`, inert when off), sim TCP barrier server with timeout/degrade; two-process integration gate: 100 barriers → bit-identical trace |
| 35 | — | §2.4 barrier drives the real FSW cycle: `SitlBridge` fires a SITL `Svc::PassiveRateGroup` synchronously per STEP; new `SitlTime` serves sim time as the FSW clock (wall clock when SITL off); `SitlHandler` split into decode + caller-supplied-command reply; placeholder `ScriptedCmdSource` commands actuators from a shared deterministic profile (`-c` to enable). Second integration gate: scripted profile over the wire → bit-identical to in-process |
| 36 | — | SITL packaged as the `PolarisSitl` subtopology (`flight/PolarisFsw/PolarisSitl/`): nine instances + internal wiring moved out of the flat topology behind `import PolarisSitl.Subtopology`, base IDs preserved so the dictionary is byte-identical; `SitlTime` kept in the main topology as the deployment-wide time source. Deliver-without-SITL is a documented topology-edit recipe (FPP has no conditional-import switch), not a CMake option; passive SITL components register no health pings per the §health active-only guidance. Both two-process gates stay bitwise green |
| 37 | — | Onboard tables (§11.3, §22): new `OnboardTables` flight component wrapping the flight-safe `lib/onboard::TableStore`. Loads leap seconds (in-code `historical()`), IERS EOP (`finals.all`, windowed to the ephemeris span), and Sun/Moon Chebyshev fits (committed `.cheb`, planets skipped) at setup; serves `getEopAt`/`getBodyPosition`/`getTaiUtcOffset` typed ports over the lib evaluators. `RELOAD_TABLES` command is the upload→activate path (FileUplink writes the file, reload restages into an inactive double-buffer slot and swaps only on full success — failed reload keeps the previous tables); `Svc.Sched` coverage-expiry warning; health telemetry (counts/spans/state). Flight component (no `lib/sitl` dep, outside `PolarisSitl`); `<cstdio>` reads at load/reload only. 7 unit tests vs lib ground truth on the committed fixtures |
| 38 | — | Coarse fallbacks + source-quality grading (§8.1, §11.3): table loss degrades to coarse operation instead of no answer. New table-independent analytic ephemerides (`lib/ephemeris/analytic_{sun,moon}`, Vallado §5.1 Alg 29 / §5.3.2 — pure functions of the clock, no data dependency) and a zero-EOP fallback (UT1 ≈ UTC ⇒ `UT1−TAI = −ΔAT`, zero polar motion). Every `TableStore` query returns a grade (`kPrecise`/`kCoarse`/`kUnavailable`); ΔAT stays precise always (in-code leap record, load-independent). Grade surfaced on the F´ ports (`grade` on `EopSample`/`PosEciMeters`) and per-domain telemetry; `TableDegraded`/`TableRecovered` EVRs on precise↔coarse transitions; the watchdog reports the served grade. This is the mechanism that makes the §10 Safe-mode coarse sun-pointing floor star-tracker- and table-independent. Analytic vs DE440 agree ≤ 0.37°/0.48° (Sun/Moon) across the fixture; +2 test files/tests |
| 39 | — | Phase 4 kickoff — coarse attitude chain in the flight-safe `lib/gnc` (§8.1): `gnc::triad` deterministic two-vector initializer (sun primary, mag secondary) with **Shuster's TRIAD covariance** on the body-frame `δθ` error [black1964][markley2014][shuster1981], degenerate geometry gated in both frames and reported as invalid rather than asserted; `gnc::CoarseAttitudeEstimator` — closed-form gyro propagation, TRIAD acquisition/update, fixed-gain eigenaxis complementary blend (not a Kalman update: the safe-mode floor wants always-available, not optimal), covariance propagated and blended over a **systematic floor** (each source's σ splits into a white part the blend may reduce and a systematic part — ephemeris/IGRF/alignment — that never averages down, so the reported uncertainty converges to the floor instead of to zero). Eclipse coasts on the gyro with a growing covariance and a reported age; past the coast horizon the attitude goes **invalid** and the next TRIAD re-acquires whole. Overlong gaps are dropouts, not extrapolation; non-increasing clocks (backwards *and* stuck) and non-finite measurements are refused without destroying the solution, and a refused cycle publishes a default-constructed product rather than a half-written one. Output written to the canonical `EstimatedState` (attitude block only, and only when valid). 25 unit tests: exact recovery over random attitudes, covariance pinned analytically in the triad basis + for budget linearity + by whitened 20 000-sample Monte Carlo (NEES = 3), degeneracy/malformed-input rejection, 30 s eclipse propagation, coast invalidation + re-acquisition, floor-bounded blend convergence, clock-fault refusal, `q0 ≥ 0` and positive-definiteness throughout |

---

## What's next

1. **F´ `AttitudeEstimator` component:** wrap the `lib/gnc` coarse chain
   (Push 39) in a real GNC component on the SITL rate group, replacing the
   placeholder `ScriptedCmdSource`. It reads the STEP_REQ sensor records (still
   unread today), pulls its inertial sun reference and EOP from the
   `OnboardTables` query ports (Push 37/38) and gates on the returned
   source-quality grade, and publishes `EstimatedState` with the estimation mode
   telemetered (REQ-ADET-004).
2. **Phase 4 remainder:** MEKF fine mode, QUEST, multi-IMU/multi-sun-sensor
   fusion (§8.2), and the fine↔coarse arbitration surfaced to FDIR.
3. **Phase 2 close-out (optional):** CMG and thruster truth models (§7) — or
   defer to the phases that consume them (§8.5 control, §17 maneuvering).

---

## Build & verify locally

Dependencies are managed with [uv](https://docs.astral.sh/uv/); `uv sync`
creates `.venv`, provisions Python 3.12, and installs the pinned toolchain
(`cmake`/`ninja` as wheels). Only `doxygen` is a system package.

```bash
# One-time
uv sync                             # F´ toolchain + dev tools (from uv.lock)

# F´ flight build
uv run fprime-util generate && uv run fprime-util build

# C++ test suites (unit runs under ASan/UBSan)
uv run cmake --build build-fprime-automatic-native-ut \
    --target polaris_unit_tests polaris_integration_tests polaris_golden_tests -j4
./build-fprime-automatic-native-ut/bin/Linux/polaris_unit_tests
./build-fprime-automatic-native-ut/bin/Linux/polaris_integration_tests
./build-fprime-automatic-native-ut/bin/Linux/polaris_golden_tests

# Python suites (config compiler, GMAT harness, space weather, orbit)
uv run pytest tests/tools

# Docs site (warnings are errors; same gate as CI)
sudo apt-get install -y doxygen     # one-time
uv run --only-group docs bash tools/dev/build_docs.sh   # -> docs/_build/html/index.html

# Lint (CI runs exactly this; the pre-commit git hook runs it per-commit)
uv run --only-group dev pre-commit run --all-files
```
