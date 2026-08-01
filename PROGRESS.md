# Polaris — Progress

Living status of the build-out, tracked against the design doc's development
phasing (`docs/design/Polaris_Design_Document.md` §24). This file summarizes
**what exists and is verified**; the authoritative spec and the requirements
baseline (`docs/requirements/`) remain the sources of truth. Per-push detail
lives in the merged PR descriptions and the design doc's "Implemented (Push N)"
notes — this tracker stays a rollup so it cannot rot the way a narrative does.

**Current phase:** Phase 4 — attitude determination (coarse chain closed-loop in SITL; the fine-mode MEKF now exists at the `lib/gnc` level, with its F´ component and multi-unit fusion next)
**Last updated:** Push 42 (`lib/gnc` fine mode: Davenport q-method + 6-state attitude/gyro-bias MEKF, Monte-Carlo NEES/NIS consistency)

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
| `AttitudeEstimator` F´ component: coarse chain on the 10 Hz GNC rate group, multi-unit measurement port arrays, OnboardTables sun + onboard IGRF-14 magnetic references, ParameterDb tuning with no defaults, mode/health telemetry (REQ-ADET-004) | ✅ done + tested (SITL sensor decode + flight IGRF loader unit tests; two-process lockstep run through the estimator) |
| Tuning delivery: `configc` → `Svc::PrmDb` parameter file (IDs from the FPP dictionary) → `-P` at startup → estimator configures and acquires attitude in SITL | ✅ done + tested (byte layout vs the F´ reader; full-chain SITL integration gate) |
| `lib/gnc` fine mode: Davenport q-method N-vector initializer (inverse-Fisher covariance) + 6-state MEKF (attitude error + gyro bias, closed-form Φ, Farrenkopf Qd with cross-coupling, Joseph update, NIS gating, NEES diagnostic) | ✅ done + tested (Monte-Carlo NEES = 6.01 / NIS = 2.02, bias convergence, outlier rejection, clock-fault refusal) |
| `AttitudeEstimator` fine-mode wiring (MEKF on the GNC cycle, fine↔coarse arbitration), FSW control components (control wiring, actuator commanding), sensor fusion | ⬜ Phase 4 (replace `ScriptedCmdSource`) |

**Test gates (all green):** 476 C++ lib unit (ASan/UBSan) · 9 F´ component unit ·
23 integration ·
4 GMAT golden · 103 Python, 2 skipped without GMAT (config compiler, PrmDb emitter,
GMAT harness, space weather, orbit) ·
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
| 40 | — | F´ `AttitudeEstimator` component (§8.1, §10; REQ-ADET-004): the Push 39 coarse chain running on the vehicle, as member 0 of the barrier-driven 10 Hz GNC rate group (§2.4) so estimation precedes anything acting on the estimate. New interface-only `GncPorts` module defines the measurement/estimate seam — `ImuMeas`/`SunSensorMeas`/`MagnetometerMeas`/`GnssMeas`/`StarTrackerMeas`/`AttitudeEstimate` — as **port arrays** (`GncMaxUnits = 8`) so the multi-sensor vehicle and the §8.2 fusion layer never need a port-interface rework; this push consumes the first valid, fresh unit of each type (deterministic build-order priority, not fusion) and leaves the star-tracker input latched-but-unfused so coarse mode stays tracker-independent (§10). `SitlBridge` now decodes the per-unit sensor records the STEP_REQ always carried and republishes them on that seam before cycling the rate group; a length that does not match the HELLO-declared suite is rejected whole, leaving the previous measurements intact. Sun reference from `OnboardTables.getBodyPosition` with its source grade carried through; magnetic reference from the **onboard IGRF-14** at the GNSS position, rotated ECEF→ECI with onboard EOP — which moved the IAGA parser into the flight-safe `lib/environment/igrf_iaga` (`<cstdio>`, fixed buffers, load-time only) with `sim/world/igrf_file.cpp` now a façade over it, and `decimalYear` into `lib/time/utc`, so sim and FSW share one parser and one epoch convention. Twelve tuning values are **F´ parameters with no defaults** (§19.3), including the §9.1 staleness window and the geocentric-radius band a GNSS fix must fall in: missing/out-of-range emits edge-gated `ConfigInvalid` and refuses the cycle rather than inventing a noise budget. Position is GNSS-only until §8.3, flagged by an edge-gated `PositionUnavailable` warning rather than a guessed position. Health telemetry (mode, quaternion, rate, covariance trace, age, TRIAD accept/reject, refused cycles, reference grade, per-source validity) + `AttitudeAcquired`/`AttitudeLost`/`ReferenceDegraded`/`ReferenceRecovered`/`MagneticReferenceStale` EVRs, the grade alerts carrying which domain (ephemeris vs EOP) degraded. 5 new lib unit tests (STEP_REQ sensor decode incl. positional unit identity and rejected-message atomicity; the flight IGRF loader against the sim façade coefficient-for-coefficient) plus the deployment's **first F´ component GTest harness** — 9 tests covering acquisition of a known attitude through the real reference assembly, eclipse coast + whole re-acquisition, parameter refusal, the §9.1 staleness and position-range gates, grade passthrough/alerting, position loss, an expired IGRF snapshot, and `RESET_ESTIMATOR` re-arming every edge-gated alert. Writing it uncovered that `cmake/Dependencies.cmake` had been forcing `BUILD_TESTING` off repo-wide (to suppress Eigen's test tree), which silently disabled `register_fprime_ut` for every F´ component; it now saves and restores the flag |
| 41 | — | Tuning delivery — the `configc` → `ParameterDb` path (§19.3), which is what turns Push 40's estimator from a component that refuses to run into a closed attitude loop. FSW tuning now lives in the vehicle config as `spacecraft.fsw_parameters`, a flat map keyed by **fully-qualified F´ parameter name**; the compiler emits `PrmDb.dat` in the `Svc::PrmDb` on-disk format (CRC-32 header + `0xA5`/record-size/ID/value records, F´ big-endian serialization) and the deployment's new `-P` option re-points `prmDb` at it. Parameter **IDs are read from the FPP-generated topology dictionary**, never hand-written, so a base-ID move or a reordered `param` declaration cannot silently desynchronise a delivered file from the flight build; validation runs both ways (an unknown name and an unset declared parameter are both compile failures), because with no flight defaults an incomplete file ships a component that refuses to fly. The reference vehicle's twelve estimator values are **derived from the units it carries** — GomSpace FSS field-edge accuracy plus a 2° albedo term and the ~0.4° analytic-ephemeris floor for the sun pair, generic-magnetometer noise against a ~30 µT LEO field plus its uncalibrated 1 µT hard-iron bias for the magnetic pair, STIM300 angle random walk for the gyro — so the reported covariance is the one the hardware justifies. Wiring this uncovered that the deployment's hand-rolled `setupTopology` had never called the autocoded `readParameters()` phase, so `prmDb` had been loading nothing at all regardless of the file it was configured with. New end-to-end gate `tests/integration/sitl_attitude_tuning_test.cpp` runs the whole chain against a live SITL session (`leo_smallsat.yaml` → `configc` → `PrmDb.dat` → `prmDb` → `attitudeEstimator`) and asserts on the deployment's own event stream: all twelve records loaded, no `ConfigInvalid`, `AttitudeAcquired` on the first cycle with a sun/magnetometer pair (cov trace 4.8e-3 rad², i.e. ~2.3° per axis — the systematic floor the two vector sources justify, as designed). 17 Python tests pin the byte layout against the F´ v4.2.2 reader rather than against the emitter's own helpers, plus NaN/inf, out-of-range and type-coercion refusals |
| 42 | — | Fine mode in the flight-safe `lib/gnc` (§8.1; REQ-ADET-001, REQ-ADET-004). `gnc::davenport` — Davenport's q-method [davenport1968][markley2014] solving Wahba's problem for **N** weighted vector pairs (TRIAD handles exactly two), the 4×4 K matrix built scalar-first to match the JPL layout and solved with Eigen's fixed-size `SelfAdjointEigenSolver`: a bounded iteration budget with a checkable `info()` status, which is why the q-method rather than QUEST's characteristic-polynomial Newton iteration. Maximum-likelihood weights `wᵢ = 1/σᵢ²` make the reported covariance the **inverse Fisher information** [shuster1981], and the Wahba loss at the optimum comes out as a mutual-disagreement diagnostic that geometry gating alone cannot catch. Observability (`λ_min/λ_max` of the information matrix) is gated in **both** frames; collinear sets, malformed inputs and non-convergence are refusals, never asserts. `gnc::Mekf` — the 6-state multiplicative EKF [lefferts1982][markley2014 §6.2.4]: `[δθ; δb]` in the canonical `ErrorState` block order, closed-form quaternion propagation with the **exact** `Φ₁₂ = −∫exp(−[ω̂×]s)ds` cross block, Farrenkopf discrete `Qd` with its `−½σ_u²Δt²` cross-coupling (the diagonal shortcut passes every convergence test and never learns the bias), vector measurements processed **one at a time** so §8.2 N-sensor fusion is just more calls, `H = [[b̂_pred×] 0₃]`, Joseph-form covariance update, exact axis-angle multiplicative reset, covariance symmetrised after every step. Measurement noise is a **per-update caller-supplied** `σ` (`R = σ²I`); the header states plainly that the filter treats it as white, so fusing a systematic budget unadjusted makes it overconfident. **NIS is exposed per update and gates the measurement** (χ²₂ — the innovation between two unit vectors is transverse, so the effective DOF is 2, not 3) with a rejection count for FDIR; `nees()` gives the analysis-side 6-state consistency check. Seeded from Davenport/TRIAD as **cold-start/re-init only**, never in the steady-state loop. Past the coast horizon the attitude reads invalid but the solution is *kept* — unlike the coarse mode, a Kalman gain against a grown covariance takes the returning measurement almost whole, so there is nothing to gain by discarding a converged bias. Written to the canonical `EstimatedState` with the attitude, gyro-bias **and cross** covariance blocks, mode `Fine`/`Invalid`. 19 unit tests: exact N-pair recovery and 20 000-sample whitened Monte-Carlo covariance for the q-method (sample NEES = 3) plus a mean-square win over TRIAD on the same noisy two-vector data; for the MEKF, exact propagation on a perfect gyro, bias convergence to 10% of an injected 0.06 °/s truth, cold-start convergence under noise, **150-run Monte-Carlo consistency (mean NEES 6.01 against 6, mean NIS 2.02 against 2)**, outlier rejection leaving state and covariance bit-unchanged, dead-gyro/overlong-gap holding, stuck- and backwards-clock refusal proven by bit-identity against an unfaulted filter, malformed-input refusal, a non-positive-definite seed covariance refused at the `initialize` trust boundary (an indefinite P makes S indefinite and a negative NIS would sail through a one-sided gate, so the gate is written as the accept range `0 ≤ NIS ≤ gate`), and a direct algebraic check of one propagation step against a hand-computed `ΦPΦᵀ + Qd` — the only test that can see Qd's cross blocks, which sit six orders of magnitude below `Q11` where no statistical test resolves them, and which also pins that `Φ₁₂` is **zero** on a step where the gyro was not used |

---

## What's next

1. **Phase 4 remainder:** the MEKF on the F´ `AttitudeEstimator` (fine mode on
   the GNC cycle, its tuning through `configc`/`ParameterDb`), multi-IMU/
   multi-sun-sensor fusion (§8.2), and the fine↔coarse arbitration surfaced to
   FDIR.
2. **Phase 2 close-out (optional):** CMG and thruster truth models (§7) — or
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

# FSW tuning: compile the vehicle config into a Svc::PrmDb parameter file and
# run the deployment against it (the flight build must exist first — the
# parameter IDs come from the generated topology dictionary)
PYTHONPATH=tools uv run python -m configc \
    --config config/spacecraft/leo_smallsat.yaml --hardware config/hardware \
    --dictionary build-artifacts/Linux/flight_PolarisFsw/dict/PolarisFswTopologyDictionary.json \
    --out build/config
./build-artifacts/Linux/flight_PolarisFsw/bin/flight_PolarisFsw -P build/config/PrmDb.dat -s 50051

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
