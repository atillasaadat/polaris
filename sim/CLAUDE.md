# CLAUDE.md — `sim/` (Truth / Environment Simulation, C++)

The simulation is the **plant** the FSW runs against. It is **not flight code** — the no-heap / no-exception flight rules do **not** apply here. But it has its own discipline. Read the root `CLAUDE.md` first.

## What's different from `flight/`

- **Heap and standard containers are fine.** Use Eigen freely (dynamic sizes OK). Prioritize fidelity and clarity over flight constraints.
- The sim may use **SPICE/NAIF kernels** (DE440/DE441) directly — these live on the ground/sim side only, never onboard.

## What's still strict

- **Determinism is mandatory.** Everything stochastic draws from an explicitly **seeded** RNG stream (per-source seed derivation). A run must be **bit-reproducible from `{config, seed}`** — including across the two-process boundary. Never seed from wall clock or `std::random_device` in a run path.
- **Sim time is the master clock.** Execution is sim-time-driven and lockstepped with the FSW over the F´ TCP transport (macro-step handshake). Never pace simulation logic off wall clock except in the optional real-time throttle.
- **Truth must differ from onboard models, deliberately.** Truth gravity degree, ephemeris fidelity, sensor biases/noise/latency are set to exercise the estimators. Don't accidentally hand the FSW a truth-fidelity quantity.
- **`TruthState`** is the canonical truth product (`lib/state/`). Keep it on the sim side of the boundary; the FSW never receives it.
- SI units, frame-tagged vectors, and reference provenance (textbook/paper in `refs.bib`) apply here too.

## Core content

- **6DOF dynamics + RK89** (configurable tolerance/step). Energy/momentum conservation checks available as diagnostics.
- **Environment:** EGM2008 (settable degree/order) + tides, third-body via SPICE, **NRLMSIS 2.1** drag, SRP + conical eclipse, **IGRF-14** (WMM backup), disturbance torques.
- **Sensor/actuator truth models** with full error stacks; shared **line-of-sight occlusion** model (Earth limb / Sun / Moon) for optical sensors; IMU emits delta-angle/delta-velocity at native rate.
- **Fault-injection hooks are first-class:** every sensor/actuator/subsystem model exposes scriptable fault injection (bias jumps, dropouts, occlusions, GPS outage/spoofing, stuck/runaway actuators, subsystem limits) so the FDIR integration suite (`tests/integration/`, design doc §23.1.1) can drive them per scenario. Build these hooks in from the start, not retrofitted.

## Validation

New environment/dynamics/conversion functions are validated against **GMAT golden fixtures** (`tests/golden/`) within documented tolerances. Use the **test-vv** subagent / `/verify-golden` command.
