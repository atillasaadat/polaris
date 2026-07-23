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
- **Sensor/actuator truth models** with full error stacks; IMU emits delta-angle/delta-velocity at native rate.
- **One occlusion model for every optical sensor** (`sensors/occlusion`). Any new sensor with a line of sight — sun sensor, camera, horizon sensor — checks its keep-outs through `checkLineOfSight`, never its own geometry: two optical sensors disagreeing about whether the Earth is in the way is the kind of inconsistency that produces an estimator that works in sim and not in flight. Bodies are checked from their **limb**, not their centre.
- **Fault-injection hooks are first-class:** every sensor/actuator/subsystem model exposes scriptable fault injection (bias jumps, dropouts, occlusions, GPS outage/spoofing, stuck/runaway actuators, subsystem limits) so the FDIR integration suite (`tests/integration/`, design doc §23.1.1) can drive them per scenario. Build these hooks in from the start, not retrofitted.

## Reaction-wheel jitter (imbalance) — planned analysis

`ReactionWheel` (`sim/actuators/reaction_wheel.hpp`) already carries **static imbalance `Us`** (kg·m → radial force `Us·ω²`) and **dynamic imbalance `Ud`** (kg·m² → radial torque `Ud·ω²`), and emits both per step in the wheel frame at the rotor phase (`jitter_force_n`, `jitter_torque_nm`). This is deliberate groundwork: the near-term goal is **micro-vibration / jitter analysis**. When that work lands, plan for:
- **Harmonic content:** the fundamental is once-per-rev at the wheel speed; real wheels also show bearing/structural harmonics (integer and half-integer multiples). The current model is the fundamental only — add configurable harmonic amplitudes when needed.
- **Waterfall plots:** the standard product is a spectrogram of disturbance amplitude vs frequency vs wheel speed over a spin-up/spin-down sweep (the "waterfall"), which reveals structural resonances where a harmonic crosses a mode. Keep the disturbance outputs per-wheel and phase-resolved so this is a post-processing step over a swept run, not a model change.
- **Imbalance values are per-unit balance-report data**, not datasheet values — they default to zero in the catalog and must be filled from a unit's measured imbalance before a jitter study means anything.

## Hardware params come from YAML — there is no in-code catalog

`config/hardware/**.yaml` → the config compiler resolves the `model_id` and inlines that unit's params → `sim_setup.json` → `SimConfig::UnitConfig` → the spec's `fromParams(map)` → the model. `scenario/vehicle.cpp` is the one place that runs that last hop. There are no `catalog::` factories; they were deleted in Push 22 (design doc §19.4).

- **Never introduce a hardware or vehicle constant into C++/Python source.** It belongs in `config/`. To add a COTS unit, write the YAML entry; touch C++ only if the spec needs a new `fromParams` key.
- Every layer between the compiler and `fromParams` passes the param map through **uninterpreted** — don't add key-name knowledge to `sim_config.cpp` or `vehicle.cpp`.
- Hardcoded specs **are** fine inside `tests/` — a fixture that isolates one effect (frictionless wheel, exaggerated MTQ residual) is deliberately non-physical. A fixture mirroring a real catalog entry must say so and pin the *conversion*; the datasheet values are pinned against the YAML in `tests/tools/test_config_compiler.py`.
- Unit **names** key the per-unit random streams (`vehicle.cpp`), so they must be unique across the vehicle, and adding hardware must never perturb an existing unit's stream.

## Validation

New environment/dynamics/conversion functions are validated against **GMAT golden fixtures** (`tests/golden/`) within documented tolerances. Use the **test-vv** subagent / `/verify-golden` command.
