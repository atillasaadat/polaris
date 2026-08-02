# `sim/` — Truth / Environment Simulation

The **plant** the flight software runs against (design doc §5–§7): a
high-fidelity 6DOF simulation of the vehicle and its environment, with sensor
and actuator truth models built from the compiled config. Not flight code — the
no-heap/no-exception flight rules do not apply here — but determinism is
mandatory: every run is **bit-reproducible from `{config, seed}`** (§3.5).

Agent/developer operating rules live in [`CLAUDE.md`](CLAUDE.md); this README is
the orientation map.

| Directory | What it holds |
|---|---|
| [`dynamics/`](dynamics/README.md) | 6DOF rigid-body equations of motion + the RK8(9) adaptive integrator |
| [`world/`](world/README.md) | Environment models: EGM2008 gravity, third-body, drag (exponential/NRLMSIS + space weather), SRP + eclipse, IGRF, the §5.3 disturbance torques (gravity-gradient, residual dipole), and the loaders for their committed reference data |
| [`sensors/`](sensors/README.md) | Sensor truth models (IMU, star tracker, sun sensor, magnetometer, GNSS + jamming regions, generic payload sensor) + shared error stack and occlusion — **start here to add a sensor** |
| [`actuators/`](actuators/README.md) | Actuator truth models (reaction wheel, magnetorquer) + the W-matrix wheel assembly — **start here to add an actuator** |
| [`scenario/`](scenario/README.md) | The config→models bridge: `sim_setup.json` loader, vehicle builder, GNSS fault schedule, and the sim runner |
| [`io/`](io/README.md) | The §2.4 closed loop: plant → sensors → FSW callback → actuators, sim-time-driven and bit-reproducible. The F´ SITL transport (Phase 3) drives the same callback over TCP |
| `main.cpp` | The truth-sim executable: compiled config in, trajectory out |

## The one-paragraph data flow

`config/spacecraft/*.yaml` + `config/hardware/**.yaml` → `tools/configc` compiles
and provenance-hashes → `sim_setup.json` → `scenario/sim_config` parses →
`scenario/vehicle` builds every sensor/actuator model from its params
(`fromParams`, seeded per unit **by instance name**) → `scenario/sim_runner`
assembles the force/torque stack from `world/` + `dynamics/` and propagates.
`io/closed_loop` binds it all together: sensors sampled at native rates into
§2.4 buffers, FSW commands (a callback until the F´ SITL lands) fed back into
the plant as wheel reaction torques and magnetorquer m×B.

## House rules (the short version)

- **No hardware constants in code** — catalog YAML only (§19.4).
- **One occlusion model** for every optical line of sight (`sensors/occlusion`).
- **Model a unit at the interface it presents** — analogue parts emit counts,
  digital parts emit processed outputs at datasheet accuracy.
- **Fixed RNG draw counts** per sample call; unit streams keyed by instance
  name so adding hardware never perturbs existing streams (§3.5).
- **Fault-injection hooks are first-class** on every model (§9).

Full rationale for each rule: [`CLAUDE.md`](CLAUDE.md) and design doc §5–§7, §19.
