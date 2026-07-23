# `sim/scenario/` — Config → Running Sim

The bridge from the compiled configuration to executing models (design doc
§19.3/§19.4, REQ-CFG-001/002). Everything downstream of `tools/configc` and
upstream of the physics lives here.

| File | Role |
|---|---|
| `sim_config.{hpp,cpp}` | Parses `sim_setup.json` into plain config structs (`SimConfig`, `SpacecraftConfig`, `EnvironmentConfig`, `UnitConfig`, `GnssFaultEvent`). Validates at the boundary — malformed artifacts fail here, not downstream. Param maps pass through **uninterpreted**. |
| `vehicle.{hpp,cpp}` | Builds the installed hardware suite: each unit's params → its model's `fromParams`, seeded by FNV-1a(instance name) under the master seed (§3.5), noise switches applied (global + per-unit), wheel spin axes consolidated into the `RwAssembly` W matrix. The **only** place a catalog number becomes a model. |
| `gnss_faults.{hpp,cpp}` | Resolves the scenario's time-windowed GNSS fault schedule (outage/spoof/clock-jump) to the state active at time *t* and reconciles a receiver's fault hooks to it (idempotent; window end clears the fault). |
| `sim_runner.{hpp,cpp}` | Assembles the force/torque stack (gravity, third-body, drag, SRP, magnetic) from the environment config + committed reference data, and propagates the 6DOF plant to a trajectory. |

**Not yet here:** the §2.4 macro-step loop that samples sensors, applies the
fault schedule per step, and feeds actuator torques back into the plant — that
arrives with `sim/io` (Phase 3). Until then `vehicle`/`gnss_faults` are built
and unit-tested but not driven by `sim_runner`.

## Invariants worth knowing

- **Unit names are load-bearing:** they key the per-unit RNG streams, so they
  must be unique per vehicle, and renaming a unit re-rolls its realised errors.
- **A unit that would build silently-perfect is rejected** (no FOV, no rotor
  inertia, no position accuracy…) — a config error is better than an
  implausibly good sensor.
- **Keplerian→Cartesian happens in the compiler**, not here; the C++ side has
  no orbital-element code to drift from the Python.
