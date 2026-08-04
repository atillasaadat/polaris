# `sim/actuators/` — Actuator Truth Models

Command-in → delivered-effect-out models of the vehicle's actuators (design doc
§7). Each takes the FSW's command, applies the device's real limits and losses,
and returns the force/torque the body actually receives plus telemetry
(speed, momentum, power). Deliberately **deterministic** — friction, imbalance,
and hysteresis are physics, not noise — which is why there is no
`noise_enabled` here (§6.2).

| File | Model | Command interface |
|---|---|---|
| `reaction_wheel.{hpp,cpp}` | Torque box + momentum/speed ceiling, RW-0.4 rundown friction (dry + viscous + aero), torque quantization, copper/mechanical/regenerative power, static & dynamic imbalance (jitter groundwork), stuck/runaway faults | `commandTorque(τ)` **or** `commandSpeed(ω)` (onboard speed loop, ideal or finite-bandwidth) |
| `rw_assembly.{hpp,cpp}` | The wheel array as a **3×N W matrix** (columns = spin axes in body frame): `τ_body = W·τ_wheels`, `h_body = W·h_wheels`, degenerate-array detection | built by `scenario/vehicle` from per-wheel `spin_axis` |
| `magnetorquer.{hpp,cpp}` | Dipole limit, ±linearity, residual moment + hysteresis (play operator), and the §7 interlock pair: an exponential **post-switch-off settle transient** off the catalog's `settle_time_s`, and `dipoleNearField` — the static dipole field a rod puts on a magnetometer at its mounted location [jackson1999], which is what makes an interlock violation *visible* rather than assumed away. `injectStuckOnAt` is the deterministic stuck-on fault: the drive latched at a stated moment, since a rod that is off when `setStuckOn` fires is stuck *off* | `commandDipole(m)` for the on-window, `deenergize()` at its end, `settlingDipole(t)` through the quiet window; torque `τ=m×B` applied by the environment path |

Implements **REQ-SIM-003** (actuator truth models) and **REQ-SIM-005** (fault
injection), and the sim half of **REQ-ACTL-004** (the MTQ/MAG duty-cycle
interlock). CMGs and thrusters remain (§7); the allocation inverse `W⁺` is the
control layer's (`lib/gnc/rw_allocation`, Push 54), not a truth model.

## How to add a new actuator

Same skeleton as a sensor minus the RNG machinery (see
[`../sensors/README.md`](../sensors/README.md) for the shared steps in detail):

1. **`Spec` + `fromParams`** — datasheet-native keys → SI, one place, missing
   key ⇒ 0 ⇒ term disabled. Derived quantities (e.g. rotor inertia from
   momentum/speed) get an accessor with the fallback logic, not a second key.
2. **Command interface(s) the real device exposes.** If the drive electronics
   offer multiple modes (torque *and* speed), model both — the plant behaves
   differently in each, and "reduce it upstream" hides real hardware behavior.
3. **`step(dt)` advances the internal state** and returns an output struct:
   the delivered effect on the body (mind Newton's third law signs), telemetry
   (speed/momentum/power — regenerative sign conventions matter), and any
   disturbance outputs (imbalance jitter, phase-resolved).
4. **Saturations must be consistent**: when a limit clamps the motion, back out
   the effect actually delivered so the reported reaction matches the state
   change — a saturated device reporting its commanded effect is lying to the
   controller.
5. **Fault hooks** (§9): stuck (drive off, physics continue), runaway (full
   authority regardless of command), plus class-specific modes. `clearFaults()`.
6. **Wire + catalog + tests** — identical to the sensor steps 8–10: schema
   `kind`, `vehicle.cpp` branch (reject an unbuildable spec loudly),
   `CMakeLists.txt`, a §21.3-standard catalog YAML with pinned datasheet
   numbers, and a `tests/unit/sim_actuators_*` suite covering the physics,
   limits, faults, and (if array-mounted) the geometry consolidation.

## Placement conventions

- A wheel needs only its **`spin_axis`** (body frame, any length — normalised
  into W). The full 3×3 `mounting_dcm` is the fallback (`col(2)` = spin axis)
  and remains for devices that genuinely need an orientation.
- Names key the per-unit config identity; keep them unique across the vehicle.
