# `lib/state/` — Canonical State Structs

The single nav product (design doc §8.0, REQ-SYS-005): produced once, consumed
everywhere — no per-component ad-hoc state passing.

| File | Role |
|---|---|
| `estimated_state.hpp` | The onboard product: TAI epoch, `Quat<Body,ECI>`, rates, ECI pos/vel, gyro/accel biases, 15-state covariance (documented block order), per-field validity, active estimation mode, schema version |
| `truth_state.hpp` | The sim-only analogue without covariance/validity — truth-vs-onboard is a **type distinction**, so flight code is structurally unable to consume truth (§2.3) |

Plain aggregates, no heap/exceptions; every vector frame-tagged.
