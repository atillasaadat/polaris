# `sim/dynamics/` — 6DOF Plant & Integrator

The equations of motion and the numerical integrator that propagates them
(design doc §5.1, REQ-SIM-001).

| File | Role |
|---|---|
| `rigid_body.hpp` | 6DOF rigid-body state derivative: translational kinematics + Euler's rotational equation with the full inertia tensor; quaternion kinematics with per-step renormalisation |
| `force_torque.hpp` | The composable force/torque-model interface the environment models implement; `sim_runner` stacks them into one derivative |
| `integrator.{hpp,cpp}` | RK8(9) (Verner) adaptive-step integrator with absolute/relative tolerance control and a max-step bound [verner1978] |

## Verification approach

Layered so a failure localises (see `tests/integration/sim_runner_test.cpp`):
free drift (Newton's first law, exact) → torque-free rotation (|H| conserved)
→ two-body (energy + angular momentum to 1e-12, orbit closure after one Kepler
period to sub-mm) → the full perturbed stack against GMAT goldens (bounds, not
identities).

Determinism note: the integrator is deterministic by construction; all
stochastic behavior lives in the sensor models (§3.5), never in the plant.
