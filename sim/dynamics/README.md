# `sim/dynamics/` — 6DOF Plant & Integrator

The equations of motion and the numerical integrator that propagates them
(design doc §5.1, REQ-SIM-001).

| File | Role |
|---|---|
| `rigid_body.hpp` | 6DOF rigid-body state derivative: translational kinematics + Euler's rotational equation with the full inertia tensor; quaternion kinematics with per-step renormalisation |
| `force_torque.hpp` | The composable force/torque-model interface the environment models implement; `sim_runner` stacks them into one derivative |
| `integrator.{hpp,cpp}` | RK8(9) (Verner) adaptive-step integrator with absolute/relative tolerance control and a max-step bound [verner1978]; optionally records its accepted-step nodes (`StepNode`: t, y, ẏ) |
| `dense_output.hpp` | Cubic-Hermite interpolation between those nodes — how an observation is served at an arbitrary instant without stopping the integrator there |

## Dense output: observations are not events

`propagate()` takes an optional node vector; `RigidBody6Dof::stateAt(nodes, dt,
epoch)` then reconstructs the truth state anywhere inside that propagation
(position/velocity from (r, v) and (v, a), body rate from (ω, ω̇), quaternion
component-wise then renormalised). The §2.4 closed loop uses this so a sensor
sample — which reads the plant without changing it — costs no integration stop;
only genuine dynamics discontinuities (a macro boundary, the MTQ duty-window
edge) break the step. Recording the nodes costs one extra derivative evaluation
per propagation: each step's left-end derivative is its own first stage.

The interpolant is O(h⁴) local, coarser than the RK8(9) nodes it joins. That is
bounded and deliberate — `sim_dynamics_test.cpp`'s `DenseOutputMatchesTheStopped
Grid` pins agreement with a stopped grid over a 100 ms macro step at 5 °/s to
<1e-6 m and <1e-9 rad, orders below any sensor noise floor. States that must be
exact (the published macro-boundary truth) always use the integration endpoint.

## Verification approach

Layered so a failure localises (see `tests/integration/sim_runner_test.cpp`):
free drift (Newton's first law, exact) → torque-free rotation (|H| conserved)
→ two-body (energy + angular momentum to 1e-12, orbit closure after one Kepler
period to sub-mm) → the full perturbed stack against GMAT goldens (bounds, not
identities).

Determinism note: the integrator is deterministic by construction; all
stochastic behavior lives in the sensor models (§3.5), never in the plant.
