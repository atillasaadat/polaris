# `sim/io/` — The §2.4 Closed Loop

The execution model the architecture is built around, sim side: **plant →
sensors → FSW → actuators**, sim-time-driven and bit-reproducible from
`{config, seed}`.

| File | Role |
|---|---|
| `closed_loop.{hpp,cpp}` | The loop: exact integer-ns event grid from each sensor's native rate; steps the plant between *dynamics* events and serves sensor samples by dense-output interpolation (below); §2.4 buffers (IMU delta accumulation, latest-valid for discrete sensors); reads the wheel tachometers at each macro boundary for the §8.5 momentum management (an actuator that reports state, so it crosses as a measurement); fires `FswCallback` at each macro boundary; holds returned commands for the *next* interval (causality); feeds actuator output into the dynamics (`CommandedWrench`: RW reaction torques through the assembly's W, MTQ m×B); applies the GNSS jamming map + fault schedule at sample time (§9.2) |
| `sitl_server.{hpp,cpp}` | The two-process barrier (§2.2/§2.4): an `FswCallback` backed by a live `flight_PolarisFsw -s <port>` process. Sim listens on loopback; the FSW's `Drv::TcpClient` connects; payloads are the shared `lib/sitl/wire.hpp` PODs inside standard F´ frames (start word + length + CRC-32, byte-identical to `Svc::FprimeFramer`). The blocking STEP_REQ→STEP_REPLY read *is* the barrier; a protocol failure degrades to open loop (`healthy()` false) rather than crashing the run |

## Which events stop the integrator

Two kinds of thing happen on the ns grid, and only one of them is allowed to
break an integration step:

- **Dynamics events** — a macro boundary (new FSW commands apply) and the MTQ
  duty-window off edge — change the equations of motion, so they are exact
  integration stops. A step straddling the duty edge would drive the rods
  through the magnetometer's quiet window, which is the violation the §7 model
  exists to expose.
- **Sensor samples** are observations: they read the plant and change nothing.
  Stopping for them only fragments the grid — the flown config's STIM300 IMUs at
  2000 Hz cut it to 0.5 ms, and every fragment pays RK8(9)'s 16-stage minimum,
  which is what made the SITL rows cost more CPU than sim time. So the loop
  integrates straight through them and serves each sample from the propagation's
  own accepted-step nodes (`dynamics/dense_output.hpp`), in the same time order
  and therefore with the same RNG draw order as before.

Everything published — the trace, the FSW inputs, the `POLARIS_SIM_STREAM` tap —
is an exact integration endpoint, never an interpolant, and epochs stay pinned to
the integer-ns grid.

## The contract, in five tests

`tests/unit/sim_io_closed_loop_test.cpp` pins: commands apply one macro-step
late (causality); body+wheel angular momentum is conserved under internal
torques (the feedback sign convention end to end); a 10 Hz FSW receives every
250 Hz IMU sample with no loss; discrete sensors publish on their own grid;
and two identical runs produce bitwise-identical trajectories *and* noisy
measurements.

## Using it

```cpp
scenario::Vehicle vehicle;                       // buildVehicle(...)
scenario::SimRunner runner;
io::ClosedLoop loop(runner, vehicle);
runner.build(config, paths, &err, loop.wrench()); // wrench MUST be composed
loop.run([](const io::FswInputs& in) {            // the FSW (stub/script/F´)
  io::FswOutputs out; /* fill wheel + MTQ commands */ return out;
}, &trace, &err);
```

The FSW sees **measurements only** — `TruthState` never crosses the callback
(§2.3). For a real F´ process instead of a lambda, use `SitlServer` (Push 34):

```cpp
io::SitlServer::Counts counts;                    // per-type unit counts,
counts.imu = 1; counts.wheel = 4;                 // vehicle build order
io::SitlServer server(counts, /*macro_dt_ns=*/100'000'000LL);
server.start(0);                                  // 0 = OS-chosen port
// launch: flight_PolarisFsw -s <server.port()>
loop.run(server.callback(), &trace, &err);        // blocks per macro step
server.stop();                                    // sends SHUTDOWN
```

`tests/integration/sim_sitl_lockstep_test.cpp` pins the whole thing: 100
barriers against the real flight binary produce a trace **bitwise identical**
to the in-process zero-command run. `main.cpp`'s open-loop trajectory mode is
unchanged. (The flight bridge answers zero commands until the rate-group
coupling lands — next push.)

## Live truth-state stream (`POLARIS_SIM_STREAM`)

When the environment variable `POLARIS_SIM_STREAM` names a file, `ClosedLoop::run`
appends one JSON line per macro boundary — TAI epoch, ECI position/velocity, the
Body←ECI quaternion (JPL scalar-first) and the body rate — flushed per line, so
an external viewer can follow the run while it executes. The consumer is the
FreeFlyer visualization client (`python -m freeflyer viz --stream <file>
[--follow]`, `tools/freeflyer/viz.py`). Pure output: nothing reads it back, so
determinism and the sim-time clock are untouched, and it works identically under
any harness (unit rows, the SITL integration suite, a long local run).
