# `sim/io/` — The §2.4 Closed Loop

The execution model the architecture is built around, sim side: **plant →
sensors → FSW → actuators**, sim-time-driven and bit-reproducible from
`{config, seed}`.

| File | Role |
|---|---|
| `closed_loop.{hpp,cpp}` | The loop: exact integer-ns event grid from each sensor's native rate; micro-steps the plant between events; §2.4 buffers (IMU delta accumulation, latest-valid for discrete sensors); fires `FswCallback` at each macro boundary; holds returned commands for the *next* interval (causality); feeds actuator output into the dynamics (`CommandedWrench`: RW reaction torques through the assembly's W, MTQ m×B); applies the GNSS jamming map + fault schedule at sample time (§9.2) |

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
(§2.3). Phase 3's F´ SITL transport implements the same callback over TCP with
the §2.4 barrier handshake; `main.cpp`'s open-loop trajectory mode is unchanged.
