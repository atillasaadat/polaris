# The Closed Loop

The §2.4 execution model — plant → sensors → flight software → actuators,
sim-time-driven and bit-reproducible — is implemented sim-side by
`polaris::sim::io::ClosedLoop` ({doc}`/api/sim_io`). Its README, included below,
is the usage reference and the source of truth for repository browsing.

One macro-step (the FSW rate) of the loop, with micro-steps set by the sensors'
native rates on an integer-nanosecond event grid:

```{mermaid}
sequenceDiagram
    participant P as Plant (6DOF + RK89)
    participant S as Sensors (native rates)
    participant B as Buffers (§2.4)
    participant F as FSW callback
    participant A as Actuators (ZOH)

    Note over P,A: macro-step k — commands from step k−1 held
    loop each micro-interval (next sensor event)
        A->>P: net wrench (RW W-matrix τ, MTQ m×B)
        P->>P: propagate to next event time
        P->>S: true state at sensor epoch
        S->>B: IMU Δθ/Δv accumulate · discrete latest+validity
    end
    B->>F: FswInputs (measurements only — no TruthState)
    F->>A: FswOutputs (wheel τ/ω, MTQ dipoles)
    Note over A: applied over macro-step k+1 (causality)
```

```{include} ../../sim/io/README.md
```
