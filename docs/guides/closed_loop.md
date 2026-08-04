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

## Two processes (§2.2): the SITL lockstep

In a SITL run the FSW callback above is not an in-process function but a live
F´ deployment on the other end of a TCP loopback socket. The sim is the master
clock and listens; the flight process connects (`flight_PolarisFsw -s <port>`)
over its own dedicated comm stack, disjoint from the GDS ground link. Each
macro boundary is one blocking STEP_REQ → STEP_REPLY exchange — that blocking
read *is* the §2.4 barrier — and on the flight side each STEP_REQ synchronously
fires a real 10 Hz rate-group cycle on sim time before the reply is built:

```{mermaid}
flowchart LR
    subgraph SIM ["truth-sim process (master clock)"]
        CL["ClosedLoop<br/>(plant + sensors + buffers)"] -->|FswInputs| SS["SitlServer<br/>(listens; barrier = blocking read)"]
        SS -->|FswOutputs| CL
    end

    SS <-->|"F´ frames over TCP loopback<br/>HELLO · STEP_REQ · STEP_REPLY · SHUTDOWN<br/>(lib/sitl/wire.hpp, measurements only — §2.3)"| TC

    subgraph FSW ["flight process (flight_PolarisFsw -s port)"]
        subgraph PS ["PolarisSitl subtopology (excludable for a flight build)"]
            TC["Drv.TcpClient +<br/>FrameAccumulator / Deframer / Framer<br/>(dedicated SITL comm stack)"] --> SB["SitlBridge"]
            SB -->|"2 — cycle"| RG["Svc.PassiveRateGroup<br/>(10 Hz, barrier-driven)"]
            RG -->|run| CS["AttitudeController<br/>(§8.5 B-dot / PID + allocation)"]
            CS -->|"3 — wheel τ / MTQ dipoles"| SB
            SB -->|"4 — STEP_REPLY"| TC
        end
        SB -->|"1 — sim epoch (TAI ns)"| ST["SitlTime<br/>(deployment-wide time source;<br/>stays in main topology)"]
        ST -.->|time port| CS
    end
```

The numbered order inside the flight process is the determinism contract: the
epoch is published first, the rate group runs to completion on the receive
task, and only then is the reply assembled from the commands the cycle latched
— so every FSW decision is a pure function of sim time, and the two-process
truth trace is bitwise identical to the in-process run
(`tests/integration/sim_sitl_lockstep_test.cpp`).

```{include} ../../sim/io/README.md
```
