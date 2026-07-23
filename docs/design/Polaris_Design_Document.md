# Polaris — GNC Flight Software & Simulation — Design Document

**Project:** Polaris — spacecraft GNC software suite (mission planning, flight software, 6DOF simulation, analysis)
**FSW backbone:** JPL F´ (F Prime), C++
**Development model:** Solo developer, single-owner repository
**Status:** Design baseline — source of truth for `CLAUDE.md` generation and from-scratch development. All §18 design decisions are **resolved** and propagated into the body below.

---

## 0. Purpose of This Document

This is the authoritative design baseline for a ground-up GNC software suite. It defines scope, architecture, conventions, and the engineering standards the implementation must follow. It is written to be machine-consumable: the eventual `CLAUDE.md` will be derived directly from it, so every section is concrete enough to drive implementation decisions. Resolved decisions live in **§18**; remaining setup work to do before writing code lives in **§23.7 (Pre-Kickoff Checklist)**.

---

## 1. Vision & Scope

### 1.1 Goal

Develop a 6DOF GNC simulation and flight-software suite containing all features required for an onboard spacecraft GNC FSW, plus the ground-side analysis tooling needed to validate the design. The suite spans four concerns:

1. **Flight Software (FSW)** — F´/C++, flight-grade, NASA-standard, the deliverable that "flies."
2. **Truth/Environment Simulation** — high-fidelity 6DOF plant the FSW runs against (software-in-the-loop).
3. **Mission Analysis Tools** — Python tooling for sizing, budgeting, scheduling, link analysis, and design validation.
4. **Monte Carlo Framework** — dispersion-driven verification across the full stack.

### 1.2 In Scope

**Dynamics, timing & execution**
- 6DOF rigid-body dynamics with high-order environment models.
- **Real-time and faster-than-real-time** simulation execution (§2.4).
- Onboard control cycle at **10 Hz default, configurable per mission**.
- **Onboard time persistence**, disciplined from GPS, surviving resets. Internal master timescale is **TAI** (§3.2); UTC is derived for ground processing. The GNSS receiver reports **GPS time** (and ECEF), so the GNC/OD loops convert GPS→TAI and ECEF→ECI on ingest (§3.2, §6.2, §8.3).

**GNC chain**
- Full ADCS chain: sensing → estimation → guidance → control → actuation → FDIR, including a **coarse attitude mode** (SS + MAG + IMU) for safe/degraded operation when star trackers are unavailable (§8.1).
- Onboard orbit determination (GNSS-sim + MEKF) and ground-based OD (batch least squares).
- **Onboard orbit propagation** with the full force model, including **self-covariance propagation** (the vehicle knows its own state uncertainty).
- **Multi-object propagation:** the FSW can propagate up to **N (~5) secondary objects** from an uploaded state vector + covariance or TLE (§11.2) — groundwork for relative nav / conjunction / future RPOD.

**Conventions & conversions (all unit-tested, §3)**
- ECI ↔ ECEF (modern reduction, latest JPL EOP) for Earth-relative pointing; onboard uses a lightweight EOP table (§3.2, §11.3).
- **ICRF / J2000** as the inertial reference frame for attitude; FSW exposes functions/external calls to obtain ECEF.
- Time conversions: onboard in TAI; GPS time at the GNSS interface; all ground-facing outputs additionally available in UTC.
- Attitude frame conversions among LVLH, ECI, RIC/RTN, body.
- Quaternion math: **JPL convention, scalar-first**, canonical form, normalization checks, numerically safe operations (§3.3).
- User-facing convenience conversions (km, degrees, etc.) at the ground/analysis boundary only.

**Subsystems (modular, "simple now, extensible later")**
- **Power:** simple modular battery + solar-array model for power/SoC simulation (§14).
- **Thermal:** simple modular lumped-node model (§15).
- **C&DH:** fully designed, leveraging F´ machinery as much as possible (§4).
- **EPS:** simple modular model, present for future expansion.
- **Comms (medium fidelity):** patch antenna on the Earth-pointing face, simple antenna + ground-station model, link budget with EIRP / RSSI / margin against the GS network (§16).

**Maneuvering & mission planning**
- **Thruster orbit change:** simple orbit maintenance and raise/lower to target Keplerian elements (SMA / altitude), expandable later (§17).
- **Formation flying / RPOD:** architecture and empty module stubs in place for future open- and closed-loop relative guidance; not implemented in baseline.
- Mission mode state machine and ground-contact handling.

**Outputs & interoperability**
- Sim outputs (non-MC): **CCSDS OEM ephemeris and TLEs, pre- and post-burn**; export to **STK (.e)** and **FreeFlyer** ephemeris for visualization (§20).
- Standardized, frame-tagged, unit-declared data products.

**Engineering**
- **SI everywhere** inside the FSW, always.
- **WGS84** (latest realization) as the Earth reference ellipsoid; **IGRF-14** as the onboard modeled magnetic field (WMM available as backup) for measured-vs-modeled checks and coarse attitude (§3.1, §5.2, §6.2).
- **Comprehensive onboard health telemetry**, with vehicle state published in multiple frames/parameterizations (ECI, ECEF, Keplerian, geodetic, LVLH/RIC) for convenience and onboard algorithms (§20).
- Smart, reusable implementations of common computation (conversions, matrix math, quaternions) via shared libraries (§3.6, fixed-size Eigen).
- F´ GDS and ground-side stack used for real-time operations simulations.
- Every algorithm/model carries a **documented reference source** and the suite has **auto-generated documentation** (§21).
- CI/CD, unit-to-integration test coverage, Monte Carlo verification, and **FDIR fault-injection integration testing** (§23.1.1).

### 1.3 Deliberately Simplified or Deferred

Capped at low/medium fidelity in the baseline (modular so fidelity can be raised later): simple power/thermal/EPS (no detailed cell chemistry / multi-node FEM); comms at geometric + link-budget level (no RF PHY); SMA/altitude maneuver targeting only (no station-keeping boxes/phasing yet); RPOD/formation architecture only; SITL only for now (flight-target cross-compile is future work, §18.2); structural flexibility / fuel slosh optional later (§5.4).

---

## 2. Architecture Overview

### 2.1 Three-Layer Model

```
┌──────────────────────────────────────────────────────────────┐
│  ANALYSIS LAYER (Python, via pybind11 bindings to lib/)       │
│  Momentum budgeting · RW/CMG sizing · detumble MC ·           │
│  GS contact scheduling · link budget · post-processing        │
└───────────────▲──────────────────────────────────────────────┘
                │ pybind11 bindings — reuse the SAME C++/FSW code
┌───────────────┴──────────────────────────────────────────────┐
│  FLIGHT SOFTWARE (F´ / C++)            "the controller"        │
│  Sensor processing · MEKF (attitude+orbit) · guidance ·       │
│  control · actuator allocation · mode manager · FDIR ·        │
│  comms/power/thermal · telemetry/command/event/param          │
└───────────────▲──────────────────────────────────────────────┘
                │ sensor measurements ↓   actuator commands ↑
                │ F´ TCP byte-stream + framing, sim-time lockstep (§2.4)
┌───────────────┴──────────────────────────────────────────────┐
│  TRUTH / ENVIRONMENT SIM (C++)          "the plant"           │
│  6DOF dynamics (RK89) · gravity · third-body · drag · SRP ·   │
│  IGRF · eclipse · disturbance torques · sensor/actuator       │
│  truth models                                                 │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Plant ↔ FSW Interface (SITL)

The truth sim is the plant; the FSW is the controller, coupled in a closed loop:

- **Plant → FSW:** sensor measurement outputs (truth state hidden), time-tagged, at each sensor's native sample rate.
- **FSW → Plant:** actuator command outputs (wheel torques, MTQ dipoles, thruster on/off + duration, CMG gimbal rates).
- **Time:** the sim is the master clock; the FSW executes on rate groups driven by sim time.

**IPC mechanism (resolved, §18.11): F´ native byte-stream transport over TCP.** The two SITL processes communicate using F´'s standard comm stack — a `Drv::TcpServer`/`Drv::TcpClient` byte-stream driver carrying `Fw::Buffer`s with F´ `Svc::Framing`/`Deframing` (the same transport the GDS uses) — rather than a hand-rolled protocol. It is well-tested, easy to use, and the F´-idiomatic way to bridge deployments. Localhost/loopback for SITL.

Scoping note that matters for safety: **this TCP link is test infrastructure, not a flight interface**, so radiation/EDAC/bus-redundancy concerns do not apply to it. The **flight-representative** layer is the FSW's F´ `Drv` hardware-abstraction components (the typed ports between GNC components and the sensor/actuator drivers). On a future flight target those `Drv` components map to standard spacecraft buses — **SpaceWire, MIL-STD-1553, CAN, RS-422/485, SPI/I²C** — which is where bus redundancy, EDAC, watchdog kicking, and bus-fault detection live, surfaced through FDIR (§9). Keeping the GNC↔driver boundary as clean typed ports now makes that future swap a driver-layer change, not a GNC change. The boundary is never a shared-memory shortcut, so later **PIL/HIL** is not precluded.

### 2.3 Truth vs Onboard Model Separation

First-class principle: the FSW must never access truth state. Every onboard quantity is derived from sensor measurements and onboard models that deliberately differ from the truth models (different gravity degree, lower-fidelity ephemeris, biases, noise, latency). This separation is what makes the sim meaningful — it exercises estimators and FDIR against realistic model error. The separation is enforced structurally by the canonical state types (§8.0): the FSW consumes `EstimatedState`, never `TruthState`.

### 2.4 Time & Execution Model (Two-Process, Deterministic)

Because truth and FSW are separate processes and runs must be reproducible at any speed, execution is **sim-time-driven and lockstepped**, never wall-clock-driven:

- **Master clock:** the truth sim owns simulation time and advances in fixed micro-steps (fast rate).
- **Macro-step handshake:** each FSW tick boundary, the sim (1) integrates to the next boundary, (2) publishes sensor samples that came due, (3) the FSW rate groups fire (10 Hz control + estimation/housekeeping rates), (4) the FSW emits actuator commands, (5) the sim applies them on the next step. A barrier synchronizes the two processes per macro-step; the barrier handshake rides as a dedicated control message over the §2.2 transport.
- **Real-time mode** paces the loop to wall clock; **faster-than-real-time mode** removes the pacing. Both are **bit-reproducible from `{config, seed}`** because all logic keys off sim time, not wall time.
- **Sensor rate vs FSW rate:** truth runs fast; each sensor samples at its native rate (e.g., IMU 250 Hz, star tracker lower). An interface buffer holds samples between FSW reads:
  - **IMU:** emits accumulated **delta-angle / delta-velocity** (coning/sculling-compensated) since the last FSW read, so the 10 Hz FSW consumes all 250 Hz information with no loss.
  - **Star tracker / discrete sensors:** FSW reads the **latest valid** sample (with its validity flag and time tag).

---

## 3. Conventions & Standards (Foundational)

Decided once, documented here, enforced everywhere. Inconsistency here is the single largest source of GNC bugs.

### 3.1 Coordinate Frames

| Frame | Definition | Primary use |
|---|---|---|
| **ICRF / J2000 (ECI)** | Earth-centered inertial; **J2000 = attitude reference** | dynamics integration, inertial attitude |
| **ECEF / ITRF** | Earth-centered Earth-fixed | gravity, IGRF, ground-station geometry, **GNSS input** |
| **LVLH** | Local-vertical/local-horizontal | nadir/orbit-relative pointing |
| **RIC / RTN (Hill)** | Radial / In-track / Cross-track | relative state, covariance display, orbit-relative |
| **Body (B)** | Spacecraft structural frame | inertia, actuators, control |
| **Sensor frames** | Per-sensor mounting frames | sensor models, alignment, misalignment |

- **RIC and RTN are the same triad**; the document uses **RIC** as canonical and treats RTN as a synonym.
- Frame transformations (precession/nutation/polar motion/sidereal: **IAU 2006/2000A**) come from a single transform library. ECI↔ECEF uses a documented reduction with **latest JPL/IERS EOP**; no ad-hoc rotations anywhere.
- **Earth reference model: WGS84** (latest realization) is the canonical Earth ellipsoid for geodetic latitude/longitude/altitude, ground-station coordinates, and ECEF↔geodetic conversions. Constants live in the physical-constants registry shared by truth and onboard so they cannot silently disagree.
- Every vector/quaternion in code and telemetry is **frame-tagged** — no anonymous 3-vectors.

**Typed frame/unit vectors — boundary-only.** Frame (and unit) tagging is enforced through the *type system at module/component boundaries*: public interfaces, ports, and the canonical state structs (§8.0) use thin compile-time-tagged wrappers over fixed-size Eigen — `Vec3<ECI>`, `Vec3<ECEF>`, `Vec3<Body>`, `Quat<Body,ECI>`, etc. — so that mixing frames across a boundary is a **compile error**, not a silent runtime bug. Inside hot math kernels (estimator/control inner loops) code may drop to raw fixed-size Eigen for performance; the tagged type is reattached when the result crosses back out to a boundary. This captures most of the frame-safety benefit with minimal friction in the numerics. The wrapper conversion operators route exclusively through the single transform library (no ad-hoc rotations).

### 3.2 Time Systems

- **Onboard master timescale = TAI** — continuous, monotonic, leap-second-free ("true to time"). Running the onboard clock in UTC would inject leap-second discontinuities into propagation/integration; TAI avoids this.
- **GNSS interface is GPS time.** The receiver reports **GPS time** and **ECEF** state. On ingest, the GNC/OD loops apply the constant **TAI = GPS + 19 s** conversion and an **ECEF → ECI/J2000** transform (onboard EOP) before the inertial-frame filter and propagator run. This GPS→TAI / ECEF→ECI conversion on every fix is a documented, unit-tested step (§6.2, §8.3).
- **UTC is derived** from TAI via a leap-second table for ground-facing outputs/display. **TT/TDB** used for ephemeris.
- **Onboard representation:** monotonic **int64 nanoseconds** since the TAI epoch for the FSW master clock — exact, cheap, ~292-year range. A two-part high-precision form (int64 s + double frac) is used where long-arc precision matters (propagation, ephemeris).
- **Discipline & persistence:** the onboard clock is disciplined from GPS time and **persists across resets** (§23.6); re-acquired at first fix on cold start.
- **EOP & leap seconds** are data-driven from loadable IERS files on the ground/truth side; onboard carries an **uploaded EOP table** (UT1-UTC, polar motion) and leap-second table. A **settable/frozen** mode supports reproducible MC.

### 3.3 Attitude Representation & Conventions

- **Quaternion convention: JPL, scalar-first** `q = [q0, q1, q2, q3]`, `q0` scalar. Fixed; propagates into the MEKF error-state algebra and every quaternion op.
- Canonical form enforced (`q0 ≥ 0`); normalization checks and safe-renormalization policy documented.
- MEKF internal attitude error uses a **3-parameter representation** (generalized Rodrigues / rotation-vector); reference attitude is a unit quaternion. Reset policy documented.
- DCM ↔ quaternion ↔ Euler conversions in one library; Euler sequence(s) documented per use.

### 3.4 Units

- **SI throughout** (m, s, rad, kg, N, N·m, T, A·m²). No mixed units across module boundaries.
- Telemetry units declared per channel. Display-only conversions (deg, km) happen at the ground/analysis layer only.

### 3.5 Numerical Reproducibility

- All stochastic processes draw from explicitly seeded RNG streams; a run is fully reproducible from `{config, seed}`, including across the two-process boundary (logic is sim-time-driven, §2.4).
- Per-source seed derivation (counter-based / SplitMix) so adding a noise source doesn't perturb existing streams.
- Floating-point determinism policy across platforms (WSL/Linux/macOS) documented.

### 3.6 Coding Standards (FSW)

- **JPL "Power of Ten"** rules and the **JPL Institutional Coding Standard for C/C++** as baseline; deviations require documented rationale.
- **No dynamic memory allocation after initialization.** All buffers/state fixed-size, statically/stack-allocated. Linear algebra uses **fixed-size Eigen** (stack-allocated, no heap); dynamic-size Eigen types banned in flight paths.
- No recursion in flight paths; bounded loops; no unbounded blocking.
- All return codes checked; no silent failure. Defensive range/finiteness checks on estimator and control outputs.
- `const`-correctness; **no C++ exceptions in flight logic** (§18.10) — faults are surfaced as F´ events/EVRs that drive FDIR responses (state changes) and/or operator alerts, every alert mapped to an action or explicit no-action.
- **Reference provenance:** every algorithm, method, model, or numerical technique cites its source (textbook chapter preferred, else paper) in the code header/docstring, keyed into the project bibliography (§21). No unsourced "magic" formulas.
- Static analysis is part of the gate (§23.2): `clang-tidy`, `cppcheck`, warnings-as-errors, sanitizers in test builds.

### 3.7 External Reference Data (Original Format Is the Source of Truth)

External reference data — IERS Earth-orientation (`finals.all.iau2000`), TLEs, space-weather (F10.7 / Ap / Kp), EGM/gravity coefficient files, SPICE/JPL ephemeris and PCK kernels, magnetic-field models (IGRF), leap seconds, etc. — is committed and consumed **in the original format its authoritative source publishes**, byte-for-byte. We do **not** invent a pre-processed intermediate (a bespoke JSON/CSV/binary) as the committed artifact.

- **Source of truth = the upstream file, verbatim.** What we commit is what the source serves, so it is trivially refreshable by re-download and diff-able against upstream. Provenance (the source URL) is recorded next to it.
- **Parsers consume the native format as-is.** Each format gets a small parser in this repo; the format spec (fixed-width columns, record layout) is the contract. This lets a fetch tool auto-download straight from the source with **no pre-processing step** between "downloaded" and "committed."
- **Processing/derivation is allowed, but downstream of the original — never a replacement for it.** Deriving trimmed windows, unit conversions, continuous-quantity reconstructions (e.g. UT1−TAI from UT1−UTC + ΔAT), or uploadable tables is fine; those are computed *from* the committed original, not substituted *for* it.
- **Fetch tools live in `tools/`, data in `tests/golden/` (or the relevant data dir), and CI never downloads** — committed data is static; downloads are a manual, mirror-tolerant regeneration step.
- **Rationale:** one canonical artifact, no drift between "what the URL says" and "what we ship," and updates that are a plain overwrite. See §11.3 (onboard EOP/ephemeris are *uploaded* tables derived ground-side; flight never reads these files) and §21.1 (reference provenance).

Worked example (Push 9): `tests/golden/finals.all.iau2000.txt` is the raw IERS product committed verbatim; `tools/eop/` fetches it (with mirror fallback); the fixed-width Bulletin-A columns are parsed directly in C++/Python; and the flight `EopTable` is populated by upload (`addEntry`), never by reading the file.

---

## 4. Flight Software (F´ / C++) & C&DH

### 4.1 F´ Architecture
- **Components** encapsulate functions; **ports** define typed interfaces; the **topology** wires them. No back-channel communication outside ports.
- **Rate groups** drive deterministic execution (10 Hz control loop; estimation, OD, housekeeping at configured rates). Rates are configuration, not hard-coded.
- Each component exposes **commands, telemetry channels, events (EVRs), and parameters** via standard F´ machinery.
- **C&DH leverages F´ as much as possible:** command dispatch, telemetry DB, parameter DB, event logging, rate-group driver, sequencing, and the file/data-product subsystem are F´-native rather than re-implemented.
- **Command sequencing** (absolute/relative-time sequences) supports scripted autonomous ops (scheduled burns, contacts). GDS used for command/telemetry during dev and ops sims.

### 4.2 GNC Component Decomposition

| Component group | Responsibility |
|---|---|
| **Sensor processing** | per-sensor acquisition, validity flagging, calibration, time-tagging, GPS→TAI / ECEF→ECI conversion |
| **Attitude estimation** | MEKF (quaternion + gyro-bias); multi-IMU and multi-sun-sensor fusion feeds it |
| **Orbit estimation/propagation** | onboard MEKF OD from GNSS-sim; self + multi-object propagation with covariance |
| **Guidance** | reference attitude/trajectory, slew planning, keep-out/keep-in constraints, maneuver targeting |
| **Control** | B-dot, PID, RW L-norm/L-inf laws, momentum management |
| **Actuator allocation/management** | RW pyramid **or** CMG steering (modular, §18.9), MTQ + desaturation, thruster pulse management |
| **Mode manager** | mission state machine, transition guards, mode-dependent guidance/control sets |
| **FDIR** | fault monitors, isolation, response, safing escalation |
| **Comms / link** | antenna model, contact-window + link-margin telemetry |
| **Power / thermal** | simple modular subsystem models + health flags into FDIR/Safe mode |
| **Health & telemetry** | diagnostic channels, event logging, data products |

### 4.3 Memory & Real-Time Model
- Static memory budget per component, allocated at init. No heap in steady state.
- WCET considered for high-rate paths; deterministic, bounded execution; a timing budget table (§23.5) bounds the 10 Hz frame.
- Estimator/control state is double-precision.

---

## 5. Dynamics & Environment Simulation (Truth Model)

### 5.1 6DOF Dynamics
- Coupled translational + rotational rigid-body dynamics.
- **RK89** (adaptive or fixed-step, configurable) with documented tolerance/step control. Energy/momentum conservation checks as validation diagnostics.
- Quaternion kinematics integrated with renormalization; state vector and Jacobian conventions documented.

### 5.2 Environment Models

| Effect | Model | Notes |
|---|---|---|
| **Gravity** | Spherical-harmonic, **EGM2008**; **settable zonal/tesseral degree & order** | + optional **solid-Earth and ocean tide** corrections |
| **Third-body / multi-body** | Point-mass from JPL ephemeris | **SPICE / NAIF (DE440/DE441)** on ground/truth |
| **Atmospheric drag** | **NRLMSIS 2.1** (latest; 2.0-equivalent for mass density) | density driven by space-weather files |
| **Space weather** | Loadable latest files (F10.7, Ap/Kp) **or settable/frozen** | reproducibility mode for MC |
| **Solar radiation pressure** | Cannonball + optional panel/facet model | coupled to eclipse |
| **Eclipse** | Umbra/penumbra (conical) + cylindrical option | gates SRP, sun sensor, power |
| **Geomagnetic field** | **IGRF-14** (valid through 2030); **WMM** available as alternative/backup | onboard modeled field backs up the magnetometer measurement (measured-vs-modeled consistency check, FDIR, and coarse attitude estimation — §6.2, §8.1, §9) |

### 5.3 Disturbance Torques
Explicitly modeled on the body: gravity-gradient (gravity field + inertia), aerodynamic (CP–CM offset × drag), SRP (CP–CM offset, panel geometry), residual magnetic dipole (residual dipole × B).

### 5.4 Advanced / Optional Fidelity (later phases)
Fuel slosh (pendulum/mass-spring) coupled to dynamics; structural flexibility for large appendages; CG migration and inertia change with propellant depletion.

---

## 6. Sensor Models

### 6.1 Generic Sensor Framework
Each sensor is a configurable module producing measurements from truth, with a standard error stack: bias (constant + drift), scale-factor error, axis **misalignment/non-orthogonality**, random noise, quantization, **latency/transport delay**, native sample rate, saturation/range limits, and a **validity model** (dropouts, occultation). Truth-in → measurement-out; the FSW only ever sees the measurement. Per §2.4, sensors run at native rate and the FSW consumes via buffer.

**Line-of-sight occlusion (shared model).** A common geometric occlusion model is applied to all optical/celestial sensors: given the sensor boresight, FOV, and mounting, the model computes whether the **Earth limb** (and, where relevant, Sun/Moon) blocks or intrudes on the required line of sight, driving the sensor invalid or degraded. This covers Earth occlusion of sun sensors and star trackers, not just bright-body keep-out, and is consistent with the eclipse model (§5.2).

### 6.2 Per-Sensor Models

| Sensor | Key error/behavior model |
|---|---|
| **IMU (gyro + accel)** | bias instability, ARW/VRW, rate random walk, scale factor, g-sensitivity, misalignment, quantization; outputs delta-angle/delta-velocity |
| **Star tracker** | noise-equivalent angle, update rate, slew-rate limit, **Sun/Earth/Moon keep-out + Earth-limb occlusion** (§6.1), valid/invalid solution flag |
| **Sun sensor** | per-diode **counts** output, FOV limits, albedo error, **Earth-limb occlusion** and eclipse → invalid; fusion reconstructs direction |
| **Magnetometer** | bias, hard/soft-iron, noise, misalignment; truth from IGRF; onboard **IGRF-14 modeled field** (WMM backup) provides the inertial reference for measured-vs-modeled checks and coarse attitude (§8.1) |
| **GNSS receiver (sim)** | pseudorange/position from TLE-propagated GNSS constellation + receiver clock model, noise, biases, dropouts. **Reports GPS time and ECEF state**, requiring GPS→TAI and ECEF→ECI conversion (§3.2); also disciplines the onboard clock |

**Implemented:** magnetometer and IMU (Pushes 19–20, via the §6.1 error stack); **star tracker + the shared occlusion model (Push 23)**. The occlusion model (`sim/sensors/occlusion`) reuses the §5.2 apparent-disk geometry — every body is checked from its **limb**, not its centre, which at LEO is the difference between a 70° constraint and a point source. The star tracker (`sim/sensors/star_tracker`) reports a full attitude solution rather than a measured vector, so its error is a small-angle perturbation of the truth attitude that is deliberately **anisotropic**: about-boresight accuracy is ~6× worse than cross-boresight, because a roll about the boresight barely moves the identified stars. Validity covers Earth intrusion on the FOV, Sun/Moon keep-out cones, and slew-rate image smear, each reported distinctly. Sun sensor and GNSS receiver remain.

---

## 7. Actuator Models

| Actuator | Model |
|---|---|
| **Reaction wheels** | **Pyramidal configuration**; torque/momentum limits, friction (Coulomb + viscous), quantization, **static & dynamic imbalance**, bias/noise, bearing drag |
| **Magnetorquers (MTQ)** | per-axis dipole limits, **hysteresis model**, residual dipole, B-field-measurement dead-time, momentum **desaturation** |
| **CMGs** | gimbal-rate dynamics, gimbal limits, errors; **singularity behavior** central (§8.5). Per §18.9 a vehicle uses **either** RWs **or** CMGs, not both |
| **ACS thrusters (cold gas)** | on/off, **minimum impulse bit**, thrust rise/fall, multiple thrusters, plume/duty-cycle limits |
| **Main propulsion** | chemical (Isp, thrust, mass flow) **and** electric (low-thrust, throttling); mass depletion + CG shift coupled back to dynamics |

**Implemented (Push 21):** `sim/actuators/reaction_wheel` (torque box + momentum/speed ceiling, Coulomb+viscous+aero friction per the RW-0.4 rundown model, torque quantization, copper+mechanical/regenerative power, static/dynamic imbalance) and `sim/actuators/magnetorquer` (dipole limit, ±linearity, residual moment + hysteresis via a play operator; torque τ=m×B reuses the §5.2 field path). Catalog entries (`config/hardware/`): Rocket Lab RW-0.4 and generic wheel; NewSpace Taurus rod, AAC Clyde Space MTQ800, and generic MTQ. Since Push 22 the installed suite is **built from the compiled config** (`scenario/vehicle`, §19.4); actuator outputs are not yet fed back into the plant's torque sum — that arrives with the §2.4 macro-step loop (`sim/io`), which is what supplies the commands. CMGs and thrusters remain.

**Reaction-wheel jitter (forward-looking).** The RW model carries **static imbalance `Us`** (radial force `Us·ω²`) and **dynamic imbalance `Ud`** (radial torque `Ud·ω²`) and emits both phase-resolved per step, specifically to feed a future **micro-vibration / jitter analysis**: the once-per-rev fundamental (plus structural/bearing harmonics to be added), rendered as **waterfall spectrograms** (disturbance amplitude vs frequency vs wheel speed over a spin sweep) to expose resonance crossings. Imbalance magnitudes are per-unit balance-report data (default zero in the catalog). See `sim/CLAUDE.md` "Reaction-wheel jitter".

---

## 8. GNC Algorithms

### 8.0 Canonical State Structs (the single nav product)
The GNC chain has exactly **one** definition of vehicle state, produced once and consumed everywhere — there is no per-component ad-hoc state passing.

- **`EstimatedState`** — the onboard navigation product: TAI time tag, attitude quaternion (JPL scalar-first, Body←ECI), body rates, position/velocity (canonical in ECI), gyro/accel biases, the full **covariance**, per-field **validity flags**, the **active estimation mode** (fine/coarse), and **frame tags** on every vector (§3.1 boundary typing). This is what the estimators (§8.1, §8.3) emit and what guidance, control, FDIR, and telemetry consume.
- **`TruthState`** — the simulation's ground-truth analogue, the same kinematic/dynamic fields without covariance/validity. The truth-vs-onboard separation (§2.3) is therefore a **type distinction**: the FSW is structurally unable to consume `TruthState`.
- **Versioned:** both structs carry a schema version (§22.1) so serialized logs, telemetry, and golden fixtures remain readable as the state evolves.
- **Derived views, not copies:** the multi-frame representations in telemetry (ECI, ECEF, Keplerian, geodetic, LVLH/RIC — §20) are **computed views** of the one `EstimatedState`, so they cannot disagree with each other or with what the controller used.

### 8.1 Attitude Determination
- **Fine mode (nominal):** **MEKF (multiplicative EKF)** — unit-quaternion reference (JPL scalar-first) + 3-parameter error state, gyro-bias states; star tracker(s), multi-IMU, and multi-sun-sensor fusion feed measurements.
- **Coarse mode (degraded / safe):** when star trackers are unavailable (occlusion, fault, slew-limited), estimation falls back to **sun sensor + magnetometer vector measurements + IMU gyro propagation**. The two body-frame vectors (sun direction, magnetic field) are paired with their inertial references (sun ephemeris, onboard IGRF-14 modeled field, §5.2/§6.2) to form a coarse attitude; the IMU propagates attitude between/through vector updates. This is the estimator used in **Safe** and **Sun Point** acquisition and whenever the fine solution is invalid.
- **Initialization / coarse determination:** deterministic single-frame initializers (**TRIAD / QUEST / q-method**) seed both the coarse solution and the MEKF from two vector measurements, giving a defined cold-start/acquisition path rather than assuming a pre-converged filter.
- **Mode arbitration:** transitions between fine and coarse are driven by sensor validity (§9.1) and surfaced to FDIR and the mode manager (§10); the active estimation mode is telemetered.
- Consistency monitoring (NEES/NIS) as diagnostics.

### 8.2 Sensor Fusion
- **Multi-IMU fusion (2×):** weighted/voted gyro fusion with disagreement detection feeding FDIR.
- **Multi-sun-sensor fusion:** models output **diode counts**; the algorithm reconstructs a fused sun-direction vector with FOV/eclipse handling.

### 8.3 Orbit Determination & Propagation
- **Onboard MEKF OD** from GNSS-sim measurements (biases/noise), onboard force model, **with self-covariance propagation**. GNSS inputs are converted from GPS time/ECEF to TAI/ECI before the inertial-frame filter runs (§3.2).
- **Multi-object propagation** of up to N secondaries (§11.2).
- **Batch least-squares** estimator (ground/analysis) for orbit fit and validation.
- **SGP4** TLE propagation; **CCSDS OEM** ephemeris generation as a data product.
- GNSS pseudo-noise: GNSS satellite states from TLE propagation, pseudorange construction, receiver clock and ionosphere error contributions.

### 8.4 Guidance
- **Pointing modes** generate reference attitude/rate: sun-point, nadir, LVLH, inertial hold, star-track, ground-track.
- **Slew planning:** eigenaxis slews with rate/accel limits; **constrained attitude guidance** honoring keep-out cones (star-tracker vs Sun/Earth/Moon) and keep-in cones (comms, solar-array sun-pointing).
- **Maneuver targeting:** compute burns to reach target Keplerian elements / altitude; impulsive and finite-burn reference generation (§17).

### 8.5 Control
- **B-dot** detumble (MTQ-based).
- **PID** for RW-based pointing.
- **RW control via L-norm / L-∞ allocation** across the pyramid (min-effort / min-max torque distribution).
- **Momentum management & desaturation:** RW momentum monitoring with MTQ (and/or thruster) desaturation.
- **CMG steering law:** singularity-robust steering (singularity-robust inverse / null-motion) when CMG-equipped.
- **Actuator abstraction:** guidance/control produce a commanded body torque; the allocation layer (RW-pyramid **or** CMG-steering) is the swappable piece, keeping upstream logic actuator-agnostic.

---

## 9. Sensor/Actuator Health: Flags & FDIR

### 9.1 Validity Flags
Per-sensor validity flag with documented criteria: range checks, rate-of-change checks, staleness/timeout, cross-sensor consistency, solution-quality flags. Downstream consumers must respect flags — invalid measurements are excluded, never silently used.

### 9.2 FDIR Architecture
- **Monitors** (per sensor/actuator and system-level) → **isolation** → **response** → **safing escalation**.
- Sensor FDIR: dropout/disagreement detection, voting where redundancy exists; **measured-vs-modeled magnetic field consistency** (onboard IGRF) as a magnetometer reasonableness check.
- **GNSS FDIR:** outage detection and graceful coasting on propagation; **spoofing/meaconing** detection via innovation/consistency checks and position/time reasonableness bounds, with measurement rejection.
- **Estimator fallback:** loss/occlusion of star trackers triggers fine→coarse attitude estimation (§8.1); the active estimation mode is part of FDIR state.
- Actuator FDIR: wheel stall/runaway, saturation, thruster stuck-on/-off, dipole saturation.
- Subsystem FDIR: low SoC (power) and thermal-limit monitors can trigger Safe mode.
- **No exceptions:** faults are F´ events with severity; every alert maps to either an autonomous FDIR action (state-machine transition / reconfiguration) or an operator alarm with a documented (possibly null) action. Responses are tiered and escalate to **Safe Mode**. FDIR state and triggers are fully telemetered.
- **Verification:** every monitor/response is exercised by the FDIR fault-injection integration suite (§23.1.1).

---

## 10. Mission Modes / State Machine

Mode manager implementing a documented state machine with explicit entry/exit conditions and transition guards. Each mode binds a guidance reference + control/gain set + active actuator/sensor configuration.

| Mode | Purpose |
|---|---|
| **Safe** | recover to a power/thermal-safe, controllable attitude; FDIR sink |
| **Detumble** | B-dot rate reduction from tip-off |
| **Sun Point** | array-to-sun acquisition/hold |
| **Nadir Track** | Earth-pointing |
| **Star Track** | inertial target / observation pointing |
| **LVLH Pointing** | orbit-relative attitude |
| **Inertial Hold** | fixed inertial attitude |
| **Ground-Station Tracking** | track a station from the onboard GS list during contact windows |
| **Delta-V / Maneuver** | attitude + thruster sequencing for orbit-change burns (§17) |

- Transitions: autonomous (FDIR/conditions) + commanded; guards prevent illegal/unsafe transitions.
- **Onboard ground-station list** (configurable): lat/lon/alt, mask angles; used for contact prediction, tracking, and link analysis.

---

## 11. Orbital Mechanics & Multi-Object Handling

### 11.1 Mission Planning
- **SGP4/TLE** propagation; **CCSDS OEM/OMM** ephemeris generation/ingest.
- **Ground-station contact prediction** (elevation mask, horizon geometry) — feeds link analysis (§16).
- **Eclipse prediction** windows — feeds power (§14).

### 11.2 Multi-Object Propagation & Tracking
- Onboard **secondary object catalog** of up to **N (~5)** objects, each from an uploaded **state vector + covariance** or **TLE**.
- Propagated with the onboard force model (or SGP4 for TLE-sourced), **with covariance propagation**.
- **Hooks (architectural now, empty):** relative geometry (range, range-rate, RIC offsets) and conjunction assessment (miss distance, Pc) — supports the future RPOD path.

### 11.3 Onboard Ephemeris & Earth Orientation
- Sun/Moon/planet positions onboard via **Chebyshev polynomial fits** generated on the ground from SPICE (DE440) and uploaded as coefficient sets per interval (§18.6).
- Onboard **EOP table** (UT1-UTC, polar motion) and **leap-second table** uploaded as low-rate parameters for ECI↔ECEF and UTC derivation.

---

## 12. Mission Analysis Tools (Python, via pybind11)

Per §18.7, analysis tools **reuse the same C++/FSW code through pybind11 bindings** — Python exercises the exact flight implementations, minimizing reimplementation and divergence. These same bindings later power the live web tools (§21.4).

| Tool | Function |
|---|---|
| **RW/CMG momentum budgeting & sizing** | validate actuator sizing vs target slew rates, torque, momentum storage for the mode set |
| **Detumble-time analysis** | time-to-detumble across initial rates, Monte Carlo dispersed |
| **Ground-station contact scheduling** | contact windows + schedule across the GS network |
| **Link budget / RF margin** | EIRP, path loss, pointing loss, RSSI, Eb/N0, margin per pass (§16) |
| **Pointing-budget / coverage** | end-to-end pointing error budget; optional ground-track coverage |
| **Post-processing & plotting** | standardized analysis of sim/MC data products |

---

## 13. Monte Carlo Framework

- Dispersions over initial conditions, mass properties, sensor/actuator errors, environment (space weather), and timing.
- Seeded, reproducible (§3.5); parallel execution; per-run config capture.
- **Pass/fail criteria with margin** (pointing-error percentiles, settling time, momentum/power margins) — margin against requirement thresholds is reported, not just pass/fail (§22.2).
- **Estimator consistency** metrics (NEES/NIS) aggregated across runs.
- Runnable as a CI smoke subset and as full off-line campaigns.

---

## 14. Power Subsystem (Simple, Modular)

- **Solar array:** generated power = efficiency × solar flux × area × cos(incidence), gated by eclipse; incidence from sun vector and array normal (attitude-dependent).
- **Battery:** state-of-charge via energy/coulomb integration; capacity, charge/discharge limits, round-trip efficiency.
- **Loads:** per-mode configurable load profile.
- **Outputs:** SoC and power-margin time-series; low-SoC monitor feeds FDIR/Safe mode.
- **Modular** so a higher-fidelity EPS can drop in.

---

## 15. Thermal Subsystem (Simple, Modular)

- **Lumped-node** (1–few nodes): absorbed environmental flux (solar, albedo, IR) + internal dissipation − radiated; simple heater on/off control.
- **Outputs:** node temperature time-series; thermal-limit monitor feeds FDIR/Safe mode.
- **Modular** for future multi-node expansion.

---

## 16. Communications & Link Analysis (Medium Fidelity)

- **Antenna:** patch antenna on the Earth-pointing (nadir) face; boresight gain + simple pattern (cosine roll-off), polarization.
- **Link budget:** EIRP (Tx power + line loss + antenna gain) − free-space path loss (slant range) − **pointing loss** (off-boresight angle from attitude + pass geometry) − atmospheric/margin terms + ground-station G/T → received power, C/N0, Eb/N0, **link margin** vs required.
- **Pass products:** during each GS contact, time-series of elevation, slant range, off-boresight angle, RSSI/EIRP, and margin → predicted contact quality / usable data rate.
- **Modular:** antenna and GS RF parameters configurable; future work adds modulation/coding and multiple antennas. Geometric contact windows from §11.1; this layer adds RF margin.

---

## 17. Orbit Maintenance & Maneuvering

- **Thruster orbit change:** simple maintenance and raise/lower to **target Keplerian elements (SMA / altitude)**.
- **Targeting:** compute required Δv; **finite-burn** modeling (thrust, Isp, duration) with guidance hand-off (§8.4) and a **Delta-V/Maneuver mode** (§10).
- **Products:** CCSDS OEM + TLE ephemeris **pre- and post-burn** (§20).
- **Expandable later** to station-keeping boxes, phasing, and drag make-up scheduling.

---

## 18. Resolved Design Decisions

All previously-open decisions are resolved and propagated into the body above. Recorded here as a decision log.

| # | Decision | Resolution |
|---|---|---|
| 1 | **Quaternion convention** | **JPL, scalar-first** `[q0,q1,q2,q3]` (§3.3) |
| 2 | **Compute target** | **SITL only for now**; build targets = WSL/Linux + macOS, local machines only. Flight-target cross-compile is future work |
| 3 | **Plant↔FSW coupling** | **Two processes**, sim-time lockstep (§2.4). Truth fast; sensors native rate; FSW buffer-consumes (IMU delta-angle/-velocity; latest-valid otherwise) |
| 4 | **Time representation** | Onboard master **TAI**; int64 ns; two-part (int64 s + double frac) for long arcs; **GNSS reports GPS time/ECEF → convert to TAI/ECI**; UTC derived for ground (§3.2) |
| 5 | **Linear-algebra library** | **Fixed-size Eigen** (stack-allocated, no heap); dynamic-size types banned in flight (§3.6) |
| 6 | **SPICE onboard?** | **Chebyshev/polynomial fits onboard**; SPICE/NAIF on ground/truth only (§11.3) |
| 7 | **Analysis-tool reuse** | **pybind11 bindings** reusing the exact C++/FSW code (§12), also powering live web tools (§21.4) |
| 8 | **Default vs configurable fidelity** | **No defaults — fully configurable**, with template configs and a **hardware model library** (§19.2): hardware selected by model-ID string, params defined in the model |
| 9 | **CMG inclusion** | **Included**, but **RW-or-CMG, not both** — a modular actuator choice per spacecraft config (§7, §8.5) |
| 10 | **Exception policy** | **No code exceptions** in flight logic. Faults → F´ events → FDIR actions and/or operator alarms; every alert maps to an action or explicit no-action (§3.6, §9.2) |
| 11 | **SITL IPC mechanism** | **F´ native byte-stream transport over TCP** (`Drv::TcpServer/Client` + `Svc::Framing`), localhost. Test infra, not a flight bus; flight-representative layer is the F´ `Drv` HAL mapping to SpaceWire/1553/CAN/RS-422 on a future target (§2.2) |
| 12 | **V&V reference tool** | **GMAT** for golden-data generation and cross-validation (propagation, conversions, frames, eclipse, contacts). **Orekit explicitly not used** (§23.1) |
| 13 | **Frame/unit type safety** | **Boundary-only typed vectors** — compile-time frame/unit-tagged wrappers (`Vec3<ECI>`, `Quat<Body,ECI>`) at interfaces/ports/state structs; raw fixed-size Eigen allowed inside hot kernels (§3.1) |
| 14 | **Canonical state** | **`EstimatedState` / `TruthState`** as the single versioned nav product; truth-vs-onboard is a type distinction; telemetry frames are derived views (§8.0) |
| 15 | **Config pipeline** | **One config compiler** emits F´ params + sim + analysis from the single source-of-truth config; uplinks are a versioned overlay on the authoritative ground baseline (§19.3) |

---

## 19. Configuration System & Hardware Model Library

### 19.1 Spacecraft & Scenario Configuration
- **Human-readable, settable configuration** (YAML/JSON) with a documented, validated schema.
- A single config fully defines a spacecraft + scenario: mass / CoM / inertia, sensor suite, actuator suite, control gains per mode, mode parameters, GS network, antenna/RF, power/thermal params, environment/epoch, MC dispersions.
- Same config drives sim, FSW parameters, and analysis tools — **one source of truth** per design.

### 19.2 Hardware Model Library (per §18.8)
- A library of **parameterized hardware model definitions keyed by model ID** (e.g., IMU `STIM300`, star tracker `ST-16`, reaction wheel `RW-X`). Each entry carries that unit's error/performance parameters.
- A spacecraft config **references hardware by model ID**; swapping the string swaps the hardware — re-fly the same vehicle with a different IMU by changing one identifier.
- **Template configs** (e.g., a LEO smallsat) provide ready test setups.

### 19.3 Configuration Pipeline (one compiler, one source of truth)
The config (§19.1) + hardware-library references (§19.2) are the **single source of truth**, compiled by one mechanism into every consumer so the three representations that must agree — YAML config, F´ `ParameterDb`, sim setup — cannot drift:

- **Config compiler:** one tool resolves hardware model-IDs against the library, validates against the schema, and emits the derived artifacts: **F´ parameter sets** (loaded into `ParameterDb`), **sim/truth setup**, and **analysis-tool inputs** (via the same pybind11 layer, §12). A single resolved, validated config object underlies all three — no consumer re-parses raw YAML independently.
- **Provenance:** each emitted artifact records the source config hash/version (§22.1) so any FSW param, sim run, golden fixture, or MC campaign is traceable to the exact config that produced it.
- **Uplink vs source-of-truth reconciliation:** on-orbit parameter uplinks (§4) may change `ParameterDb` values at runtime. The rule: the **ground config remains the authoritative baseline**; uplinked deltas are captured as a versioned overlay on top of the baseline (never silent divergence), the effective onboard parameter set is telemetered, and ground tooling can diff effective-vs-baseline so the as-flown configuration is always reconstructable.

### 19.4 No hardware or vehicle constants in code
**Implemented (Push 22).** Hardware and vehicle parameters live in `config/` and nowhere else. The in-code `catalog::` factories that had duplicated datasheet numbers in C++ (`catalog::stim377h()`, `catalog::rocketLabRw04()`, `catalog::aacMtq800()`, …) are **deleted**, not deprecated, and the runtime path is wired end to end: `config/hardware/**.yaml` → the config compiler resolves the `model_id` and inlines the unit's params → `sim_setup.json` → `SimConfig::UnitConfig` → the model's own `fromParams` → the built model.

The standing rules:
- **The YAML library and spacecraft config are the only source of hardware and vehicle parameters.** Every sim model is constructed from params carried in the compiled artifact (§19.3), resolved by model ID. Swapping a `model_id` string is the entire mechanism for re-flying a vehicle with different hardware, and it cannot be overridden from C++.
- The parameter map is passed through **uninterpreted** by every layer between the compiler and `fromParams`, so adding a parameter to a catalog entry touches the model and the YAML only.
- **No physical constant describing a *unit* or a *vehicle* lives in C++ or Python source.** Physical/mathematical constants that are not configuration (G, µ⊕, WGS84 defining parameters, unit conversions) stay in `lib/constants/` — those are not hardware.
- **The only permitted hardcoded specs are in unit and integration tests** — a test may construct a spec inline to isolate one physical effect (a wheel with zero friction, an MTQ with exaggerated residual), because those are deliberately non-physical fixtures, not catalog entries. Where a test fixture mirrors a real catalog entry it says so and pins the *conversion*, not the datasheet value; the datasheet values are pinned against the YAML on the Python side.
- A unit that resolves to **no** parameters is a hard error, not a degenerate device: it would otherwise build as an ideal, noiseless, unlimited instrument and produce a clean-looking run.
- A configured unit whose device class has no truth model yet is **reported, never silently dropped** (`Vehicle::unmodelled`).
- **Seeding follows the same rule.** The master seed is a config field (`scenario.seed`), and each unit's stream is derived from its *instance name*, not its list position, so installing or reordering hardware leaves every other unit's stream bit-identical (§3.6).

---

## 20. Telemetry, Logging, Data Products & Interoperability

- F´ **telemetry channels, events, parameters** for every component; **comprehensive onboard health & status** — sensor/actuator states and validity, estimator mode (fine/coarse) and covariance summaries, FDIR state and active faults, momentum/power(SoC)/thermal margins, mode and command status — sufficient to assess full vehicle health and root-cause anomalies offline.
- **Multi-frame state representation:** the vehicle state is published in **multiple frames/parameterizations for convenience and onboard algorithms** — position/velocity in **ECI** and **ECEF**, **Keplerian/orbital elements**, **geodetic** lat/lon/alt (WGS84), and orbit-relative **LVLH/RIC** — plus attitude as quaternion, and where useful Euler angles and body rates. These are **derived views of the single canonical `EstimatedState`** (§8.0), computed through the one transform library, so the representations cannot disagree with each other or with the state the controller actually used.
- Standardized internal data-product format (time-series, frame-tagged, unit-declared).
- **Ephemeris/interop exports:** **CCSDS OEM** + **OMM/TLE** (pre- and post-burn), **STK (.e)** and **FreeFlyer** ephemeris for visualization.
- Optional **CCSDS packetization** for telemetry to mirror flight practice.

---

## 21. Documentation & Reference Standards

Documentation is a first-class deliverable, not an afterthought. Three pillars:

### 21.1 Reference Provenance (every method is sourced)
- Every algorithm, method, model, filter, and numerical technique documents its **source** — a **textbook chapter (preferred)** or **paper** — in the code header/docstring, with the exact reference (author, title, edition/chapter or DOI).
- Sources are aggregated into a single project bibliography (`docs/refs.bib`); docstrings cite by bibkey so citations render in the generated docs (via `sphinxcontrib-bibtex`).
- This is enforced by convention and reviewed in CI doc builds; unsourced derivations are flagged.

### 21.2 Auto-Generated API Documentation (docstring-driven)
A single, unified, nicely-formatted docs site built from code docstrings — matching the NumPy docs experience you like:
- **Python:** **Sphinx** + **numpydoc** (NumPy docstring style) + napoleon; the **PyData Sphinx Theme** (the theme NumPy itself uses) for the look and feel.
- **C++:** **Doxygen** for the API, bridged into Sphinx via **Breathe** (+ optional **Exhale** for the API tree) so C++ and Python render in one consistent site.
- **F´:** component documentation and dictionaries (commands, telemetry, events, params) auto-generated from the FPP models; linked/embedded into the site.
- **References:** `sphinxcontrib-bibtex` renders the bibliography and inline citations from `docs/refs.bib`.
- **Build & publish:** the docs are built in CI and published to **GitHub Pages**; a broken docstring, missing reference, or doc-build failure **fails the CI gate**.

### 21.3 Docstring & Documentation Conventions
- NumPy-style docstrings for Python; structured Doxygen comment blocks for C++ — both carrying parameters, units, frames, returns, references, and examples.
- Units and frames are stated in every docstring for any physical quantity (reinforces §3.1/§3.4).
- Worked examples in docstrings are runnable and doubled as doctest where practical.

### 21.4 Live Web Tools (future phase)
- A public interactive site exposing tools such as **RW momentum budgeting/allocation**, detumble-time, and link budget — **driven by the same codebase** so the web tool and the FSW compute identically.
- Mechanism: the **pybind11-bound C++** is either compiled to **WebAssembly (Pyodide/Emscripten)** for client-side execution or served by a thin backend; either way the live tools call the exact §12 implementations — no reimplementation, no divergence.
- Builds naturally on the unified docs site (§21.2) and the pybind11 decision (§18.7).

---

## 22. Interfaces, Requirements & Repository Layout

### 22.1 Interface Control Documents (versioned)
Define and version before implementation churns:
- **Canonical state structs** — `EstimatedState` / `TruthState` schema and version (§8.0), the single nav product crossing every GNC boundary.
- **Configuration schema + config-compiler outputs** (spacecraft, scenario, hardware-model entries, MC dispersions; the F´-param / sim / analysis artifacts the compiler emits, §19.3).
- **Plant↔FSW IPC + sim-time-sync ICD** — F´ TCP byte-stream framing, message schema, macro-step handshake (§2.2/§2.4).
- **Data-product / interop formats** (internal time-series; CCSDS OEM/OMM; STK/FreeFlyer).
- **Onboard upload formats** (Chebyshev ephemeris coeffs, EOP/leap-second tables, secondary-object catalog, **parameter-uplink overlay** §19.3).

### 22.2 Requirements Baseline & Verification (REQ ICD before kickoff)
- A **Requirements ICD** (`docs/requirements/`, `REQ-###` IDs grouped by subsystem) is authored **before development starts**.
- **Bidirectional traceability:** every capability → ≥1 requirement; every requirement → ≥1 verifying test, tracked in CI.
- **Coverage & margin targets:** the test suite targets **high coverage** (line + branch thresholds enforced in CI) and verification confirms that **all requirements are met with quantified margin** — each requirement records its verification method, result, and margin against threshold (this feeds the MC pass/fail-with-margin reporting in §13). The intent is a green traceability matrix where every REQ is demonstrably satisfied with headroom, not merely passed.

### 22.3 Repository Layout (proposed)
```
polaris/
├── CLAUDE.md  README.md  LICENSE
├── docs/            # design, ICDs, requirements; Sphinx site, refs.bib, Doxygen config
├── fprime/          # F´ framework (submodule)
├── flight/          # FSW deployment: components/ ports/ topology/ config/
├── lib/             # shared C++: math(Eigen,quat,typed-vec) frames(+EOP) time(TAI/UTC/GPS)
│                    #   state/ (EstimatedState/TruthState) constants/ (registry incl. WGS84)
│                    #   environment(grav,drag,SRP,IGRF,3body) ephemeris(SPICE+Cheby) models/
├── sim/             # truth plant: dynamics(RK89) world/ io(F´ TCP IPC + sim-time sync)
├── bindings/        # pybind11 exposing lib/ to Python (and future WASM web tools)
├── analysis/        # momentum/ detumble/ contacts/ linkbudget/ postproc/
├── config/          # hardware/ (model library)  spacecraft/  scenarios/
├── tests/           # unit/ component/ integration/ regression/ golden/ (GMAT fixtures)
├── tools/           # dev/CI scripts; gmat/ (golden-data gen); configc/ (config compiler §19.3)
├── mc/              # Monte Carlo campaign configs + runners
└── .github/workflows/
```

---

## 23. Verification, Tooling & Pre-Kickoff Checklist

### 23.1 Verification & Validation (golden data via GMAT)
- **Test pyramid:** unit (math/library) → component → integration (closed-loop SITL scenarios) → Monte Carlo → regression (golden-file comparison).
- **GMAT for golden data & cross-validation:** **GMAT** (NASA GSFC, open-source, scriptable, headless/batch) is installed and used to **generate reference ("golden") datasets** for unit/function-level V&V — orbit **propagation** (force-model cases), **time/coordinate conversions**, **frame transforms** (ECI↔ECEF), **eclipse**, and **contact geometry**. Golden datasets are stored as **versioned fixtures** (`tests/golden/`), regenerated by scripted GMAT runs (`tools/gmat/`), and Polaris functions are compared against them within **documented per-quantity tolerance bands** in the test suite and CI. **Orekit is explicitly not used.**
- **Property/analytic checks:** energy/momentum conservation, two-body analytic comparisons, frame round-trips, quaternion/DCM identities.
- **Requirements traceability:** every capability → test; coverage and per-REQ margin tracked in CI (§22.2).

### 23.1.1 FDIR Fault-Injection Integration Testing
A dedicated closed-loop SITL test suite injects faults into the truth sim / sensor / actuator models and asserts that FDIR **detects, isolates, and responds** correctly (correct event raised, correct mode/reconfiguration, recovery, and bounded time-to-detect/respond). Fault-injection hooks are a first-class capability of the sim (scriptable per scenario). Each fault case is a traceable test mapped to the FDIR requirements. Case library (extensible):

- **Sensor degradation/faults:** bias jump/drift, scale-factor shift, increased noise, stuck/frozen value, NaN/out-of-range.
- **Sensor loss:** dropout/death of an IMU, star tracker, sun sensor, or magnetometer; verify fallback (e.g., multi-IMU voting, fine→coarse estimation, §8.1).
- **Occlusions:** Earth-limb occlusion of star tracker / sun sensor and eclipse entry (§6.1) → estimator mode transition and continued control.
- **GNSS faults:** **GPS outage** (loss of fix) and recovery; degraded geometry; clock jumps; **GPS spoofing/meaconing** (inconsistent or slowly-walked position/time) → detection via innovation/consistency checks and reasonableness bounds, rejection, and graceful coasting on propagation.
- **Actuator faults:** reaction-wheel stall/runaway/saturation, MTQ dipole saturation, thruster stuck-on/stuck-off, CMG approaching singularity.
- **Subsystem limits:** low battery SoC, thermal limit exceedance → Safe-mode entry.
- **Timing/data faults:** stale/late measurements, out-of-sequence arrivals, clock discontinuities.
- **Multi-fault / cascade:** combined faults (e.g., ST loss during eclipse with a degraded IMU) to test escalation and safing.

Each case asserts both **detection** (right fault flagged, no false positives on nominal runs) and **response** (correct action and recovery), with detection latency checked against requirement thresholds.

### 23.2 Build, Quality Gates & CI/CD
- **CMake** + F´ build; host (WSL/Linux/macOS) baseline; Python tooling with pinned, reproducible environments.
- Gates: `clang-format` (enforced), `clang-tidy`, `cppcheck`, **warnings-as-errors**, ASan/UBSan in test builds, valgrind optional.
- **Docs gate:** unified Sphinx site (numpydoc + Breathe/Doxygen + bibtex) builds clean — broken docstrings/missing references fail CI; site published to GitHub Pages.
- **CI/CD (GitHub Actions):** build → static analysis → unit/component → integration SITL → MC smoke → GMAT-golden regression → coverage (with thresholds) → docs build/publish; artifacts published per run.

### 23.3 Solo-Developer Git Workflow
- Single-owner repo; **no merge-approval / no required reviewers** (solo).
- **Feature branches** developed and run through CI/CD before merging to `main` (self-merge once green).
- Pre-commit hooks (format + quick lint); conventional-commit messages; **semantic versioning** + tagged releases; maintained CHANGELOG. Issues/board as a personal planning tool, not a gate.

### 23.4 Licensing, Third-Party Data & Export Considerations
- **License (resolved):** **dual-license** — the public baseline is the **PolyForm Noncommercial License 1.0.0** (free for research, education, and university/non-profit smallsat use, with attribution), and **commercial use requires a separately-sold commercial license** (see `LICENSE`, `LICENSING.md`). This supersedes the original Apache-2.0 suggestion. Third-party dependencies (F´ and GMAT — both Apache-2.0 — plus Eigen, CSPICE, etc.) remain under their own licenses, tracked in `THIRD_PARTY_NOTICES.md`.
- **Third-party terms:** track usage/attribution for SPICE/NAIF kernels, IGRF coefficients, EGM2008, NRLMSIS, space-weather files, SGP4, and **GMAT** outputs.
- **Export hygiene (if public):** keep models generic/educational, rely on the public-domain/published-information posture, and avoid embedding export-controlled real-hardware performance specs. A short written policy at kickoff is worthwhile given the public-portfolio context.

### 23.5 Performance / Timing Budget
A timing-budget table bounding the 10 Hz frame (per-rate-group WCET allocation: sensor processing, estimation, guidance, control, allocation, telemetry) to verify the control cycle closes deterministically.

### 23.6 Persistent State & Restart Behavior
Onboard persistent state across resets: time/epoch, OD state + covariance, ephemeris/EOP tables, mode, calibration, secondary-object catalog. Defined cold/warm-start behavior; time re-acquisition from GPS at first fix; graceful resume.

### 23.7 Pre-Kickoff Checklist
- [ ] **Requirements ICD** authored (`REQ-###`) with traceability + coverage/margin targets wired into CI (§22.2).
- [ ] Plant↔FSW **IPC + sim-time-sync ICD** drafted (F´ TCP framing) and versioned.
- [ ] Config + hardware-model schemas drafted; **config compiler** (config → F´ params / sim / analysis) stubbed with uplink-overlay rule; one LEO template config.
- [ ] **Canonical state structs** (`EstimatedState`/`TruthState`) and **boundary typed-vector** wrappers defined and versioned (§8.0, §3.1).
- [ ] Repo skeleton (§22.3) + CI skeleton + license chosen.
- [ ] **GMAT installed**, golden-data generation harness scripted (`tools/gmat/`), first golden case captured (`tests/golden/`).
- [ ] **Docs site skeleton** stood up (Sphinx + numpydoc + PyData theme + Doxygen/Breathe + `refs.bib`), publishing to GitHub Pages, doc-build gate enabled.
- [ ] **Fault-injection hooks** designed into the sim/sensor/actuator models so the FDIR integration suite (§23.1.1) can script faults from day one.
- [ ] Glossary/acronym list started (for `CLAUDE.md` clarity).

---

## 24. Suggested Development Phasing (for `CLAUDE.md` build order)

Dependency-ordered so the suite is buildable and testable at every step:

- **Phase 0 — Foundations:** conventions (§3) including **boundary-only typed frame/unit vectors** and the **physical-constants registry**; **canonical state structs** (`EstimatedState`/`TruthState`, §8.0); math/frames/time/ephemeris libraries (fixed-size Eigen, JPL scalar-first quaternions, TAI + GPS/ECEF conversions); config schema + hardware-model library + **config compiler** (§19.3); repo + CI skeleton, **Requirements ICD**, **docs site + GMAT golden-data harness**, license.
- **Phase 1 — Truth sim core:** 6DOF + RK89, gravity (EGM2008, settable order), third-body (SPICE), drag (NRLMSIS 2.1), SRP + eclipse, IGRF-14, disturbance torques. Validate via conservation/analytic + **GMAT golden cases** (propagation/conversions).
- **Phase 2 — Sensor & actuator models + hardware library:** generic framework + each model with full error stacks; model-ID selection; GNSS reports GPS time/ECEF.
- **Phase 3 — FSW skeleton (F´) + two-process SITL:** topology, rate groups, telemetry/command/event/param, mode-manager stub, **plant↔FSW F´ TCP IPC + sim-time lockstep**, onboard time/EOP/Chebyshev ephemeris, persistence.
- **Phase 4 — Attitude determination:** initializers (TRIAD/QUEST), MEKF fine mode + **coarse mode (SS+MAG+IMU)**, multi-IMU/multi-sun-sensor fusion, validity flags + occlusion handling.
- **Phase 5 — Attitude control:** B-dot, PID, RW L-norm/L-∞ allocation **or** CMG steering (modular), momentum management + MTQ desaturation.
- **Phase 6 — Orbit determination & propagation:** GNSS-sim, onboard MEKF OD + self-covariance, multi-object propagation, batch LS, SGP4, CCSDS OEM (GMAT-validated).
- **Phase 7 — Guidance + full state machine:** pointing modes, slew planning with keep-out/keep-in cones, GS tracking, complete mode set.
- **Phase 8 — Maneuvering + interop outputs:** thruster targeting (SMA/altitude), Delta-V mode, pre/post-burn CCSDS/TLE, STK/FreeFlyer export.
- **Phase 9 — Subsystems:** simple power, thermal, comms/link budget; subsystem monitors into FDIR.
- **Phase 10 — FDIR:** monitors, isolation, responses, safing escalation across sensors/actuators/subsystems; **fault-injection integration suite** (sensor faults/loss, occlusions, GPS outage/spoofing, actuator faults, subsystem limits, cascades — §23.1.1).
- **Phase 11 — Monte Carlo + analysis tools:** dispersion framework, momentum/sizing, detumble MC, contact scheduling, link budget, consistency metrics, per-REQ margin reporting.
- **Phase 12 — Advanced / future:** CMG singularity refinements, main propulsion (chemical/electric), fuel slosh/flex, mass-depletion coupling, RPOD module fill-in, **live web tools** (§21.4).

Cross-cutting throughout: unit-to-integration tests, static analysis, sourced documentation, and traceability.

---

*End of design baseline. §18 decisions resolved; complete §23.7 before generating `CLAUDE.md`.*
