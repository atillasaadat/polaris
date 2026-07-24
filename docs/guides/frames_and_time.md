# Frames, Attitude & Time

The conventions in this guide are **enforced by the type system** — most
mistakes here are compile errors, not runtime bugs. They are the single most
important thing to internalise before writing GNC math.

## Reference frames

Every vector and rotation is tagged with its frame at the type level
({doc}`/api/math`). A `Vec3<ECI>` and a `Vec3<Body>` are different types; adding
them does not compile. Raw `Eigen` is reached only at the kernel boundary via
`.eigen()`.

- **ECI** (GCRS/J2000) — the inertial frame the propagator and filter work in.
- **ECEF** (ITRS) — Earth-fixed; where GNSS reports and the geopotential is
  defined. The one time-dependent transform, the IAU 2006/2000A reduction, lives
  in {doc}`/api/frames` — *no ad-hoc rotations* (REQ-CONV-001/002).
- **Body** — the spacecraft structural frame.
- **LVLH / RIC** — orbit-relative frames built as pure functions of the ECI
  state ({doc}`/api/math`).

A rotation carries **both** frames: `Quat<Body, ECI>` reads "Body ← ECI" and
rotates an ECI vector into the Body frame. The compiler checks that a rotation's
input frame matches the vector it is applied to.

```{mermaid}
flowchart TD
    ECI["ECI (GCRS/J2000)<br/>inertial — propagation, estimation"]
    ECEF["ECEF (ITRS)<br/>Earth-fixed — GNSS, geopotential, IGRF"]
    Body["Body<br/>structural — sensors, actuators"]
    LVLH["LVLH / RIC<br/>orbit-relative (pure functions of ECI state)"]
    Sensor["Sensor / wheel mount frames<br/>(fixed mounting DCMs)"]

    ECI <-->|"IAU 2006/2000A + EOP<br/>(lib/math/frames — the only path)"| ECEF
    ECI <-->|"attitude Quat&lt;Body,ECI&gt;<br/>(truth or MEKF estimate)"| Body
    ECI -->|"r, v"| LVLH
    Body <-->|"mounting_dcm (config)"| Sensor
```

## Attitude — one representation, quaternions

Attitude is a **JPL scalar-first unit quaternion** $q = [q_0, q_1, q_2, q_3]$,
$q_0 \ge 0$, everywhere: truth propagation, the estimator reference, telemetry.
The truth plant integrates the kinematics

$$
\dot{q} = \tfrac{1}{2}\,\Omega(\boldsymbol{\omega})\,q, \qquad
\Omega(\boldsymbol{\omega}) =
\begin{bmatrix} 0 & -\boldsymbol{\omega}^{\mathsf T} \\
\boldsymbol{\omega} & -[\boldsymbol{\omega}]_\times \end{bmatrix}
$$

and renormalises after every accepted RK89 step. Quaternions are chosen over Modified Rodrigues Parameters
deliberately — an MRP shadow-set switch near 180° is a mid-step discontinuity
the adaptive integrator cannot tolerate, while a quaternion has no singularity
anywhere on SO(3) (design doc §5.1). The onboard MEKF uses the same quaternion
reference; its 3-parameter error is only the covariance parameterisation, never
a second global state (§8.1).

## Time — the TAI master clock

The onboard master clock is **TAI**, counted as an int64 nanosecond value; time
scales are strongly typed so mixing them is a compile error ({doc}`/api/time`).

$$
\mathrm{TAI} = \mathrm{GPS} + 19\,\mathrm{s}, \qquad
\mathrm{TT} = \mathrm{TAI} + 32.184\,\mathrm{s}, \qquad
\mathrm{UTC} = \mathrm{TAI} - \Delta AT(t)
$$

- `TAI = GPS + 19 s` is applied when a GNSS fix is ingested.
- TDB is a periodic term off TT, used only as the ephemeris argument.
- **UTC is ground-facing only** — $\Delta AT(t)$ is the leap-second count (a
  table lookup, not an equation); the flight side never runs on it.

A run is **bit-reproducible from `{config, seed}`**, which requires that all
logic key off sim time, never the wall clock, and that every stochastic source
draw from a seeded per-unit stream ({doc}`/api/random`, design doc §3.5).
