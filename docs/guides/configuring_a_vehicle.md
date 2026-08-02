# Configuring a Vehicle

A vehicle is defined entirely in configuration — there is no hardware catalog in
code (design doc §19.4). A spacecraft YAML references parts by `model_id`; the
config compiler ({doc}`/api/python`) resolves them against the hardware library,
validates against a Pydantic schema, and emits provenance-stamped artifacts. The
truth sim ({doc}`/api/sim_scenario`) reads only the compiled artifact.

How one hardware value travels from a datasheet to a running model — every layer
between the compiler and `fromParams` passes the parameter map through
uninterpreted:

```{mermaid}
flowchart LR
    DS[Vendor datasheet] -->|transcribed, cited| HW["config/hardware/&lt;kind&gt;/&lt;model&gt;.yaml"]
    SC["spacecraft YAML<br/>(model_id refs)"] --> CC
    HW --> CC["config compiler<br/>tools/configc (Pydantic)"]
    CC -->|provenance hash| SJ["sim_setup.json"]
    SJ --> LC["SimConfig loader<br/>sim/scenario/sim_config"]
    LC -->|"param map (uninterpreted)"| FP["Spec::fromParams<br/>(one place: SI conversion)"]
    FP --> M["truth model<br/>(Imu, StarTracker, RW, ...)"]
    CC -.->|same resolved object| FP2["F´ params / analysis<br/>artifacts"]
```

## Fields with no default

Most vehicle properties have a sensible default in the schema. Three do not, and
the compile fails if a spacecraft YAML omits them: `cp_offset_aero_m`,
`cp_offset_srp_m` and `residual_dipole_am2` — the moment arms and dipole that
produce the §5.3 disturbance torques, all Body-frame 3-vectors. A defaulted zero
lever arm is indistinguishable in a run's output from a perfectly balanced
vehicle, so "not measured yet" has to be written down as an explicit
`[0.0, 0.0, 0.0]`. The aerodynamic and optical centres of pressure are separate
fields because one is set by the ram area and the other by the illuminated area.

`residual_dipole_am2` at the spacecraft level is the whole vehicle's leftover
magnetisation — structure and harness — and is unrelated to the magnetorquer
catalog entry's field of the same name, which is one rod's remanent moment (the
half-width of its hysteresis loop). They are separate physical quantities that
happen to share a name and a unit; the spacecraft one produces the §5.3
disturbance torque, the catalog one shapes what a commanded dipole actually
becomes.

Whether each torque is actually applied is a *scenario* switch, not a vehicle
one: `gravity_gradient_torque_enabled`, `aero_torque_enabled`,
`srp_torque_enabled` and `residual_dipole_torque_enabled` under
`scenario.environment`, all defaulting to true. They are independent of the force
switches beside them — setting `aero_torque_enabled: false` keeps the drag force,
and so the orbit decay, while removing its couple, which is what an MC study
isolating one disturbance needs.

## Where a unit points

A mounted unit's orientation is a `unit→body` rotation, written **either** as
`mounting_dcm_row_major` (nine numbers, row-major 3×3) **or** as
`mounting_quaternion_wxyz` (four numbers, scalar-first `q0,q1,q2,q3` per the
repo's JPL convention, design doc §3.1). They are two spellings of one field:
the compiler converts the quaternion to the DCM every downstream consumer
already reads, and setting both is a validation error rather than a silent
precedence rule. Prefer the quaternion for anything you write by hand — it
cannot be made non-orthogonal by a typo, which a hand-edited DCM can.

For a sensor whose boresight is its **+Z** — every payload sensor (design doc
§6.3), and the star tracker — that rotation is the *entire* pointing
definition, and the third column of the resulting DCM is the boresight in body
axes. There is deliberately no separate boresight parameter: two ways to
express one orientation is two places for it to be wrong, and the failure
(a payload looking somewhere other than the analysis assumed) is silent.

A payload sensor's field of view is a **half**-angle, and its shape follows from
which half-angles the catalog entry sets — `half_fov_deg` alone is a cone,
`half_fov_x_deg` with `half_fov_y_deg` is rectangular or, when equal, square.
Writing a full angle where a half-angle was asked for is the likeliest way to
misconfigure one, so anything at or beyond 90° is rejected at vehicle build.

Vehicle configuration also carries **FSW tuning**. `spacecraft.fsw_parameters`
is a flat map keyed by fully-qualified F´ parameter name; given the FPP topology
dictionary (`--dictionary`), the compiler resolves each name to its generated
parameter ID and emits `PrmDb.dat`, the file `Svc::PrmDb` loads at startup. The
IDs are read from the dictionary rather than written by hand, so a parameter
file can never drift out of step with the flight build, and an unknown or unset
parameter fails the compile rather than the mission — there are no flight
defaults (design doc §19.3). The deployment points `prmDb` at the file with
`-P`; see `flight/PolarisFsw/README.md` for the bring-up recipe.

The attitude estimator's thirty-two values are worth reading as **four sets**,
because they fail differently. The fifteen coarse-chain values (the magnetic
white and systematic sigmas, the sun white sigma and the four terms its
systematic is composed from, `GyroArw`, `MinSinAngle`, `TriadGain`,
`MaxCoastSec`, `MaxDtSec`, `MaxMeasAgeSec`, the GNSS radius band) are what the
§10 Safe-mode floor runs on: one missing and the vehicle has no attitude at all.
The seven fine-mode values (`MekfRrw`, `MekfNisGate`, `MekfMaxCoastSec`,
`MekfBiasSigmaInit`, `MekfRefusalStreak`, `MekfNisStreak`,
`SeedMinObservability`) gate the MEKF only: one missing costs the fine mode and
emits `FineConfigInvalid`, leaving a flyable vehicle on the coarse solution. The
seven magnetometer-calibration values (`MagCalNominalFieldT`,
`MagCalMinFieldT`, `MagCalMaxFieldT`, `MagCalMinSamples`, `MagCalMinCoverage`,
`MagCalMaxCondition`, `MagCalMinImprovement`) gate neither: they are read when
`MAG_CAL_START` is commanded, and one missing refuses that *command* with
`MagCalRejected(CONFIG)` while the vehicle flies on exactly as before —
calibration is an activity, not a flight function, so it does not get a
per-cycle alert. The three Earth-albedo values (`SunAlbedoPeakRad`,
`SunAlbedoHalfFovRad`, `SunAlbedoBoresightBody`) gate only the sun-vector albedo
correction: one missing emits `AlbedoConfigInvalid` once and every cycle is then
weighted at `SigmaSunAlbedoUncorrRad`, which is how the vehicle flew before the
correction existed. All three describe **one** sun sensor — the unit's datasheet
albedo peak and field of view, and its mounting quaternion applied to the
sensor's +Z — so re-derive them if that unit's `model_id` or its mounting
changes. `SunAlbedoBoresightBody` is written as a YAML **list** of three numbers;
`fsw_parameters` accepts a list wherever the flight build declares an array-typed
parameter, and refuses one of the wrong length. The MEKF's angle random walk and
largest propagation step are the coarse chain's `GyroArw` and `MaxDtSec` — same
gyro, same rate group, so they are not duplicated.
`config/spacecraft/leo_smallsat.yaml` derives every one of the thirty-two from
the units that vehicle carries, in comments; re-derive them whenever a `model_id`
changes.

The sun pair carries **four** systematic values rather than one, and none of it
is redundancy. The estimator composes them per cycle as

```
sigma_sun_sys = hypot(albedo term, ephemeris term)
```

and picks each side independently. The **albedo** side is the sensor's:
`SigmaSunAlbedoRad` when the correction ran (it needs a position fix, a sunlit
Earth in the sensor's field, and an attitude to place that field with) and
`SigmaSunAlbedoUncorrRad` when it did not. The **ephemeris** side is the
*reference's*: `SigmaSunEphemPreciseRad` while the onboard DE440 tables cover the
epoch and `SigmaSunEphemRad` (the analytic fallback) when they do not — chosen by
the served `TableGrade`, which is a fact about the current upload rather than
anything you configure. A single pre-composed number would be wrong on three of
the four combinations. The component refuses either pair ordered the wrong way
(`ConfigInvalid`, estimator inert): a correction that made the measurement worse,
or an analytic fallback that beat the tables, is a configuration error rather
than a flight condition.

Two consequences worth knowing when you derive these. First, `SigmaSunAlbedoRad`
is a *floor*: the component adds `SunAlbedoPeakRad·σ_att/2` in quadrature, the
correction's own error from placing the Earth with an imperfect attitude. That is
runtime state, not tuning — there is nothing to configure — but it is why the
value you derive should be the *converged* budget rather than an average over
acquisition transients, which the component already handles. Second, an
**ephemeris upload needs no parameter change at all**: the vehicle carries both
grades, so it starts using the tighter term on the first cycle the tables answer
at `PRECISE`. Set `SigmaSunEphemPreciseRad` to what your uploaded tables are
actually worth — it is a parameter rather than an assumed zero precisely so that
claim is the ground's and not the code's.

The two albedo values are also the first parameter pair the compiler
**cross-checks against the hardware library**: `SunAlbedoPeakRad` and
`SunAlbedoHalfFovRad` must equal the first sun sensor's `albedo_error_deg` and
`half_fov_deg` converted to radians, or the compile fails naming both files. They
describe one physical quantity in two places, and unlike most such duplication a
mismatch here does not degrade gracefully — the flight correction subtracts a
model of the error the sim generates from the catalog value, so a stale
parameter removes an error the sensor never had, invisibly.

> **A successful calibration invalidates three of the values above.**
> `SigmaMagWhiteRad` and `SigmaMagSysRad` describe an *uncalibrated*
> magnetometer — on the reference vehicle the 33.7 mrad systematic is almost
> entirely the hard iron the fit removes — and `SeedMinObservability` was
> derived from the *ratio* of the sun and magnetic sigmas, so a ~16× tighter
> magnetic pair makes the shipped `0.0076` refuse every geometry in the band
> (≈`1.1e-4` preserves its 10°-separation meaning). Re-derive and uplink all
> three after a good fit; the ops procedure is in
> `flight/PolarisFsw/README.md`.

The parameter database holds at most `PRMDB_NUM_DB_ENTRIES` records and a longer
file loads *partially*, so `configc` refuses to emit one — a build error rather
than a vehicle with silently-invalid parameters. Polaris raises that limit from
the F´ default of 25 to **64** in `flight/config/PrmDbImplCfg.hpp`, a
`CONFIGURATION_OVERRIDES` module rather than a vendored copy of the F´ config
directory. `tools/configc/prmdb.py`'s `MAX_ENTRIES` must stay equal to it.

The hardware-library authoring contract — file layout, required keys and their
units, the datasheet-comment standard, and the step-by-step *add a catalog
entry* walkthrough — is the repository's `config/hardware/README.md`, included
here:

```{include} ../../config/hardware/README.md
```
