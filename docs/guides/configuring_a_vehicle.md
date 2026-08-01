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

Vehicle configuration also carries **FSW tuning**. `spacecraft.fsw_parameters`
is a flat map keyed by fully-qualified F´ parameter name; given the FPP topology
dictionary (`--dictionary`), the compiler resolves each name to its generated
parameter ID and emits `PrmDb.dat`, the file `Svc::PrmDb` loads at startup. The
IDs are read from the dictionary rather than written by hand, so a parameter
file can never drift out of step with the flight build, and an unknown or unset
parameter fails the compile rather than the mission — there are no flight
defaults (design doc §19.3). The deployment points `prmDb` at the file with
`-P`; see `flight/PolarisFsw/README.md` for the bring-up recipe.

The attitude estimator's nineteen values are worth reading as **two sets**,
because they fail differently. The twelve coarse-chain values (sun/magnetic
white and systematic sigmas, `GyroArw`, `MinSinAngle`, `TriadGain`,
`MaxCoastSec`, `MaxDtSec`, `MaxMeasAgeSec`, the GNSS radius band) are what the
§10 Safe-mode floor runs on: one missing and the vehicle has no attitude at all.
The seven fine-mode values (`MekfRrw`, `MekfNisGate`, `MekfMaxCoastSec`,
`MekfBiasSigmaInit`, `MekfRefusalStreak`, `MekfNisStreak`,
`SeedMinObservability`) gate the MEKF only: one missing costs the fine mode and
emits `FineConfigInvalid`, leaving a flyable vehicle on the coarse solution. The
MEKF's angle random walk and largest propagation step are the coarse chain's
`GyroArw` and `MaxDtSec` — same gyro, same rate group, so they are not
duplicated. `config/spacecraft/leo_smallsat.yaml` derives every one of the
nineteen from the units that vehicle carries, in comments; re-derive them
whenever a `model_id` changes.

The hardware-library authoring contract — file layout, required keys and their
units, the datasheet-comment standard, and the step-by-step *add a catalog
entry* walkthrough — is the repository's `config/hardware/README.md`, included
here:

```{include} ../../config/hardware/README.md
```
