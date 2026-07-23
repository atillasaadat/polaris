"""Pydantic schema for the Polaris configuration system (design doc §19.1–19.2).

The spacecraft + scenario config and the hardware-model library are the single
source of truth (§19.3); these models are the validated boundary between the
human-edited YAML and every downstream consumer. Validation is strict —
``extra='forbid'`` rejects unknown keys so a typo fails the build instead of
silently dropping a parameter.

Frames & units follow the project convention (§3.1/§3.4): SI throughout, body
frame for mass properties, WGS84 geodetic for ground stations; every physical
field states its unit in its description.

This is the **GNC-core** draft (Push 3): mass properties, the sensor/actuator
suite referenced by hardware model-ID, per-mode control gains, epoch/environment,
ground stations, and Monte-Carlo dispersion hooks. RF / power / thermal sections
(§19.1) are added when a consumer needs them.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Vec3 = tuple[float, float, float]

# Hardware kinds the library models today; extend as device classes land.
HardwareKind = Literal[
    "imu",
    "star_tracker",
    "sun_sensor",
    "magnetometer",
    "gnss",
    "reaction_wheel",
    "magnetorquer",
    "thruster",
]


class _Strict(BaseModel):
    """Base model: forbid unknown keys so config typos fail validation."""

    model_config = ConfigDict(extra="forbid")


class HardwareModel(_Strict):
    """One parameterized hardware-model-library entry, keyed by model ID (§19.2).

    A spacecraft references a unit by ``model_id``; the compiler resolves that
    string against the library and inlines these params, so swapping the string
    swaps the modeled unit (REQ-CFG-002).
    """

    model_id: str = Field(description="unique model identifier, e.g. 'STIM300'")
    kind: HardwareKind = Field(description="device class")
    description: str = Field(default="", description="human-readable summary")
    params: dict[str, float] = Field(
        description="error/performance parameters (SI); keys are device-specific"
    )


class MountedUnit(_Strict):
    """A hardware unit installed on the vehicle: a library reference + placement."""

    name: str = Field(description="instance name on this vehicle, e.g. 'imu_a'")
    model_id: str = Field(description="hardware-library model ID to resolve")
    mounting_dcm_row_major: (
        tuple[float, float, float, float, float, float, float, float, float] | None
    ) = Field(
        default=None,
        description="unit→body rotation, row-major 3x3; identity if omitted",
    )
    spin_axis: Vec3 | None = Field(
        default=None,
        description=(
            "reaction-wheel/CMG spin axis in the body frame (need not be unit — it "
            "is normalised). The clean way to place a wheel: only the spin "
            "direction matters, not a full orientation. Populates the assembly's W "
            "matrix (design doc §7). Ignored for sensors, which use mounting_dcm"
        ),
    )
    noise_enabled: bool | None = Field(
        default=None,
        description=(
            "per-unit override of the scenario's sensor_noise_enabled (§6.2): set "
            "true/false to force this sensor's noise on/off regardless of the "
            "global switch; omit (null) to inherit it. Ignored for actuators"
        ),
    )


class InertiaTensor(_Strict):
    """Body-frame inertia tensor [kg·m²], symmetric (six unique components).

    Principal moments (diagonal) are physically positive for a real rigid body;
    products of inertia (off-diagonal) may be either sign.
    """

    ixx: float = Field(gt=0.0)
    iyy: float = Field(gt=0.0)
    izz: float = Field(gt=0.0)
    ixy: float = 0.0
    ixz: float = 0.0
    iyz: float = 0.0


class Spacecraft(_Strict):
    """Vehicle definition: mass properties + sensor/actuator suite + gains (§19.1)."""

    name: str = Field(description="vehicle name")
    mass_kg: float = Field(gt=0.0, description="total mass [kg]")
    com_m: Vec3 = Field(description="center of mass in body frame [m]")
    inertia_kgm2: InertiaTensor = Field(description="body-frame inertia [kg·m²]")
    sensors: list[MountedUnit] = Field(default_factory=list)
    actuators: list[MountedUnit] = Field(default_factory=list)
    gains: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="control gains keyed by mode, e.g. gains['detumble']['k_bdot']",
    )
    drag_area_m2: float = Field(
        default=0.06, gt=0.0, description="drag reference area [m²]"
    )
    drag_cd: float = Field(default=2.2, gt=0.0, description="drag coefficient [-]")
    srp_area_m2: float = Field(
        default=0.06, gt=0.0, description="SRP reference area [m²]"
    )
    srp_cr: float = Field(
        default=1.3, gt=0.0, description="SRP reflectivity coefficient [-]"
    )
    cp_offset_m: Vec3 = Field(
        default=(0.0, 0.0, 0.0),
        description="center of pressure offset from CoM, body frame [m]; "
        "the moment arm for aero/SRP disturbance torques",
    )
    residual_dipole_am2: Vec3 = Field(
        default=(0.0, 0.0, 0.0),
        description="residual magnetic dipole moment, body frame [A·m²]",
    )


class GroundStation(_Strict):
    """A ground station in WGS84 geodetic coordinates (§19.1)."""

    name: str
    lat_deg: float = Field(ge=-90.0, le=90.0, description="geodetic latitude [deg]")
    lon_deg: float = Field(ge=-180.0, le=180.0, description="longitude [deg]")
    alt_m: float = Field(description="height above WGS84 ellipsoid [m]")
    min_elevation_deg: float = Field(
        default=5.0, ge=0.0, le=90.0, description="mask elevation for contacts [deg]"
    )


class GnssFaultEvent(_Strict):
    """A time-windowed GNSS fault to inject during a run (design doc §9.2/§23.1.1).

    Distinct from the hardware catalog (what the receiver *is*): this is what the
    scenario *does* to it. The sim applies the fault while ``start_s`` <= t <
    ``stop_s`` (seconds since epoch) on the named receiver.
    """

    unit: str = Field(description="receiver instance name, e.g. 'gps_a'")
    type: Literal["outage", "spoof", "clock_jump"] = Field(
        description="outage=loss of fix; spoof=valid-but-offset fix; clock_jump=time-tag step"
    )
    start_s: float = Field(ge=0.0, description="fault onset [s since epoch]")
    stop_s: float = Field(
        gt=0.0, description="fault clears at this time [s since epoch]"
    )
    spoof_offset_ecef_m: Vec3 = Field(
        default=(0.0, 0.0, 0.0),
        description="ECEF position offset [m] for a spoof; ignored for other types",
    )
    clock_jump_s: float = Field(
        default=0.0,
        description="clock-bias step [s] for a clock_jump; ignored otherwise",
    )

    @field_validator("stop_s")
    @classmethod
    def _check_window(cls, v: float, info) -> float:
        start = info.data.get("start_s")
        if start is not None and v <= start:
            raise ValueError(f"stop_s ({v}) must exceed start_s ({start})")
        return v


class Environment(_Strict):
    """Force/torque environment toggles and fidelity for the scenario (§19.1)."""

    gravity_degree: int = Field(
        default=8, ge=0, description="spherical-harmonic gravity degree/order"
    )
    drag_enabled: bool = True
    srp_enabled: bool = True
    third_bodies: list[str] = Field(
        default_factory=lambda: ["sun", "moon"],
        description="third-body point-mass perturbers",
    )
    atmosphere: Literal["exponential", "nrlmsis"] = Field(
        default="exponential", description="density model backing the drag force"
    )
    magnetic_field: Literal["none", "igrf"] = Field(
        default="igrf", description="geomagnetic field model"
    )
    eclipse_enabled: bool = Field(
        default=True, description="apply the conical eclipse shadow factor to SRP"
    )
    occultation_atmosphere_km: float = Field(
        default=100.0,
        ge=0.0,
        description=(
            "optically obstructing atmosphere thickness above the surface [km], "
            "used by the optical-sensor occlusion model (design doc §6.1). The "
            "100 km default is the Karman line, right for visible-band blinding; "
            "a horizon sensor in the 15 um CO2 band sees a higher limb"
        ),
    )
    gnss_jamming_kml: str | None = Field(
        default=None,
        description=(
            "path to a KML of GNSS-jamming regions (design doc §9.2). When the "
            "sub-satellite point is inside a region the GNSS receiver loses its "
            "fix. Passed through to the sim verbatim; None disables jamming"
        ),
    )
    gnss_jamming_enabled: bool = Field(
        default=True,
        description=(
            "master switch for geographic GNSS jamming; set false to keep the KML "
            "referenced but turn jamming off for a run (design doc §9.2)"
        ),
    )
    sensor_noise_enabled: bool = Field(
        default=True,
        description=(
            "master switch for ALL sensor noise (IMU, star tracker, sun sensor, "
            "magnetometer, GNSS); false flies ideal sensors (measurement = truth) "
            "for a noise-free baseline or with/without-noise comparison. Actuators "
            "have no stochastic noise, so there is no equivalent switch for them"
        ),
    )
    gnss_noise_enabled: bool = Field(
        default=True,
        description=(
            "master switch for GNSS measurement noise; false flies a truth-perfect "
            "receiver. Applied on top of sensor_noise_enabled (both must be true)"
        ),
    )
    gnss_fault_events: list[GnssFaultEvent] = Field(
        default_factory=list,
        description="scheduled GNSS faults injected during the run (§9.2/§23.1.1)",
    )


class OrbitElements(_Strict):
    """Initial osculating Keplerian elements, Earth-centered inertial (§19.1)."""

    sma_km: float = Field(gt=0.0, description="semi-major axis [km]")
    ecc: float = Field(ge=0.0, lt=1.0, description="eccentricity [-], closed orbits")
    inc_deg: float = Field(ge=0.0, le=180.0, description="inclination [deg]")
    raan_deg: float = Field(description="right ascension of the ascending node [deg]")
    argp_deg: float = Field(description="argument of periapsis [deg]")
    true_anomaly_deg: float = Field(description="true anomaly at epoch [deg]")


class InitialState(_Strict):
    """Vehicle state at the scenario epoch: orbit + attitude + body rate (§19.1)."""

    orbit: OrbitElements = Field(description="initial osculating orbit")
    attitude_quaternion: tuple[float, float, float, float] = Field(
        default=(1.0, 0.0, 0.0, 0.0),
        description="body←ECI attitude quaternion, scalar-first (q0,q1,q2,q3) "
        "per the repo's JPL convention [-]",
    )
    body_rate_rad_s: Vec3 = Field(
        default=(0.0, 0.0, 0.0),
        description="body angular rate w.r.t. ECI, body frame [rad/s]",
    )

    @field_validator("attitude_quaternion")
    @classmethod
    def _check_normalised(
        cls, v: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        norm = math.sqrt(sum(c * c for c in v))
        if abs(norm - 1.0) > 1e-9:
            raise ValueError(
                f"attitude_quaternion must be normalised to within 1e-9, "
                f"got norm {norm!r} for {v!r}"
            )
        return v


class Propagation(_Strict):
    """Run duration, sample cadence, and RK89 step control (§19.1)."""

    duration_s: float = Field(gt=0.0, description="propagation duration [s]")
    output_step_s: float = Field(gt=0.0, description="trajectory sample cadence [s]")
    abs_tol: float = Field(
        default=1e-12, gt=0.0, description="integrator absolute tolerance [-]"
    )
    rel_tol: float = Field(
        default=1e-12, gt=0.0, description="integrator relative tolerance [-]"
    )
    max_step_s: float = Field(
        default=60.0, gt=0.0, description="integrator maximum step size [s]"
    )


class McDispersion(_Strict):
    """One Monte-Carlo dispersion hook over a config parameter (§19.1)."""

    param: str = Field(
        description="dotted path into the config, e.g. 'spacecraft.mass_kg'"
    )
    distribution: Literal["normal", "uniform"] = "normal"
    sigma: float = Field(ge=0.0, description="std-dev (normal) or half-width (uniform)")


class Scenario(_Strict):
    """Epoch / environment / ground network / MC dispersions for a run (§19.1)."""

    name: str = Field(description="scenario name")
    epoch_utc: str = Field(
        description="scenario start epoch, ISO-8601 UTC, e.g. '2026-01-01T00:00:00Z'"
    )
    initial_state: InitialState = Field(description="vehicle state at the epoch")
    propagation: Propagation = Field(description="how long and how finely to run")
    seed: int = Field(
        default=0,
        ge=0,
        description=(
            "master RNG seed; every stochastic source derives its own stream from "
            "it, so a run is bit-reproducible from {config, seed} (design doc §3.6)"
        ),
    )
    environment: Environment = Field(default_factory=Environment)
    ground_stations: list[GroundStation] = Field(default_factory=list)
    mc_dispersions: list[McDispersion] = Field(default_factory=list)

    @field_validator("epoch_utc")
    @classmethod
    def _check_iso8601(cls, v: str) -> str:
        # Validate the format at the boundary rather than deferring the failure to
        # a downstream consumer of the emitted artifacts.
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"epoch_utc must be ISO-8601, got {v!r}") from exc
        return v


class Config(_Strict):
    """A full spacecraft + scenario configuration — the source of truth (§19.1)."""

    spacecraft: Spacecraft
    scenario: Scenario
