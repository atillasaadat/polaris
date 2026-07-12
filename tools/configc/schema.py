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


class GroundStation(_Strict):
    """A ground station in WGS84 geodetic coordinates (§19.1)."""

    name: str
    lat_deg: float = Field(ge=-90.0, le=90.0, description="geodetic latitude [deg]")
    lon_deg: float = Field(ge=-180.0, le=180.0, description="longitude [deg]")
    alt_m: float = Field(description="height above WGS84 ellipsoid [m]")
    min_elevation_deg: float = Field(
        default=5.0, ge=0.0, le=90.0, description="mask elevation for contacts [deg]"
    )


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
