"""The Polaris config compiler (design doc §19.3, REQ-CFG-001/003).

One mechanism turns the single source-of-truth config (spacecraft + scenario +
hardware-library references) into every downstream artifact, so the three
representations that must agree — YAML config, F´ ``ParameterDb`` params, and sim
setup — cannot drift. The pipeline is:

    load hardware library  ->  load + validate config  ->  resolve model-IDs
      ->  one resolved/validated object  ->  provenance hash  ->  emit artifacts

Every consumer reads the derived artifacts, never the raw YAML (REQ-CFG-001), and
each artifact records the source config hash so any FSW param / sim run / fixture
is traceable to the exact config that produced it (REQ-CFG-003).

**Cross-checks are half of what this module does.** A handful of quantities exist
twice by design — once as sim/hardware truth, once as the ``flight.*`` parameter
the FSW runs on — and editing one side alone is invisible: nothing crashes, no
test fails, and the flight software believes something the vehicle is not. Those
pairings are declared here (``_CATALOG_PAIRS``, ``_VEHICLE_PAIRS``, and the
per-unit direction sets above them) and refused when they disagree, because the
compile is the last place the two are in the same room. Divergence that is a real
design decision is *declarable* rather than silently exempt — see
``schema.ParameterDivergence`` — since a check with no legitimate escape is a
check somebody eventually deletes.

Four artifacts. Three are JSON: ``sim_setup.json`` is the real truth-sim input,
while ``fprime_params.json`` and ``analysis_inputs.json`` are readable summaries
whose *encoding* is still provisional. The fourth, ``PrmDb.dat``, is the flight
article: the binary ``Svc::PrmDb`` parameter file the deployment loads at
startup, encoded by :mod:`configc.prmdb` against the FPP-generated topology
dictionary. It is emitted only when a dictionary is supplied, because the
parameter IDs it needs exist only in a built flight deployment.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .orbit import keplerian_to_cartesian
from .prmdb import PrmDbError, build_param_file, load_dictionary
from .schema import Config, HardwareModel, MountedUnit


class ConfigError(ValueError):
    """A config that is structurally valid YAML but semantically wrong.

    Raised for unresolved hardware model-IDs and malformed library entries —
    failures Pydantic's per-file validation cannot catch on its own.
    """


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML\n{exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a YAML mapping, got {type(data).__name__}")
    return data


def load_hardware_library(hardware_dir: Path) -> dict[str, HardwareModel]:
    """Load every ``*.yaml`` under @p hardware_dir, keyed by model ID (REQ-CFG-002).

    The search is recursive, so the library can be organized into per-kind
    subdirectories (``imu/``, ``star_tracker/``, …); a model is located by its
    ``model_id``, never its path, so the layout is free to change.

    Raises ConfigError on a missing directory, a duplicate model ID, or a
    malformed entry — so a typo'd ``--hardware`` path fails clearly here instead
    of surfacing later as a spurious "unknown model_id".
    """
    if not hardware_dir.is_dir():
        raise ConfigError(f"{hardware_dir}: hardware library directory not found")
    library: dict[str, HardwareModel] = {}
    for path in sorted(hardware_dir.rglob("*.yaml")):
        try:
            model = HardwareModel.model_validate(_load_yaml(path))
        except ValidationError as exc:
            raise ConfigError(f"{path}: invalid hardware model\n{exc}") from exc
        if model.model_id in library:
            raise ConfigError(
                f"{path}: duplicate hardware model_id '{model.model_id}' "
                f"(already defined)"
            )
        library[model.model_id] = model
    return library


def load_config(config_path: Path) -> Config:
    """Load and validate one spacecraft+scenario config file (§19.1)."""
    try:
        return Config.model_validate(_load_yaml(config_path))
    except ValidationError as exc:
        raise ConfigError(f"{config_path}: invalid config\n{exc}") from exc


def _mounting_dcm(unit: MountedUnit) -> tuple[float, ...] | None:
    """This unit's unit→body rotation as a row-major 3x3, or None for identity.

    A mounting may be written either way round (schema ``MountedUnit``); the
    quaternion form is converted here so every downstream consumer sees one
    representation. The matrix is the **JPL passive attitude matrix** of
    ``lib/math/quaternion.hpp`` (Trawny & Roumeliotis Eq. 78),

    ``A(q) = (2*q0^2 - 1) I - 2*q0 [q_v x] + 2 q_v q_v^T``,

    with ``v_body = A(q) v_unit`` — the same formula the C++ side applies, since
    a mounting that rotated one way in the compiler and the other in the sim
    would place every payload's boresight at its mirror image.
    """
    if unit.mounting_quaternion_wxyz is None:
        return unit.mounting_dcm_row_major
    q0, q1, q2, q3 = unit.mounting_quaternion_wxyz
    d = 2.0 * q0 * q0 - 1.0
    rows = (
        (d + 2.0 * q1 * q1, 2.0 * (q1 * q2 + q0 * q3), 2.0 * (q1 * q3 - q0 * q2)),
        (2.0 * (q1 * q2 - q0 * q3), d + 2.0 * q2 * q2, 2.0 * (q2 * q3 + q0 * q1)),
        (2.0 * (q1 * q3 + q0 * q2), 2.0 * (q2 * q3 - q0 * q1), d + 2.0 * q3 * q3),
    )
    return tuple(value for row in rows for value in row)


def _resolve_units(
    units: list[MountedUnit], library: dict[str, HardwareModel], role: str
) -> list[dict[str, Any]]:
    """Inline each unit's referenced hardware-library params, or fail (REQ-CFG-002)."""
    resolved = []
    for unit in units:
        model = library.get(unit.model_id)
        if model is None:
            raise ConfigError(
                f"{role} '{unit.name}' references unknown hardware model_id "
                f"'{unit.model_id}' — not in the library"
            )
        resolved.append(
            {
                "name": unit.name,
                "model_id": unit.model_id,
                "kind": model.kind,
                "params": model.params,
                "mounting_dcm_row_major": _mounting_dcm(unit),
                "spin_axis": unit.spin_axis,
                "dipole_axis": unit.dipole_axis,
                "thrust_axis": unit.thrust_axis,
                "mounting_position_m": unit.mounting_position_m,
                "noise_enabled": unit.noise_enabled,
            }
        )
    return resolved


#: FSW parameter name -> (sun-sensor catalog key, degrees-to-radians).
#:
#: The Earth-albedo correction (design doc §8.1) is configured with two numbers
#: that are *already* in the sun sensor's hardware-library entry, because the
#: flight side needs them in radians through the §19.3 parameter path and the sim
#: side reads the catalog directly. That makes it the first parameter pair where
#: sim/flight agreement is exactness-critical: the correction subtracts a model
#: of the error the sim generates, so a mismatch does not degrade gracefully — it
#: removes an error the sensor never had, and nothing downstream can see it.
#: Hence the cross-check below rather than a comment asking for care.
_ALBEDO_PARAM_TO_CATALOG_KEY = {
    "flight.attitudeEstimator.SunAlbedoPeakRad": "albedo_error_deg",
    "flight.attitudeEstimator.SunAlbedoHalfFovRad": "half_fov_deg",
}


#: Per-unit albedo boresights, flattened three at a time in sun-sensor build
#: order (design doc §8.2). Same failure mode as the pair above and the same
#: remedy: the boresight the FSW corrects with must be the mounting the sim
#: places the unit at, and a mismatch *scales* the correction instead of failing
#: it — so it is checked here rather than trusted.
_ALBEDO_BORESIGHTS_PARAM = "flight.attitudeEstimator.SunAlbedoBoresightsBody"


def _boresight_from_mounting(unit: dict[str, Any]) -> tuple[float, float, float]:
    """This unit's body-frame boresight: its unit->body rotation applied to +Z.

    The sensor boresight is unit +Z by convention (design doc §6.3), so this is
    the third column of the mounting matrix `_mounting_dcm` produced. A unit with
    no mounting is identity, hence body +Z.
    """
    dcm = unit.get("mounting_dcm_row_major")
    if dcm is None:
        return (0.0, 0.0, 1.0)
    return (float(dcm[2]), float(dcm[5]), float(dcm[8]))


def _check_albedo_parameters(body: dict[str, Any]) -> None:
    """Refuse a vehicle whose albedo tuning contradicts its sun sensors' catalog entries.

    The peak error and field of view describe the sun-sensor *part*, and are
    checked against the **first** unit — the vehicle's sun sensors are one model
    today, and a mixed suite would need those per unit too (§8.2). The
    boresights are per **unit** and are checked against every installed one.
    Silent on a vehicle that sets neither side, so a config with no sun sensor or
    no albedo tuning compiles unchanged.
    """
    sc = body["spacecraft"]
    fsw = sc.get("fsw_parameters", {})
    sun_sensors = [u for u in sc.get("sensors", []) if u.get("kind") == "sun_sensor"]
    if not sun_sensors:
        return
    unit = sun_sensors[0]
    params = unit.get("params", {})

    for name, catalog_key in _ALBEDO_PARAM_TO_CATALOG_KEY.items():
        if name not in fsw or catalog_key not in params:
            continue
        expected = math.radians(float(params[catalog_key]))
        actual = float(fsw[name])
        # Tolerance is a rounding allowance, not a physical one: the YAML carries
        # the radian value to five decimals, so 1e-5 admits an honest transcription
        # and refuses a stale one.
        if abs(actual - expected) > 1.0e-5:
            raise ConfigError(
                f"{name} = {actual!r} rad contradicts the sun sensor's catalog entry: "
                f"unit '{unit['name']}' ({unit['model_id']}) declares "
                f"{catalog_key} = {params[catalog_key]!r} deg = {expected:.5f} rad.\n"
                f"These describe the same physical quantity — the flight albedo "
                f"correction subtracts a model of the error the sim generates from "
                f"the catalog value, so a mismatch removes an error the sensor never "
                f"had. Fix the vehicle config's fsw_parameters or the hardware entry, "
                f"whichever is stale."
            )

    _check_per_unit_boresights(
        fsw, _ALBEDO_BORESIGHTS_PARAM, sun_sensors, "sun sensor", _WHY_ALBEDO
    )


#: Per-unit star-tracker boresights, flattened three at a time in star-tracker
#: build order (design doc §8.2). Same shape and same failure mode as the albedo
#: set: the FSW builds each tracker's measurement covariance
#: `R = σ_xy²(I − b bᵀ) + σ_z² b bᵀ` from this boresight, so a wrong one does not
#: fail — it points the tight and loose axes of R in the wrong directions, which
#: quietly discards the whole reason two non-parallel trackers are carried.
_ST_BORESIGHTS_PARAM = "flight.attitudeEstimator.StBoresightsBody"

_WHY_ALBEDO = (
    "The flight albedo correction places the Earth in *this* unit's field, so a "
    "wrong boresight scales the correction rather than failing it."
)

_WHY_STAR_TRACKER = (
    "The flight fusion builds this unit's measurement covariance about *this* "
    "boresight (loose about it, tight across it), so a wrong one points the weak "
    "axis of R in the wrong direction and silently gives back what the second, "
    "non-parallel tracker was carried for."
)


def _axis_key(key: str):
    """Expected-direction accessor reading a named vector key off the unit."""

    def expected(unit: dict[str, Any]) -> tuple[float, float, float]:
        axis = unit.get(key)
        if axis is None:
            # No axis declared: nothing to contradict, so the caller's slot is
            # accepted by pointing the expectation at itself. Handled by the
            # zero-vector escape below rather than here.
            return (0.0, 0.0, 0.0)
        return (float(axis[0]), float(axis[1]), float(axis[2]))

    return expected


def _check_per_unit_boresights(
    fsw: dict[str, Any],
    param: str,
    units: list[dict[str, Any]],
    kind_label: str,
    why: str,
    expected_fn=None,
) -> None:
    """Refuse a flattened per-unit direction parameter that contradicts the config.

    Shared by the albedo set, the star-tracker set and the Phase-5 wheel/rod axis
    sets: all are `GncMaxUnits x 3` flat F64 arrays in build order, all use the
    zero vector as "not installed / not characterised", and all fail the same way
    — by scaling or mis-orienting a model rather than by erroring — which is
    exactly why the check exists rather than a comment asking for care. Silent
    when the parameter is absent, so a config that sets neither side compiles
    unchanged.

    @p expected_fn maps a resolved unit to the direction the parameter must
    agree with; it defaults to the unit's boresight (its mounting applied to +Z),
    which is what the sensor sets mean. The actuator sets pass an accessor for
    ``spin_axis`` / ``dipole_axis`` instead, since a wheel or a rod is placed by
    its axis rather than by a full mounting.
    """
    boresights = fsw.get(param)
    if boresights is None:
        return
    if expected_fn is None:
        expected_fn = _boresight_from_mounting
    for index, unit in enumerate(units):
        slot = boresights[3 * index : 3 * index + 3]
        if len(slot) < 3:
            raise ConfigError(
                f"{param} has no slot for {kind_label} "
                f"{index} ('{unit['name']}'): the parameter carries "
                f"{len(boresights) // 3} slots against {len(units)} installed units."
            )
        written = tuple(float(v) for v in slot)
        # The zero vector is the configured "no value for this unit" and is always
        # allowed — a mounting nobody has characterised is a legitimate state.
        if written == (0.0, 0.0, 0.0):
            continue
        expected = expected_fn(unit)
        if expected == (0.0, 0.0, 0.0):
            # The config declares no direction for this unit, so there is nothing
            # for the parameter to contradict.
            continue
        # Compared as an **angle**, not component-wise. These are directions, and
        # what matters is where the boresight points: a component-wise tolerance
        # is neither rotation-invariant nor scale-invariant, so it would reject an
        # honestly-written unit vector for round-off in one component while
        # accepting a vector of the wrong length pointing the right way. The angle
        # is formed as atan2 of the cross-product norm against the dot product
        # (lib/README.md), which stays conditioned near zero where acos does not.
        cross = (
            written[1] * expected[2] - written[2] * expected[1],
            written[2] * expected[0] - written[0] * expected[2],
            written[0] * expected[1] - written[1] * expected[0],
        )
        cross_norm = math.sqrt(sum(c * c for c in cross))
        dot = sum(a * b for a, b in zip(written, expected))
        angle = math.atan2(cross_norm, dot)
        if angle > 1.0e-6:
            raise ConfigError(
                f"{param} slot {index} = {written} points "
                f"{math.degrees(angle):.4f} deg away from the mounting of {kind_label} "
                f"'{unit['name']}', whose boresight (unit +Z through its mounting) "
                f"is {expected}. The tolerance is 1e-6 rad on the angle between them.\n"
                f"{why} "
                f"Fix the parameter or the mounting_quaternion_wxyz, whichever is stale; "
                f"write the zero vector to skip this unit."
            )


def _check_star_tracker_parameters(body: dict[str, Any]) -> None:
    """Refuse a vehicle whose star-tracker boresights contradict their mountings (§8.2)."""
    sc = body["spacecraft"]
    fsw = sc.get("fsw_parameters", {})
    trackers = [u for u in sc.get("sensors", []) if u.get("kind") == "star_tracker"]
    if not trackers:
        return
    _check_per_unit_boresights(
        fsw, _ST_BORESIGHTS_PARAM, trackers, "star tracker", _WHY_STAR_TRACKER
    )

    # The king tracker defines the body frame (design doc §8.2), so an index
    # outside the installed set is not a tuning error to discover in flight — it
    # names a unit that does not exist, and the estimator would then never reach
    # its finest mode while reporting nothing more specific than StConfigInvalid.
    king = fsw.get("flight.attitudeEstimator.StKingUnit")
    if king is not None and not 0 <= int(king) < len(trackers):
        raise ConfigError(
            f"flight.attitudeEstimator.StKingUnit = {king} names no installed star "
            f"tracker: this vehicle carries {len(trackers)} "
            f"({', '.join(u['name'] for u in trackers)}).\n"
            f"The king tracker's mounting *defines* the body frame, so this index is "
            f"a vehicle-integration decision, not a tuning knob."
        )


#: Per-unit actuator axes for the §8.5 control layer, flattened three at a time
#: in vehicle build order. Same shape and same silent failure mode as the sensor
#: boresight sets: a wrong wheel axis does not error, it allocates the commanded
#: torque onto a geometry the vehicle does not have — which reads as a controller
#: that points slightly wrong and slowly gets worse. A wrong rod axis clamps the
#: dipole on the wrong body axis.
_WHEEL_AXES_PARAM = "flight.attitudeController.WheelAxesBody"
_MTQ_AXES_PARAM = "flight.attitudeController.MtqAxesBody"

_WHY_WHEEL_AXES = (
    "The flight allocation solves A u = tau on *these* columns, so a wrong axis "
    "distributes the commanded torque over a geometry the vehicle does not have."
)

_WHY_MTQ_AXES = (
    "The flight controller resolves the commanded body dipole onto *these* axes "
    "and clamps each rod against its own rating, so a wrong axis saturates the "
    "wrong rod."
)


# --- FSW parameters that restate sim-side truth (§19.3) -----------------------
#
# Several physical quantities exist twice by design: once as the truth the sim
# and the hardware carry (a `config/hardware/**` catalog entry, a `spacecraft.*`
# field) and once as the flight parameter the FSW runs on. Changing one without
# the other is the most invisible defect this repo has: nothing crashes, no test
# fails, and the flight software simply believes something the vehicle is not.
# The instance that forced this table generalised from the albedo and axis checks
# above — a wheel re-sized to 0.002 N·m peak torque against a flight limit still
# reading 0.025, i.e. an FSW commanding 12.5x what the wheel could deliver and
# every torque margin derived from it optimistic by the same factor.
#
# So each pairing is declared here and checked, rather than trusted to a comment
# asking for care. Divergence that is a real design decision is *declarable*
# (`spacecraft.fsw_parameter_divergence`, schema `ParameterDivergence`) — a check
# strict enough to have no legitimate escape is a check the next engineer deletes
# wholesale.


class _Reduce:
    """How one flight scalar is expected to describe several installed units.

    ``BOUND`` — a capability rating (peak torque, rated dipole). A mixed array is
    bounded by its weakest unit, so the flight limit must be the **minimum**
    across the suite; this matches `analysis.control.vehicle._wheel_catalog_min`,
    which reads the same catalog for the same reason.

    ``UNIFORM`` — a model coefficient the flight side applies identically to every
    unit (bearing friction, rotor inertia). A single scalar cannot describe a
    mixed array at all, so a suite that does not share one value is refused
    outright rather than reduced to some representative of it.
    """

    BOUND = "bound"
    UNIFORM = "uniform"


def _catalog(key: str):
    """Per-unit accessor for a hardware-catalog scalar; None when unspecified.

    None and 0.0 are different answers and the distinction is load-bearing: a
    catalog that does not carry the key says nothing for the parameter to
    contradict, while one that declares zero (a frictionless fixture) is an
    assertion the flight value has to match.
    """

    def value(unit: dict[str, Any]) -> float | None:
        raw = unit.get("params", {}).get(key)
        return None if raw is None else float(raw)

    return value


def _rotor_inertia(unit: dict[str, Any]) -> float | None:
    """A wheel's rotor inertia [kg·m²], derived from its catalog entry.

    ``I = h_max / omega_max`` — the same relation `ReactionWheelSpec::inertia()`
    applies in the sim (sim/actuators/reaction_wheel.hpp), so the flight number is
    a *derivation* of two catalog values rather than a third independent one.
    None when either input is missing or the speed is not positive: there is then
    no derivation to compare against, which is not the same as one that comes out
    zero.
    """
    params = unit.get("params", {})
    momentum, speed_rpm = params.get("max_momentum_nms"), params.get("max_speed_rpm")
    if momentum is None or speed_rpm is None or float(speed_rpm) <= 0.0:
        return None
    return float(momentum) / (float(speed_rpm) * math.pi / 30.0)


#: (parameter, actuator kind, truth description, per-unit value, reduction,
#:  relative tolerance, why a mismatch is invisible).
#:
#: The tolerances are relative and per-pair because the pairs are not the same
#: kind of number: a transcription of one catalog scalar has no round-off to
#: allow for (1e-9 admits float-repr noise and nothing else), while a *derived*
#: quantity is written to the digits a human would write (1e-4 admits the five
#: significant figures of `4.7746e-5` against an exact 4.774648...e-5).
_CATALOG_PAIRS: tuple[tuple[str, str, str, Any, str, float, str], ...] = (
    (
        "flight.attitudeController.WheelMaxTorqueNm",
        "reaction_wheel",
        "max_torque_nm",
        _catalog("max_torque_nm"),
        _Reduce.BOUND,
        1.0e-9,
        "The allocation clamps every wheel command to this limit, so a value above "
        "the hardware's rating commands torque the wheel cannot deliver and makes "
        "every margin computed from it optimistic by the same factor — and the "
        "vehicle reports saturation nowhere, because as far as the FSW knows it "
        "never saturated.",
    ),
    (
        "flight.attitudeController.WheelCapacityNms",
        "reaction_wheel",
        "max_momentum_nms",
        _catalog("max_momentum_nms"),
        _Reduce.BOUND,
        1.0e-9,
        "The per-wheel capacity monitor alarms at a fraction of this on the "
        "largest single wheel — the only alarm that sees null-space momentum — so "
        "a value above the hardware's rating lets a wheel walk to its stop with "
        "the monitor still quiet, and every body-momentum threshold quiet with it.",
    ),
    (
        "flight.attitudeController.WheelInertiaKgm2",
        "reaction_wheel",
        "max_momentum_nms / (max_speed_rpm * 2*pi/60)",
        _rotor_inertia,
        _Reduce.UNIFORM,
        1.0e-4,
        "This is how the FSW turns tachometer speed into stored momentum, so a "
        "stale value mis-scales the entire momentum budget: the envelope, the "
        "desaturation hysteresis and the total-momentum telemetry all move "
        "together, which is exactly what makes the error look self-consistent.",
    ),
    (
        "flight.attitudeController.WheelDryFrictionNm",
        "reaction_wheel",
        "dry_friction_nm",
        _catalog("dry_friction_nm"),
        _Reduce.UNIFORM,
        1.0e-9,
        "The flight friction feedforward commands -tau_f on every wheel from this "
        "number, so a value larger than the hardware's real friction "
        "over-compensates — the one direction that makes the vehicle worse than no "
        "feedforward at all.",
    ),
    (
        "flight.attitudeController.WheelViscousFrictionNmS",
        "reaction_wheel",
        "viscous_friction_nm_s",
        _catalog("viscous_friction_nm_s"),
        _Reduce.UNIFORM,
        1.0e-9,
        "Same feedforward as the dry term and the same asymmetry: it is inverted "
        "against wheel speed, so on a loaded array an over-stated coefficient is a "
        "secular torque error that grows with the speed the wheels run at.",
    ),
    (
        "flight.attitudeController.BdotMaxDipoleAm2",
        "magnetorquer",
        "max_dipole_am2",
        _catalog("max_dipole_am2"),
        _Reduce.BOUND,
        1.0e-9,
        "B-dot clamps each rod to this rating independently, which is what keeps a "
        "saturated detumble dissipative (lib/gnc/bdot.hpp). A limit above the rod's "
        "rating moves the clamp into the driver, where it is no longer per-axis — "
        "and detumble convergence was argued on the per-axis one.",
    ),
)

#: The pairs whose truth side is not the hardware catalog but the vehicle's own
#: mass/magnetic properties — the fields the *sim* propagates with. Same defect
#: class, one step closer: these are literally the same numbers written twice in
#: one file, which is exactly the edit that gets made on one line and not the
#: other. (parameter, spacecraft field, accessor, description, why).
_VEHICLE_PAIRS: tuple[tuple[str, str, Any, str, str], ...] = (
    (
        "flight.attitudeController.InertiaBodyKgm2",
        "inertia_kgm2",
        lambda v: tuple(float(v[k]) for k in ("ixx", "iyy", "izz")),
        "spacecraft.inertia_kgm2 (ixx, iyy, izz)",
        "The flight controller forms the total system momentum H = J*omega + "
        "h_wheels and its gravity-gradient feedforward on this tensor, so a stale "
        "diagonal biases both against a vehicle the sim is propagating with the "
        "other one — a modelling error that reads as a plant the controller almost "
        "fits. Only the diagonal is compared, because the flight parameter is a "
        "diagonal: zero products of inertia are an assumption the FSW and the §8.5 "
        "SISO analysis both state.",
    ),
    (
        "flight.attitudeController.ResidualDipoleAm2",
        "residual_dipole_am2",
        lambda v: tuple(float(c) for c in v),
        "spacecraft.residual_dipole_am2",
        "The tier-1 disturbance feedforward subtracts m_res x B with this vector "
        "while the sim generates the torque from the other one, so a mismatch does "
        "not fail — it feeds forward a disturbance the vehicle does not have and "
        "leaves the one it does have to the integrator.",
    ),
)

#: Every parameter the cross-checks above cover. A declared divergence naming
#: anything outside this set is refused: it is either a typo or an exemption left
#: behind by a check that no longer exists, and both read as protection that is
#: not there.
_CROSS_CHECKED_PARAMS = frozenset(
    [pair[0] for pair in _CATALOG_PAIRS]
    + [pair[0] for pair in _VEHICLE_PAIRS]
    + [
        "flight.orbitEstimator.DragBallisticCoeffM2PerKg",
        "flight.attitudeController.WheelBiasNms",
    ]
)


def _close(actual: float, expected: float, rtol: float) -> bool:
    # Relative, not exact: these are floats read back through YAML and (for the
    # derived pairs) written to the digits a human writes. `math.isclose` with an
    # absolute floor so a legitimate zero compares equal to itself.
    return math.isclose(actual, expected, rel_tol=rtol, abs_tol=1.0e-300)


def _divergence_declared(
    divergences: dict[str, Any],
    param: str,
    expected: tuple[float, ...],
    truth: str,
    rtol: float,
) -> bool:
    """True when @p param carries a valid declared divergence from @p expected.

    Raises when the declaration is stale — its ``catalog_value`` no longer being
    the truth means the waiver was granted against a part this vehicle no longer
    carries, and a waiver that outlives its subject is worse than no check.
    """
    declared = divergences.get(param)
    if declared is None:
        return False
    raw = declared["catalog_value"]
    written = tuple(float(v) for v in (raw if isinstance(raw, list) else [raw]))
    if len(written) != len(expected) or not all(
        _close(w, e, rtol) for w, e in zip(written, expected)
    ):
        raise ConfigError(
            f"{param} declares a deliberate divergence from "
            f"catalog_value {_fmt(written)}, but this vehicle's {truth} now reads "
            f"{_fmt(expected)}.\n"
            f"The declaration was written against a value the config no longer "
            f"carries, so it is exempting the parameter from a comparison nobody "
            f"has made. Re-decide the divergence against the current hardware: "
            f"update catalog_value and the reason, or delete the entry."
        )
    return True


def _fmt(values: tuple[float, ...]) -> str:
    return repr(values[0]) if len(values) == 1 else repr(list(values))


def _refuse_mismatch(
    param: str,
    actual: tuple[float, ...],
    expected: tuple[float, ...],
    truth: str,
    where: str,
    why: str,
) -> None:
    raise ConfigError(
        f"{param} = {_fmt(actual)} contradicts the vehicle it flies on: "
        f"{truth} = {_fmt(expected)} ({where}).\n"
        f"{why}\n"
        f"Fix whichever is stale — the fsw_parameters entry in the vehicle config, "
        f"or the value it restates. If the divergence is deliberate, declare it "
        f"rather than loosening the check:\n"
        f"  spacecraft:\n"
        f"    fsw_parameter_divergence:\n"
        f"      {param}:\n"
        f"        catalog_value: {_fmt(expected)}\n"
        f"        reason: <why this vehicle deliberately flies a different value>"
    )


def _check_catalog_pairs(sc: dict[str, Any]) -> None:
    """Refuse flight parameters that contradict the hardware they describe (§19.3).

    Silent for a pair whose parameter is absent or whose unit kind is not
    installed, so a vehicle that predates a parameter — or does not carry that
    actuator — compiles unchanged. This is a cross-check between two things the
    config already says, never a new requirement on what it must say.
    """
    fsw = sc.get("fsw_parameters", {})
    divergences = sc.get("fsw_parameter_divergence", {})
    actuators = sc.get("actuators", [])
    for param, kind, truth, value_of, reduce_by, rtol, why in _CATALOG_PAIRS:
        units = [u for u in actuators if u.get("kind") == kind]
        if not units or param not in fsw:
            continue
        per_unit = {unit["name"]: value_of(unit) for unit in units}
        if any(v is None for v in per_unit.values()):
            # At least one installed unit's catalog entry says nothing about this
            # quantity, so the suite as a whole makes no claim the parameter can
            # contradict. (A declared zero is a claim, and is checked.)
            continue
        if reduce_by == _Reduce.UNIFORM and len(set(per_unit.values())) > 1:
            raise ConfigError(
                f"{param} is a single value, but this vehicle's {kind}s do not "
                f"share one {truth}: "
                f"{', '.join(f'{n} = {v}' for n, v in per_unit.items())}.\n"
                f"A mixed array needs this per-unit in flight, exactly as the "
                f"actuator axes already are; until it is, the flight scalar cannot "
                f"describe this suite."
            )
        expected = (min(per_unit.values()),)
        models = sorted({unit["model_id"] for unit in units})
        where = (
            f"the config/hardware entry for {', '.join(models)}, installed as "
            f"{', '.join(sorted(per_unit))}"
        )
        if _divergence_declared(divergences, param, expected, truth, rtol):
            continue
        actual = (float(fsw[param]),)
        if not _close(actual[0], expected[0], rtol):
            _refuse_mismatch(param, actual, expected, truth, where, why)


def _check_vehicle_pairs(sc: dict[str, Any]) -> None:
    """Refuse flight vectors that are not the vehicle's own properties (§19.3).

    An exact comparison (1e-9 relative, float-repr noise and nothing else): these
    are transcriptions of numbers written a few hundred lines up the same file,
    not derivations, so there is no round-off to allow for and a looser tolerance
    would only hide a real edit.
    """
    fsw = sc.get("fsw_parameters", {})
    divergences = sc.get("fsw_parameter_divergence", {})
    for param, field, accessor, truth, why in _VEHICLE_PAIRS:
        if param not in fsw or field not in sc:
            continue
        expected = accessor(sc[field])
        if _divergence_declared(divergences, param, expected, truth, 1.0e-9):
            continue
        actual = tuple(float(v) for v in fsw[param])
        if len(actual) != len(expected) or not all(
            _close(a, e, 1.0e-9) for a, e in zip(actual, expected)
        ):
            _refuse_mismatch(
                param,
                actual,
                expected,
                truth,
                "the vehicle config's own properties",
                why,
            )


def _check_declared_divergences(sc: dict[str, Any]) -> None:
    """Refuse an exemption for a parameter nothing cross-checks."""
    for param in sc.get("fsw_parameter_divergence", {}):
        if param not in _CROSS_CHECKED_PARAMS:
            raise ConfigError(
                f"fsw_parameter_divergence declares an exemption for '{param}', "
                f"which no flight/sim cross-check covers.\n"
                f"Either the name is a typo, or the check it was written against no "
                f"longer exists — both read as protection that is not there. "
                f"Checked parameters: {', '.join(sorted(_CROSS_CHECKED_PARAMS))}."
            )


def _check_control_parameters(body: dict[str, Any]) -> None:
    """Refuse control tuning that contradicts the installed actuator suite (§8.5)."""
    sc = body["spacecraft"]
    fsw = sc.get("fsw_parameters", {})
    actuators = sc.get("actuators", [])
    wheels = [u for u in actuators if u.get("kind") == "reaction_wheel"]
    rods = [u for u in actuators if u.get("kind") == "magnetorquer"]

    _check_per_unit_boresights(
        fsw,
        _WHEEL_AXES_PARAM,
        wheels,
        "reaction wheel",
        _WHY_WHEEL_AXES,
        expected_fn=_axis_key("spin_axis"),
    )
    _check_per_unit_boresights(
        fsw,
        _MTQ_AXES_PARAM,
        rods,
        "magnetorquer",
        _WHY_MTQ_AXES,
        expected_fn=_axis_key("dipole_axis"),
    )

    # The counts are what bound every loop in the flight allocation, so a count
    # that disagrees with the suite is not a tuning error to find in orbit: it
    # either drops an installed wheel out of the allocation or reads an axis slot
    # nothing filled.
    for param, units, label in (
        ("flight.attitudeController.WheelCount", wheels, "reaction wheel"),
        ("flight.attitudeController.MtqCount", rods, "magnetorquer"),
    ):
        declared = fsw.get(param)
        if declared is not None and int(declared) != len(units):
            raise ConfigError(
                f"{param} = {declared} against {len(units)} installed {label}(s) "
                f"({', '.join(u['name'] for u in units) or 'none'}).\n"
                f"The count bounds the flight control loops over this suite, so a "
                f"mismatch silently drops a unit or reads an axis nothing filled."
            )

    # The wheel-speed bias (§8.5, REQ-ACTL-012) is a per-wheel momentum the servo
    # holds on top of the pointing load. It must leave room under one wheel's
    # capacity, and — the rule the SITL bias row measured the thin end of — it
    # should exceed the worst single-wheel share of the desaturation threshold
    # (0.75 x DesatEnter along a spin axis on the pyramid), or a wheel reaches
    # zero before desaturation engages, which is the operating point the bias
    # exists to avoid. The first is refused; the second is a declared divergence.
    bias = fsw.get("flight.attitudeController.WheelBiasNms")
    capacity = fsw.get("flight.attitudeController.WheelCapacityNms")
    desat_enter = fsw.get("flight.attitudeController.MomentumDesatEnterNms")
    if bias is not None and capacity is not None:
        magnitudes = [abs(float(b)) for b in list(bias)[: len(wheels)]]
        if any(m >= float(capacity) for m in magnitudes):
            raise ConfigError(
                f"flight.attitudeController.WheelBiasNms holds a wheel at or past its "
                f"capacity ({max(magnitudes):g} >= {float(capacity):g} N*m*s): a bias is a "
                f"trim, not the whole wheel."
            )
        engaged = [m for m in magnitudes if m > 0.0]
        if engaged and desat_enter is not None:
            floor = 0.75 * float(desat_enter)
            if min(engaged) < floor and not _divergence_declared(
                sc.get("fsw_parameter_divergence", {}),
                "flight.attitudeController.WheelBiasNms",
                (floor,),
                "0.75 * MomentumDesatEnterNms",
                1.0e-6,
            ):
                raise ConfigError(
                    f"flight.attitudeController.WheelBiasNms = {min(engaged):g} N*m*s is "
                    f"under the worst single-wheel share of the desaturation threshold "
                    f"(0.75 x MomentumDesatEnterNms = {floor:g} N*m*s): a loaded wheel "
                    f"reaches zero speed before desaturation engages, which is what the "
                    f"bias exists to prevent. Raise the bias, or declare the divergence."
                )


_MAX_FIX_LATENCY_PARAM = "flight.orbitEstimator.MaxFixLatencyS"
_GNSS_CORR_H_PARAM = "flight.orbitEstimator.GnssCorrSigmaHM"
_GNSS_CORR_V_PARAM = "flight.orbitEstimator.GnssCorrSigmaVM"
_MAX_MEAS_AGE_PARAM = "flight.attitudeEstimator.MaxMeasAgeSec"
_BALLISTIC_PARAM = "flight.orbitEstimator.DragBallisticCoeffM2PerKg"
_WHY_BALLISTIC = (
    "The onboard filter's drag term integrates on this coefficient while the sim "
    "propagates the vehicle on drag_cd * drag_area_m2 / mass_kg, so a stale value "
    "is a systematic along-track acceleration error the filter absorbs into its "
    "process noise — invisible while GNSS is available, and the coast error the "
    "horizon was sized without."
)


_THRUSTER_AXES_PARAM = "flight.burnExecutor.ThrusterAxesBody"
_WHY_THRUSTER_AXES = (
    "The burn executor rotates the commanded thrust along this axis to tell the "
    "orbit filter what acceleration is acting; an axis that disagrees with the "
    "installed thruster tells the filter the vehicle is being pushed the wrong "
    "way, and the filter then rejects every fix under the burn."
)


def _check_burn_parameters(body: dict[str, Any]) -> None:
    """Refuse burn-executor tuning that contradicts the installed thrusters (§17).

    Per-unit thrust and Isp are the catalog's `thrust_n`/`isp_s` for that unit
    (a transcription, so an equality within float reading), the count is the
    installed count, and the axes are each unit's `thrust_axis`. Silent when the
    parameters are absent, so a vehicle without a burn executor compiles as
    before.
    """
    sc = body["spacecraft"]
    fsw = sc.get("fsw_parameters", {})
    thrusters = [u for u in sc.get("actuators", []) if u.get("kind") == "thruster"]
    count = fsw.get("flight.burnExecutor.ThrusterCount")
    if count is not None and int(count) != len(thrusters):
        raise ConfigError(
            f"flight.burnExecutor.ThrusterCount = {count} against {len(thrusters)} installed "
            f"thruster(s) ({', '.join(u['name'] for u in thrusters) or 'none'}).\n"
            f"The count bounds the executor's loops over the suite, so a mismatch fires "
            f"a thruster nothing installed or leaves one silent."
        )
    _check_per_unit_boresights(
        fsw,
        _THRUSTER_AXES_PARAM,
        thrusters,
        "thruster",
        _WHY_THRUSTER_AXES,
        expected_fn=_axis_key("thrust_axis"),
    )
    for param, key in (
        ("flight.burnExecutor.ThrusterThrustN", "thrust_n"),
        ("flight.burnExecutor.ThrusterIspS", "isp_s"),
    ):
        values = fsw.get(param)
        if values is None:
            continue
        for index, unit in enumerate(thrusters):
            catalog = unit.get("params", {}).get(key)
            if catalog is None or index >= len(values):
                continue
            if not _close(float(values[index]), float(catalog), 1.0e-6):
                raise ConfigError(
                    f"{param}[{index}] = {float(values[index]):g} against thruster "
                    f"'{unit['name']}' catalog {key} = {float(catalog):g}.\n"
                    f"The executor's commanded acceleration and mass depletion are "
                    f"computed from this number; a value the hardware does not deliver "
                    f"is an acceleration the orbit filter is told and never gets."
                )


def _check_orbit_parameters(body: dict[str, Any]) -> None:
    """Refuse an OD latency bound the installed receiver cannot meet (§8.3, §19.4).

    `MaxFixLatencyS` is the flight side of a flight/sim pair: the receiver's
    catalog entry carries `fix_latency_s`, the sim realises it as a delay line,
    and the filter refuses any fix older than the bound as a clock fault. A bound
    *below* the receiver's own latency therefore refuses every fix the receiver
    delivers, and the vehicle reports it as a stream of clock faults on a healthy
    receiver. An inequality rather than an equality — the bound is a ceiling with
    margin, not a transcription — so it takes no declared divergence.
    """
    sc = body["spacecraft"]
    fsw = sc.get("fsw_parameters", {})
    divergences = sc.get("fsw_parameter_divergence", {})
    # The ballistic coefficient is a *derivation* of three vehicle fields, written
    # to the digits a human writes: 1e-3 relative admits that and nothing else.
    if _BALLISTIC_PARAM in fsw and all(
        k in sc for k in ("drag_cd", "drag_area_m2", "mass_kg")
    ):
        expected = (
            float(sc["drag_cd"]) * float(sc["drag_area_m2"]) / float(sc["mass_kg"]),
        )
        truth = "drag_cd * drag_area_m2 / mass_kg"
        if not _divergence_declared(
            divergences, _BALLISTIC_PARAM, expected, truth, 1.0e-3
        ):
            actual = (float(fsw[_BALLISTIC_PARAM]),)
            if not _close(actual[0], expected[0], 1.0e-3):
                _refuse_mismatch(
                    _BALLISTIC_PARAM,
                    actual,
                    expected,
                    truth,
                    "the vehicle config's own drag_cd, drag_area_m2 and mass_kg",
                    _WHY_BALLISTIC,
                )
    if _MAX_FIX_LATENCY_PARAM not in fsw:
        return
    receivers = [u for u in sc.get("sensors", []) if u.get("kind") == "gnss"]
    latencies = {
        u["name"]: float(u.get("params", {})["fix_latency_s"])
        for u in receivers
        if u.get("params", {}).get("fix_latency_s") is not None
    }
    if not latencies:
        return
    bound = float(fsw[_MAX_FIX_LATENCY_PARAM])
    worst = max(latencies.values())
    if bound < worst:
        raise ConfigError(
            f"{_MAX_FIX_LATENCY_PARAM} = {bound} s is below the installed receiver's "
            f"own fix latency ({', '.join(f'{n} = {v} s' for n, v in latencies.items())}).\n"
            f"The orbit filter refuses any fix older than this bound as a clock "
            f"fault, so a bound under the receiver's catalogued latency refuses "
            f"every fix a healthy receiver delivers. Raise the bound to cover "
            f"fix_latency_s with margin (the reference vehicle flies 4x)."
        )


def _check_gnss_correlated_inflation(body: dict[str, Any]) -> None:
    """Refuse a filter blind to a receiver whose error the config says is correlated.

    The flight/sim pair (§19.3; Push 77): the receiver entry's
    ``correlated_position_fraction`` says how much of its datasheet variance is
    common-mode, and ``GnssCorrSigmaHM``/``VM`` is what the orbit filter inflates
    ``R`` by to survive it. Enabling the first and forgetting the second is the
    defect this exists to catch, and it is a *silent* one: nothing fails, the
    filter simply grows overconfident and starts rejecting honest fixes.

    A **bound, not an equality**, unlike the other catalog pairs. The tuned
    inflation is a multiple of the receiver's correlated sigma — measured at ~4x
    on the reference vehicle — because per-update inflation cannot reproduce time
    correlation and must be sized for the error's persistence across the fixes the
    filter averages, not for its size. That multiple depends on the fix cadence
    and on ``q_a``, so it is tuned against NEES per vehicle rather than derived
    here. What *is* checkable is that the inflation at least covers the
    magnitude: an inflation below the receiver's own correlated sigma cannot be
    a considered tuning, only an oversight.
    """
    sc = body["spacecraft"]
    fsw = sc.get("fsw_parameters", {})
    if _GNSS_CORR_H_PARAM not in fsw:
        return
    receivers = [u for u in sc.get("sensors", []) if u.get("kind") == "gnss"]
    worst_h = 0.0
    worst_v = 0.0
    named = {}
    for u in receivers:
        params = u.get("params", {})
        frac = params.get("correlated_position_fraction")
        rms_h = params.get("horizontal_position_rms_m")
        if frac is None or rms_h is None or float(frac) <= 0.0:
            continue
        # The same split the sim applies: per-axis sigma = 2D RMS / sqrt(2), of
        # which sqrt(fraction) is correlated. Vertical follows the model's
        # default 1.5x when the entry does not quote one.
        sigma_h = float(rms_h) / math.sqrt(2.0)
        rms_v = params.get("vertical_position_rms_m")
        sigma_v = float(rms_v) if rms_v is not None else sigma_h * 1.5
        corr_h = sigma_h * math.sqrt(float(frac))
        corr_v = sigma_v * math.sqrt(float(frac))
        named[u["name"]] = (corr_h, corr_v)
        worst_h = max(worst_h, corr_h)
        worst_v = max(worst_v, corr_v)
    if not named:
        return
    got_h = float(fsw[_GNSS_CORR_H_PARAM])
    got_v = float(fsw.get(_GNSS_CORR_V_PARAM, 0.0))
    if got_h + 1e-12 < worst_h or got_v + 1e-12 < worst_v:
        detail = ", ".join(
            f"{n} = {h:.4g} m horizontal / {v:.4g} m vertical"
            for n, (h, v) in sorted(named.items())
        )
        raise ConfigError(
            f"{_GNSS_CORR_H_PARAM} = {got_h} m and {_GNSS_CORR_V_PARAM} = {got_v} m are "
            f"below the installed receiver's own correlated sigma ({detail}).\n"
            f"The receiver entry declares part of its error common-mode, which the "
            f"reported fix sigmas do not describe and the filter therefore cannot "
            f"see. An inflation under that sigma does not even cover the error's "
            f"magnitude, let alone its persistence — the reference vehicle flies "
            f"~4x it, tuned against campaign NEES. Set it, or set "
            f"correlated_position_fraction to 0 if the receiver really is white."
        )


def _check_star_tracker_latency(body: dict[str, Any]) -> None:
    """Refuse a staleness window the installed star trackers cannot meet (§8.2, §9.1).

    `MaxMeasAgeSec` is the flight side of a flight/sim pair (Push 72): a tracker's
    catalog entry carries `latency_s`, the sim realises it as a delay line (the
    solution leaves the unit that long after the frame it describes), and the
    estimator refuses any sample tagged older than the window as stale. A window
    at or below the tracker's own latency therefore refuses every solution a
    healthy tracker delivers. Strict inequality: equality is a sample refused on
    the boundary of every poll. The FSW compensates the latency it accepts
    (`Mekf::updateAttitude(..., latency_s)`, NASA/TP-2018-219822 §3.1).
    """
    sc = body["spacecraft"]
    fsw = sc.get("fsw_parameters", {})
    if _MAX_MEAS_AGE_PARAM not in fsw:
        return
    trackers = [u for u in sc.get("sensors", []) if u.get("kind") == "star_tracker"]
    latencies = {
        u["name"]: float(u.get("params", {})["latency_s"])
        for u in trackers
        if u.get("params", {}).get("latency_s") is not None
    }
    if not latencies:
        return
    window = float(fsw[_MAX_MEAS_AGE_PARAM])
    worst = max(latencies.values())
    if window <= worst:
        raise ConfigError(
            f"{_MAX_MEAS_AGE_PARAM} = {window} s does not exceed the installed star "
            f"tracker's own solution latency "
            f"({', '.join(f'{n} = {v} s' for n, v in latencies.items())}).\n"
            f"The attitude estimator refuses any sample tagged older than this "
            f"window as stale, so a window at or under the tracker's catalogued "
            f"latency refuses every solution a healthy tracker delivers. Raise the "
            f"window to cover latency_s with margin (the reference vehicle flies 10x)."
        )


def _config_hash(resolved_body: dict[str, Any]) -> str:
    """SHA-256 over the canonical resolved config — everything that determines output."""
    canonical = json.dumps(resolved_body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve(
    config: Config,
    library: dict[str, HardwareModel],
    sources: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Produce the single resolved/validated config object (REQ-CFG-001).

    Model-IDs are resolved against @p library and inlined; the result is hashed
    for provenance (REQ-CFG-003). This object underlies all emitted artifacts —
    no emitter re-parses the raw YAML.
    """
    body = {
        "spacecraft": {
            **config.spacecraft.model_dump(exclude={"sensors", "actuators"}),
            # fsw_parameters comes through the model_dump above verbatim: it is a
            # flat name->value map validated against the topology dictionary at
            # emit time, not against the hardware library.
            "sensors": _resolve_units(config.spacecraft.sensors, library, "sensor"),
            "actuators": _resolve_units(
                config.spacecraft.actuators, library, "actuator"
            ),
        },
        "scenario": config.scenario.model_dump(),
    }
    # Cross-checks between the two halves of the resolved object, which only
    # become checkable once the hardware library has been inlined.
    _check_declared_divergences(body["spacecraft"])
    _check_albedo_parameters(body)
    _check_star_tracker_parameters(body)
    _check_control_parameters(body)
    _check_orbit_parameters(body)
    _check_gnss_correlated_inflation(body)
    _check_star_tracker_latency(body)
    _check_burn_parameters(body)
    _check_catalog_pairs(body["spacecraft"])
    _check_vehicle_pairs(body["spacecraft"])
    resolved = {
        "provenance": {
            "config_hash": _config_hash(body),
            "sources": sources or {},
        },
        **body,
    }
    return resolved


# --- Artifact emitters: all derive from the one resolved object ---------------


def _provenance(resolved: dict[str, Any]) -> dict[str, Any]:
    return resolved["provenance"]


def emit_fprime_params(resolved: dict[str, Any]) -> dict[str, Any]:
    """Flat name->value map destined for the F´ ``ParameterDb``.

    The human-readable twin of ``PrmDb.dat``: the same values, keyed by name
    instead of by generated ID, so a delivered parameter file can be inspected
    and diffed without decoding the binary. Vehicle mass properties and control
    gains are kept alongside as ``sc.*``/``gains.*`` entries — they are not F´
    parameters yet (no component declares them), and are stubs until one does.
    """
    sc = resolved["spacecraft"]
    params: dict[str, Any] = {
        "sc.mass_kg": sc["mass_kg"],
        "sc.com_m": sc["com_m"],
        "sc.inertia_kgm2": sc["inertia_kgm2"],
    }
    for mode, gains in sc.get("gains", {}).items():
        for key, value in gains.items():
            params[f"gains.{mode}.{key}"] = value
    return {
        "provenance": _provenance(resolved),
        "parameters": params,
        "fsw_parameters": dict(sc.get("fsw_parameters", {})),
    }


def emit_sim_setup(resolved: dict[str, Any]) -> dict[str, Any]:
    """Truth/plant setup: everything the sim executable needs to propagate a run.

    The Keplerian elements are converted to an ECI state here (§19.3: the compiler
    resolves, consumers do not re-derive) and emitted alongside the original
    elements, which are kept purely for traceability. Inertia is emitted as the
    six unique components of the symmetric body-frame tensor, matching the schema.
    """
    sc = resolved["spacecraft"]
    scn = resolved["scenario"]
    init = scn["initial_state"]
    orb = init["orbit"]
    position_m, velocity_m_s = keplerian_to_cartesian(
        sma_m=orb["sma_km"] * 1000.0,
        ecc=orb["ecc"],
        inc_deg=orb["inc_deg"],
        raan_deg=orb["raan_deg"],
        argp_deg=orb["argp_deg"],
        true_anomaly_deg=orb["true_anomaly_deg"],
    )
    return {
        "provenance": _provenance(resolved),
        "scenario_name": scn["name"],
        "epoch_utc": scn["epoch_utc"],
        "seed": scn["seed"],
        "spacecraft": sc,
        "initial_state": {
            "position_m": list(position_m),
            "velocity_m_s": list(velocity_m_s),
            "attitude_quaternion": init["attitude_quaternion"],
            "body_rate_rad_s": init["body_rate_rad_s"],
            "keplerian": orb,
        },
        "propagation": scn["propagation"],
        "environment": scn["environment"],
    }


def emit_analysis_inputs(resolved: dict[str, Any]) -> dict[str, Any]:
    """Analysis-tool inputs: epoch, ground network, vehicle summary (stub)."""
    sc = resolved["spacecraft"]
    scn = resolved["scenario"]
    return {
        "provenance": _provenance(resolved),
        "vehicle": {"name": sc["name"], "mass_kg": sc["mass_kg"]},
        "epoch_utc": scn["epoch_utc"],
        "ground_stations": scn["ground_stations"],
        "mc_dispersions": scn["mc_dispersions"],
    }


_ARTIFACTS = {
    "fprime_params.json": emit_fprime_params,
    "sim_setup.json": emit_sim_setup,
    "analysis_inputs.json": emit_analysis_inputs,
}


#: Name of the emitted ``Svc::PrmDb`` file. Fixed: it is what the topology's
#: ``prmDb.configure(...)`` call names, and what the deployment's ``-P`` option
#: points at (design doc §19.3).
PRMDB_FILENAME = "PrmDb.dat"


def emit_prmdb(resolved: dict[str, Any], dictionary_path: Path) -> bytes:
    """Encode the resolved FSW tuning into a ``Svc::PrmDb`` file image (§19.3).

    IDs and types come from the FPP-generated topology dictionary at
    @p dictionary_path, never from this config — see ``configc.prmdb``.
    """
    dictionary = load_dictionary(dictionary_path)
    values = resolved["spacecraft"].get("fsw_parameters", {})
    return build_param_file(values, dictionary)


def compile_config(
    config_path: Path,
    hardware_dir: Path,
    out_dir: Path,
    dictionary_path: Path | None = None,
) -> dict[str, Any]:
    """Full pipeline: load -> validate -> resolve -> emit artifacts to @p out_dir.

    Emits the three JSON artifacts always, plus the binary ``PrmDb.dat`` when
    @p dictionary_path names an FPP topology dictionary — that file needs the
    generated parameter IDs, so it can only be produced against a built flight
    deployment.

    Returns the resolved config object. Raises ConfigError on any semantic
    failure (unknown model-ID, malformed library/config, a tuning value that
    does not match the flight build's parameter set).
    """
    library = load_hardware_library(hardware_dir)
    config = load_config(config_path)
    sources = {
        "config": f"{config_path.name}:{_file_hash(config_path)}",
        "hardware_dir": f"{hardware_dir.name}:{_hardware_hash(hardware_dir)}",
    }
    if dictionary_path is not None:
        sources["dictionary"] = (
            f"{dictionary_path.name}:{_file_hash(dictionary_path)}"
            if dictionary_path.is_file()
            else f"{dictionary_path.name}:missing"
        )
    resolved = resolve(config, library, sources=sources)

    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, emit in _ARTIFACTS.items():
        (out_dir / filename).write_text(
            json.dumps(emit(resolved), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if dictionary_path is not None:
        try:
            image = emit_prmdb(resolved, dictionary_path)
        except PrmDbError as exc:
            raise ConfigError(
                f"{config_path}: parameter file not emitted\n{exc}"
            ) from exc
        (out_dir / PRMDB_FILENAME).write_bytes(image)
    return resolved


def _file_hash(path: Path) -> str:
    # Truncated to 16 hex chars: this is a human-readable source tag in the
    # provenance block, not the integrity hash. The authoritative fingerprint is
    # the full-length `config_hash`, which already covers every resolved value.
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _hardware_hash(hardware_dir: Path) -> str:
    """Content hash over the hardware library, so provenance identifies which
    library revision produced an artifact (not just the directory name)."""
    digest = hashlib.sha256()
    for path in sorted(hardware_dir.rglob("*.yaml")):
        # Relative path (not just the name) so the hash reflects the subdirectory
        # layout and two same-named files in different kind folders never collide.
        digest.update(path.relative_to(hardware_dir).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]
