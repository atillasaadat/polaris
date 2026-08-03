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

    boresights = fsw.get(_ALBEDO_BORESIGHTS_PARAM)
    if boresights is None:
        return
    for index, sun_unit in enumerate(sun_sensors):
        slot = boresights[3 * index : 3 * index + 3]
        if len(slot) < 3:
            raise ConfigError(
                f"{_ALBEDO_BORESIGHTS_PARAM} has no slot for sun sensor "
                f"{index} ('{sun_unit['name']}'): the parameter carries "
                f"{len(boresights) // 3} slots against {len(sun_sensors)} installed units."
            )
        written = tuple(float(v) for v in slot)
        # The zero vector is the configured "no correction for this unit" and is
        # always allowed — a mounting nobody has characterised is a legitimate
        # state, and the estimator then takes the uncorrected sun sigma.
        if written == (0.0, 0.0, 0.0):
            continue
        expected = _boresight_from_mounting(sun_unit)
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
                f"{_ALBEDO_BORESIGHTS_PARAM} slot {index} = {written} points "
                f"{math.degrees(angle):.4f} deg away from the mounting of sun sensor "
                f"'{sun_unit['name']}', whose boresight (unit +Z through its mounting) "
                f"is {expected}. The tolerance is 1e-6 rad on the angle between them.\n"
                f"The flight albedo correction places the Earth in *this* unit's field, "
                f"so a wrong boresight scales the correction rather than failing it. "
                f"Fix the parameter or the mounting_quaternion_wxyz, whichever is stale; "
                f"write the zero vector to skip the correction for this unit."
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
    _check_albedo_parameters(body)
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
