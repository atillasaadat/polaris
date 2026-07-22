"""The Polaris config compiler (design doc §19.3, REQ-CFG-001/003).

One mechanism turns the single source-of-truth config (spacecraft + scenario +
hardware-library references) into every downstream artifact, so the three
representations that must agree — YAML config, F´ ``ParameterDb`` params, and sim
setup — cannot drift. The pipeline is:

    load hardware library  ->  load + validate config  ->  resolve model-IDs
      ->  one resolved/validated object  ->  provenance hash  ->  emit 3 artifacts

Every consumer reads the derived artifacts, never the raw YAML (REQ-CFG-001), and
each artifact records the source config hash so any FSW param / sim run / fixture
is traceable to the exact config that produced it (REQ-CFG-003).

The emitters here are **stubs** (design doc Phase-0 checklist): they write the
resolved values as JSON rather than a real F´ ``ParameterDb`` binary or sim
harness input. The pipeline — validation, model-ID resolution, single resolved
object, provenance — is real; only the artifact *encoding* is provisional.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .orbit import keplerian_to_cartesian
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
                "mounting_dcm_row_major": unit.mounting_dcm_row_major,
            }
        )
    return resolved


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
            "sensors": _resolve_units(config.spacecraft.sensors, library, "sensor"),
            "actuators": _resolve_units(
                config.spacecraft.actuators, library, "actuator"
            ),
        },
        "scenario": config.scenario.model_dump(),
    }
    resolved = {
        "provenance": {
            "config_hash": _config_hash(body),
            "sources": sources or {},
        },
        **body,
    }
    return resolved


# --- Artifact emitters (stubs): all derive from the one resolved object -------


def _provenance(resolved: dict[str, Any]) -> dict[str, Any]:
    return resolved["provenance"]


def emit_fprime_params(resolved: dict[str, Any]) -> dict[str, Any]:
    """Flat name->value map destined for the F´ ``ParameterDb`` (stub)."""
    sc = resolved["spacecraft"]
    params: dict[str, Any] = {
        "sc.mass_kg": sc["mass_kg"],
        "sc.com_m": sc["com_m"],
        "sc.inertia_kgm2": sc["inertia_kgm2"],
    }
    for mode, gains in sc.get("gains", {}).items():
        for key, value in gains.items():
            params[f"gains.{mode}.{key}"] = value
    return {"provenance": _provenance(resolved), "parameters": params}


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


def compile_config(
    config_path: Path, hardware_dir: Path, out_dir: Path
) -> dict[str, Any]:
    """Full pipeline: load -> validate -> resolve -> emit 3 artifacts to @p out_dir.

    Returns the resolved config object. Raises ConfigError on any semantic
    failure (unknown model-ID, malformed library/config).
    """
    library = load_hardware_library(hardware_dir)
    config = load_config(config_path)
    sources = {
        "config": f"{config_path.name}:{_file_hash(config_path)}",
        "hardware_dir": f"{hardware_dir.name}:{_hardware_hash(hardware_dir)}",
    }
    resolved = resolve(config, library, sources=sources)

    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, emit in _ARTIFACTS.items():
        (out_dir / filename).write_text(
            json.dumps(emit(resolved), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
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
