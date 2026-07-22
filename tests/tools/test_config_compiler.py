"""Tests for the config compiler (design doc §19.3, REQ-CFG-001/002/003).

Exercises the real hardware library and LEO template config plus in-memory
edge cases: model-ID resolution, single-resolved-object provenance shared across
all three emitted artifacts, deterministic hashing, and strict validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from configc import (
    Config,
    ConfigError,
    compile_config,
    load_config,
    load_hardware_library,
    resolve,
)

_REPO = Path(__file__).resolve().parents[2]
_HARDWARE = _REPO / "config" / "hardware"
_TEMPLATE = _REPO / "config" / "spacecraft" / "leo_smallsat.yaml"


def _minimal_config_dict(model_id: str = "STIM300") -> dict:
    return {
        "spacecraft": {
            "name": "T",
            "mass_kg": 10.0,
            "com_m": [0.0, 0.0, 0.0],
            "inertia_kgm2": {"ixx": 0.1, "iyy": 0.1, "izz": 0.1},
            "sensors": [{"name": "imu_a", "model_id": model_id}],
        },
        "scenario": {
            "name": "s",
            "epoch_utc": "2026-01-01T00:00:00Z",
            "initial_state": {
                "orbit": {
                    "sma_km": 6878.137,
                    "ecc": 0.0,
                    "inc_deg": 97.4,
                    "raan_deg": 0.0,
                    "argp_deg": 0.0,
                    "true_anomaly_deg": 0.0,
                }
            },
            "propagation": {"duration_s": 60.0, "output_step_s": 10.0},
        },
    }


@pytest.mark.verifies("REQ-CFG-001")
def test_compiles_template_and_emits_three_artifacts(tmp_path):
    resolved = compile_config(_TEMPLATE, _HARDWARE, tmp_path)
    names = {"fprime_params.json", "sim_setup.json", "analysis_inputs.json"}
    assert {p.name for p in tmp_path.iterdir()} == names
    # F´ params carry the vehicle mass and a resolved per-mode gain.
    fparams = json.loads((tmp_path / "fprime_params.json").read_text())["parameters"]
    assert fparams["sc.mass_kg"] == 12.0
    assert fparams["gains.pointing.kp"] == 0.2
    assert resolved["provenance"]["config_hash"]


@pytest.mark.verifies("REQ-CFG-001")
def test_one_resolved_object_underlies_all_three_artifacts(tmp_path):
    # The single resolved object's hash must be identical across every artifact —
    # proof no emitter re-parsed the raw YAML independently.
    compile_config(_TEMPLATE, _HARDWARE, tmp_path)
    hashes = {
        json.loads((tmp_path / name).read_text())["provenance"]["config_hash"]
        for name in ("fprime_params.json", "sim_setup.json", "analysis_inputs.json")
    }
    assert len(hashes) == 1


@pytest.mark.verifies("REQ-CFG-002")
def test_resolves_hardware_model_ids_into_params():
    library = load_hardware_library(_HARDWARE)
    resolved = resolve(load_config(_TEMPLATE), library)
    imu = next(s for s in resolved["spacecraft"]["sensors"] if s["name"] == "imu_a")
    assert imu["model_id"] == "STIM300"
    assert imu["kind"] == "imu"
    # Library params are inlined, not just referenced.
    assert imu["params"]["gyro_arw_deg_sqrt_hr"] == 0.15


def test_stim377h_catalog_entry_carries_the_full_imu_spec():
    # The STIM377H catalog entry must expose every key the C++ ImuSpec::fromParams
    # reads (sim/sensors/imu.cpp), so selecting it actually configures the model
    # rather than silently falling back to zeros. A few datasheet values are pinned.
    library = load_hardware_library(_HARDWARE)
    stim = library["STIM377H"]
    assert stim.kind == "imu"
    required = {
        "gyro_range_deg_s",
        "gyro_arw_deg_sqrt_hr",
        "gyro_bias_instability_deg_hr",
        "gyro_bias_correlation_s",
        "gyro_bias_repeatability_deg_hr",
        "gyro_scale_factor_ppm",
        "gyro_misalignment_mrad",
        "gyro_resolution_deg_hr",
        "gyro_g_sensitivity_deg_hr_g",
        "accel_range_g",
        "accel_vrw_m_s_sqrt_hr",
        "accel_bias_instability_mg",
        "accel_bias_correlation_s",
        "accel_bias_repeatability_mg",
        "accel_scale_factor_ppm",
        "accel_misalignment_mrad",
        "accel_resolution_ug",
        "sample_rate_hz",
    }
    assert required <= set(stim.params), sorted(required - set(stim.params))
    assert stim.params["gyro_bias_instability_deg_hr"] == 0.3  # datasheet
    assert stim.params["accel_vrw_m_s_sqrt_hr"] == 0.07
    # The generic template exposes the same keys, so it is a complete starting point.
    assert required <= set(library["IMU-GENERIC"].params)


def test_actuator_catalog_entries_carry_the_full_spec():
    # The RW and MTQ catalog entries must expose the keys the C++ specs read
    # (sim/actuators/*.cpp), so selecting one configures the model rather than
    # defaulting silently to zeros.
    library = load_hardware_library(_HARDWARE)
    rw = library["RW-0.4"]
    assert rw.kind == "reaction_wheel"
    rw_required = {
        "max_torque_nm",
        "max_momentum_nms",
        "max_speed_rpm",
        "motor_kt_nm_a",
        "motor_resistance_ohm",
        "dry_friction_nm",
        "viscous_friction_nm_s",
        "aero_friction_nm_s2",
        "torque_quantization_nm",
        "static_imbalance_kg_m",
        "dynamic_imbalance_kg_m2",
        "idle_power_w",
    }
    assert rw_required <= set(rw.params), sorted(rw_required - set(rw.params))
    assert rw.params["max_momentum_nms"] == 0.4  # datasheet
    assert rw_required <= set(library["RW-X"].params)  # generic template is complete

    nss = library["NSS-TAURUS-30"]
    assert nss.kind == "magnetorquer"
    mtq_required = {"max_dipole_am2", "residual_dipole_am2", "linearity", "power_max_w"}
    assert mtq_required <= set(nss.params)
    assert nss.params["linearity"] == 0.05  # ±5% datasheet
    assert mtq_required <= set(library["MTQ-GENERIC"].params)


@pytest.mark.verifies("REQ-CFG-002")
def test_swapping_model_id_swaps_resolved_params():
    library = load_hardware_library(_HARDWARE)
    a = resolve(Config.model_validate(_minimal_config_dict("STIM300")), library)
    b = resolve(Config.model_validate(_minimal_config_dict("ST-16")), library)
    assert a["spacecraft"]["sensors"][0]["kind"] == "imu"
    assert b["spacecraft"]["sensors"][0]["kind"] == "star_tracker"


@pytest.mark.verifies("REQ-CFG-002")
def test_unknown_model_id_raises():
    library = load_hardware_library(_HARDWARE)
    cfg = Config.model_validate(_minimal_config_dict("DOES-NOT-EXIST"))
    with pytest.raises(ConfigError, match="unknown hardware model_id"):
        resolve(cfg, library)


def _write_hw(dir_: Path, model_id: str, arw: float = 0.15) -> None:
    (dir_ / f"{model_id}.yaml").write_text(
        f"model_id: {model_id}\nkind: imu\nparams: {{gyro_arw: {arw}}}\n"
    )


@pytest.mark.verifies("REQ-CFG-002")
def test_duplicate_model_id_raises(tmp_path):
    _write_hw(tmp_path, "DUP")
    (tmp_path / "other.yaml").write_text(
        "model_id: DUP\nkind: imu\nparams: {gyro_arw: 0.2}\n"
    )
    with pytest.raises(ConfigError, match="duplicate hardware model_id"):
        load_hardware_library(tmp_path)


@pytest.mark.verifies("REQ-CFG-002")
def test_malformed_hardware_entry_raises(tmp_path):
    (tmp_path / "bad.yaml").write_text("model_id: X\nkind: not_a_device\nparams: {}\n")
    with pytest.raises(ConfigError, match="invalid hardware model"):
        load_hardware_library(tmp_path)


@pytest.mark.verifies("REQ-CFG-002")
def test_missing_hardware_dir_raises_clearly(tmp_path):
    with pytest.raises(ConfigError, match="hardware library directory not found"):
        load_hardware_library(tmp_path / "does_not_exist")


@pytest.mark.verifies("REQ-CFG-003")
def test_hardware_param_change_changes_config_hash(tmp_path):
    # Provenance must track hardware-library content, not just the spacecraft dict.
    _write_hw(tmp_path, "STIM300", arw=0.15)
    cfg = Config.model_validate(_minimal_config_dict("STIM300"))
    h1 = resolve(cfg, load_hardware_library(tmp_path))["provenance"]["config_hash"]
    _write_hw(tmp_path, "STIM300", arw=0.99)  # same model-ID, different param
    h2 = resolve(cfg, load_hardware_library(tmp_path))["provenance"]["config_hash"]
    assert h1 != h2


@pytest.mark.verifies("REQ-CFG-003")
def test_provenance_hash_is_deterministic_and_value_sensitive():
    library = load_hardware_library(_HARDWARE)
    base = _minimal_config_dict()
    h1 = resolve(Config.model_validate(base), library)["provenance"]["config_hash"]
    h2 = resolve(Config.model_validate(base), library)["provenance"]["config_hash"]
    assert h1 == h2  # deterministic

    changed = _minimal_config_dict()
    changed["spacecraft"]["mass_kg"] = 11.0
    h3 = resolve(Config.model_validate(changed), library)["provenance"]["config_hash"]
    assert h3 != h1  # a changed value changes the hash


@pytest.mark.verifies("REQ-CFG-001")
def test_sim_setup_carries_the_resolved_cartesian_initial_state(tmp_path):
    # The compiler resolves elements -> ECI state so no consumer re-derives it.
    compile_config(_TEMPLATE, _HARDWARE, tmp_path)
    setup = json.loads((tmp_path / "sim_setup.json").read_text())
    init = setup["initial_state"]
    assert len(init["position_m"]) == 3 and len(init["velocity_m_s"]) == 3
    # 500 km circular: |r| is the semi-major axis, and the elements are kept.
    radius = sum(c * c for c in init["position_m"]) ** 0.5
    assert radius == pytest.approx(6_878_137.0, rel=1e-9)
    assert init["keplerian"]["inc_deg"] == 97.4018
    assert setup["propagation"]["duration_s"] == 5677.0
    assert setup["environment"]["atmosphere"] == "exponential"
    assert setup["spacecraft"]["residual_dipole_am2"] == [0.002, -0.001, 0.0015]
    assert setup["epoch_utc"] == "2026-01-01T00:00:00Z"


@pytest.mark.verifies("REQ-CFG-001")
def test_unnormalised_attitude_quaternion_is_rejected():
    bad = _minimal_config_dict()
    bad["scenario"]["initial_state"]["attitude_quaternion"] = [1.0, 0.5, 0.0, 0.0]
    with pytest.raises(ValidationError, match="normalised to within 1e-9"):
        Config.model_validate(bad)


@pytest.mark.verifies("REQ-CFG-001")
def test_unknown_config_key_is_rejected():
    # extra='forbid' — a typo'd field fails validation instead of being dropped.
    bad = _minimal_config_dict()
    bad["spacecraft"]["typo_field"] = 1.0
    with pytest.raises(ValidationError, match="typo_field"):
        Config.model_validate(bad)


@pytest.mark.verifies("REQ-CFG-001")
def test_load_config_wraps_validation_errors(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("spacecraft: {name: x}\nscenario: {name: s, epoch_utc: t}\n")
    with pytest.raises(ConfigError, match="invalid config"):
        load_config(bad)
