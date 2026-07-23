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
    # The GNSS jamming KML path and fault controls are carried through for the sim.
    setup = json.loads((tmp_path / "sim_setup.json").read_text())
    env = setup["environment"]
    assert env["gnss_jamming_kml"] == "config/scenario/jamming/eastern_europe.kml"
    assert env["gnss_jamming_enabled"] is True
    assert env["gnss_noise_enabled"] is True
    events = env["gnss_fault_events"]
    assert [e["type"] for e in events] == ["outage", "spoof"]
    assert events[1]["spoof_offset_ecef_m"] == [500.0, 0.0, 0.0]


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


_STAR_TRACKER_KEYS = {
    # Spatial errors, low and high frequency, XY (cross) / Z (about boresight).
    "lf_spatial_xy_arcsec_3sigma",
    "lf_spatial_z_arcsec_3sigma",
    "hf_spatial_xy_arcsec_3sigma",
    "hf_spatial_z_arcsec_3sigma",
    "lf_spatial_correlation_s",
    "hf_spatial_correlation_s",
    # White noise.
    "temporal_noise_xy_arcsec_3sigma",
    "temporal_noise_z_arcsec_3sigma",
    # Fixed and thermal terms.
    "bias_deg",
    "thermo_elastic_arcsec_per_c",
    # Separate acquisition and tracking envelopes, plus time to first fix.
    "acquisition_rate_deg_s",
    "tracking_rate_deg_s",
    "acquisition_accel_deg_s2",
    "tracking_accel_deg_s2",
    "lost_in_space_s",
    # Interfaces and geometry.
    "update_rate_hz",
    "fov_deg",
    "sun_exclusion_deg",
    "earth_exclusion_deg",
    "moon_exclusion_deg",
}


def test_star_tracker_catalog_entries_carry_the_full_spec():
    # Every key the C++ StarTrackerSpec reads (sim/sensors/star_tracker.cpp) must
    # be present, or the model silently defaults that mechanism to zero — which
    # for a star tracker means an instrument better than any that exists.
    library = load_hardware_library(_HARDWARE)
    for model_id in ("ST-16", "ST-GENERIC", "AURIGA"):
        entry = library[model_id]
        assert entry.kind == "star_tracker"
        missing = _STAR_TRACKER_KEYS - set(entry.params)
        assert not missing, f"{model_id} missing {sorted(missing)}"


def test_auriga_entry_matches_the_sodern_datasheet():
    # Sodern AURIGA brochure p.5, end-of-life worst case. These are the numbers a
    # pointing budget closes against, so a silent edit here changes a mission
    # analysis; pinning them makes that edit fail loudly.
    auriga = load_hardware_library(_HARDWARE)["AURIGA"]
    p = auriga.params
    assert p["lf_spatial_xy_arcsec_3sigma"] == 9.0
    assert p["lf_spatial_z_arcsec_3sigma"] == 51.0
    assert p["hf_spatial_xy_arcsec_3sigma"] == 6.6
    assert p["hf_spatial_z_arcsec_3sigma"] == 38.0
    assert p["temporal_noise_xy_arcsec_3sigma"] == 11.0
    assert p["temporal_noise_z_arcsec_3sigma"] == 70.0
    assert p["bias_deg"] == 0.017
    assert p["thermo_elastic_arcsec_per_c"] == 1.5
    assert p["lost_in_space_s"] == 3.8
    assert p["sun_exclusion_deg"] == 35.0
    assert p["earth_exclusion_deg"] == 22.0
    # "Full Moon in the field of view: no performance degradation" — a zero here
    # is a datasheet claim about the baffle, not a missing value.
    assert p["moon_exclusion_deg"] == 0.0

    # Acquisition is strictly the tighter envelope in both rate and acceleration.
    # A catalog edit that inverted these would make a slew look recoverable when
    # it is not, which is the failure mode this whole split exists to prevent.
    assert p["acquisition_rate_deg_s"] < p["tracking_rate_deg_s"]
    assert p["acquisition_accel_deg_s2"] < p["tracking_accel_deg_s2"]


@pytest.mark.parametrize("model_id", ["ST-16", "ST-GENERIC", "AURIGA"])
def test_star_tracker_about_boresight_error_is_the_weak_axis(model_id):
    # Roll is always worse than cross-boresight, for every error mechanism. An
    # entry that lost the asymmetry would quietly make the tracker better than
    # any real unit and leave an estimator overconfident in roll.
    p = load_hardware_library(_HARDWARE)[model_id].params
    for xy, z in (
        ("lf_spatial_xy_arcsec_3sigma", "lf_spatial_z_arcsec_3sigma"),
        ("hf_spatial_xy_arcsec_3sigma", "hf_spatial_z_arcsec_3sigma"),
        ("temporal_noise_xy_arcsec_3sigma", "temporal_noise_z_arcsec_3sigma"),
    ):
        assert p[z] > p[xy], f"{model_id}: {z} must exceed {xy}"


def test_sun_sensor_catalog_entries_carry_the_full_spec():
    # Two output contracts, so two key sets. An analogue part needs the diode
    # geometry and readout; a digital part needs its accuracy-vs-angle figures.
    # A missing key defaults that mechanism to zero, which for a sun sensor means
    # an instrument with no field limit or no signal at all.
    library = load_hardware_library(_HARDWARE)

    analogue = {
        "diode_count",
        "half_fov_deg",
        "full_scale_counts",
        "albedo_coefficient",
    }
    for model_id in ("CSS-GENERIC", "FSS-GENERIC"):
        params = library[model_id].params
        assert library[model_id].kind == "sun_sensor"
        missing = analogue - set(params)
        assert not missing, f"{model_id} missing {sorted(missing)}"

    digital = {
        "half_fov_deg",
        "accuracy_inner_half_angle_deg",
        "accuracy_inner_deg_3sigma",
        "accuracy_outer_deg_3sigma",
        "sample_period_ms",
    }
    fss = library["GS-NANOSENSE-FSS"]
    assert fss.kind == "sun_sensor"
    assert digital <= set(fss.params), sorted(digital - set(fss.params))


def test_gnss_catalog_entries_carry_the_full_spec():
    # A GNSS entry with no horizontal position accuracy would report truth-perfect
    # fixes — the C++ builder rejects it, but the catalog should carry the figure
    # in the first place. Every key the C++ GnssSpec reads (sim/sensors/gnss.cpp).
    library = load_hardware_library(_HARDWARE)
    keys = {
        "horizontal_position_rms_m",
        "velocity_accuracy_m_s_rms",
        "time_accuracy_ns_rms",
        "max_rate_hz",
    }
    for model_id in ("NOVATEL-OEM7600", "GNSS-GENERIC"):
        entry = library[model_id]
        assert entry.kind == "gnss"
        assert keys <= set(
            entry.params
        ), f"{model_id} missing {sorted(keys - set(entry.params))}"


def test_oem7600_entry_matches_the_novatel_datasheet():
    # NovAtel OEM7600 Product Sheet, single-point L1/L2. These are the numbers an
    # OD budget closes against; pinning them makes a silent edit fail loudly.
    p = load_hardware_library(_HARDWARE)["NOVATEL-OEM7600"].params
    assert p["horizontal_position_rms_m"] == 1.2  # "Single point L1/L2 1.2 m"
    assert p["velocity_accuracy_m_s_rms"] == 0.03  # "Velocity accuracy < 0.03 m/s RMS"
    assert p["time_accuracy_ns_rms"] == 5.0  # "Time accuracy < 5 ns RMS"
    assert p["max_rate_hz"] == 100.0  # "Position up to 100 Hz"
    assert p["cold_start_s"] == 34.0  # "Cold start < 34 s (typ)"
    assert p["reacquisition_s"] == 0.5  # "Signal reacquisition L1 < 0.5 s (typ)"


def test_gomspace_nanosense_fss_matches_the_datasheet():
    # GomSpace NanoSense FSS datasheet DS 1018157 rev 3.1, section 8. Pinning
    # these makes a silent edit fail loudly, because they are what a coarse
    # pointing budget is closed against.
    p = load_hardware_library(_HARDWARE)["GS-NANOSENSE-FSS"].params
    assert p["half_fov_deg"] == 60.0  # "Field of view: half angle 60 deg"
    assert p["accuracy_inner_half_angle_deg"] == 45.0
    assert p["accuracy_inner_deg_3sigma"] == 0.5  # "FOV < 45 deg, no albedo"
    assert p["accuracy_outer_deg_3sigma"] == 2.0  # "FOV < 60 deg, no albedo"
    assert p["sample_period_ms"] == 10.0  # "Sample period: max 10 ms"

    # Accuracy must get *worse* toward the field edge. Inverting these would make
    # the sensor best where it is physically weakest, and a coarse estimator
    # tuned on that would be confidently wrong exactly at wide sun angles.
    assert p["accuracy_outer_deg_3sigma"] > p["accuracy_inner_deg_3sigma"]
    assert p["accuracy_inner_half_angle_deg"] < p["half_fov_deg"]

    # The datasheet warns uncorrected albedo can exceed 10 deg — an order of
    # magnitude above the clean-sky figure, and the reason it is modelled at all.
    assert p["albedo_error_deg"] > 10.0


def test_magnetometer_catalog_entry_carries_the_full_spec():
    mag = load_hardware_library(_HARDWARE)["MAG-GENERIC"]
    assert mag.kind == "magnetometer"
    required = {
        "range_ut",
        "bias_ut",
        "noise_ut_rms",
        "resolution_nt",
        "scale_factor_pct",
        "misalignment_mrad",
    }
    assert required <= set(mag.params), sorted(required - set(mag.params))
    # The LEO field peaks near 50 uT, so a range below that would saturate in
    # normal operations and silently clip the measurement.
    assert mag.params["range_ut"] >= 50.0


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

    mtq800 = library["MTQ800"]
    assert mtq800.kind == "magnetorquer"
    assert mtq_required <= set(mtq800.params)
    assert mtq800.params["max_dipole_am2"] == 30.0  # boost limit (datasheet)
    assert mtq800.params["linearity"] == 0.02  # ±2% design accuracy


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
    # Optical-limb height for the sensor occlusion model (§6.1), a scenario knob
    # distinct from the drag atmosphere: what blocks a line of sight, not what
    # produces force.
    assert setup["environment"]["occultation_atmosphere_km"] == 100.0
    assert setup["spacecraft"]["residual_dipole_am2"] == [0.002, -0.001, 0.0015]
    assert setup["epoch_utc"] == "2026-01-01T00:00:00Z"


@pytest.mark.verifies("REQ-CFG-001")
def test_sim_setup_carries_the_resolved_hardware_suites(tmp_path):
    # The sim builds every sensor/actuator model from these params alone — there
    # is no in-code hardware catalog to fall back on (design doc §19.4). So the
    # artifact must carry each unit's identity, kind, and full param map, plus the
    # master seed the run's random streams derive from.
    compile_config(_TEMPLATE, _HARDWARE, tmp_path)
    setup = json.loads((tmp_path / "sim_setup.json").read_text())
    assert setup["seed"] == 20260101

    sc = setup["spacecraft"]
    imu = next(u for u in sc["sensors"] if u["name"] == "imu_a")
    assert imu["kind"] == "imu"
    assert imu["params"]["gyro_arw_deg_sqrt_hr"] == 0.15  # inlined, not referenced

    kinds = [u["kind"] for u in sc["actuators"]]
    assert kinds.count("reaction_wheel") == 4
    assert kinds.count("magnetorquer") == 3
    wheel = sc["actuators"][0]
    assert wheel["params"]["max_momentum_nms"] > 0.0
    # Every emitted unit must be usable: an empty param map is rejected by the
    # C++ side rather than building an ideal, unlimited device.
    assert all(u["params"] for u in sc["sensors"] + sc["actuators"])


@pytest.mark.verifies("REQ-CFG-003")
def test_seed_is_part_of_the_provenance_hash():
    # A run is reproducible from {config, seed}; two runs differing only in seed
    # are different runs and must not share a config hash.
    library = load_hardware_library(_HARDWARE)
    base = _minimal_config_dict()
    h1 = resolve(Config.model_validate(base), library)["provenance"]["config_hash"]
    seeded = _minimal_config_dict()
    seeded["scenario"]["seed"] = 7
    h2 = resolve(Config.model_validate(seeded), library)["provenance"]["config_hash"]
    assert h1 != h2


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


def test_gnss_fault_event_validation():
    # A back-to-front window is a scenario authoring error, caught at the boundary.
    bad = _minimal_config_dict()
    bad["scenario"]["environment"] = {
        "gnss_fault_events": [
            {"unit": "gps_a", "type": "outage", "start_s": 200.0, "stop_s": 100.0}
        ]
    }
    with pytest.raises(ValidationError, match="must exceed start_s"):
        Config.model_validate(bad)

    # An unknown fault type fails rather than being silently ignored.
    bad["scenario"]["environment"]["gnss_fault_events"][0] = {
        "unit": "gps_a",
        "type": "meteor",
        "start_s": 0.0,
        "stop_s": 10.0,
    }
    with pytest.raises(ValidationError):
        Config.model_validate(bad)


@pytest.mark.verifies("REQ-CFG-001")
def test_load_config_wraps_validation_errors(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("spacecraft: {name: x}\nscenario: {name: s, epoch_utc: t}\n")
    with pytest.raises(ConfigError, match="invalid config"):
        load_config(bad)
