#ifndef POLARIS_SIM_SCENARIO_SIM_CONFIG_HPP
#define POLARIS_SIM_SCENARIO_SIM_CONFIG_HPP

/// @file
/// @brief Parsed `sim_setup.json` — the truth-sim half of the config pipeline
/// (design doc §19.3, REQ-CFG-001).
///
/// The compiled artifact, not the YAML. `tools/configc` is the single mechanism
/// that reads `config/spacecraft/*.yaml` + the hardware library, validates
/// against the pydantic schema, resolves model-IDs, and emits `sim_setup.json`;
/// the sim reads only that. §19.3 is explicit that "no consumer re-parses raw
/// YAML independently" — three representations (YAML, F´ `ParameterDb`, sim
/// setup) that must agree cannot be kept in agreement if each parses its own
/// source.
///
/// One consequence worth stating: the **Keplerian-to-Cartesian conversion
/// happens in the compiler**, not here. The artifact carries ECI position and
/// velocity, with the original elements alongside for traceability. So there is
/// no orbital-element code on the C++ side at all, and no second implementation
/// to drift from the first.
///
/// This struct is a plain data carrier: it holds what the file said, and makes
/// no attempt to build models. `sim_runner.hpp` turns it into a running sim.
///
/// Ground/sim-side: file I/O, heap, exceptions-free error returns.

#include <cstdint>
#include <Eigen/Core>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "sensors/occlusion.hpp"
#include "state/truth_state.hpp"
#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"

namespace polaris::sim::scenario {

/// Which atmosphere the drag model is driven by.
enum class AtmosphereModel {
  kExponential,  ///< data-free piecewise-exponential fit (Vallado Table 8-4)
  kNrlmsis,      ///< NRLMSIS 2.1; requires POLARIS_HAS_NRLMSIS and a parm file
};

/// Which geomagnetic field model to fly.
enum class MagneticModel {
  kNone,
  kIgrf,
};

/// One hardware unit installed on the vehicle, as the compiler resolved it: the
/// library entry's parameters already inlined (design doc §19.2/§19.3).
///
/// `params` is carried as the artifact's own datasheet-native key/value map and
/// handed unmodified to the model's `fromParams`, which owns the conversion to
/// SI. Nothing here interprets a key, so adding a parameter to a catalog entry
/// needs no change on this side — and there is no second place where a unit's
/// numbers could be written down (§19.4).
struct UnitConfig {
  std::string name;      ///< instance name on this vehicle, e.g. "imu_a"
  std::string model_id;  ///< hardware-library model ID it resolved from
  std::string kind;      ///< device class: "imu", "reaction_wheel", …
  std::map<std::string, double> params;
  /// Unit→body rotation. Identity when the config omitted a mounting.
  Eigen::Matrix3d mounting_dcm{Eigen::Matrix3d::Identity()};
  /// Reaction-wheel/CMG spin axis in the body frame, or zero when unset. When
  /// nonzero it defines the wheel's axis (the clean alternative to a full
  /// mounting DCM); the vehicle builder normalises it into the assembly's W.
  Eigen::Vector3d spin_axis{Eigen::Vector3d::Zero()};
  /// Thruster nominal thrust axis in the body frame, or zero when unset (the
  /// unit's +z through `mounting_dcm` then applies). Same idea as `spin_axis`:
  /// a thruster has one direction and no meaningful roll about it (§7, §17).
  Eigen::Vector3d thrust_axis{Eigen::Vector3d::Zero()};
  /// Unit origin in the body frame [m]; zero (the body origin) when the config
  /// omitted it. Consumed by the effects that depend on *where* a unit sits
  /// rather than which way it points — today the §7 MTQ/MAG near-field coupling,
  /// which goes as 1/r^3 between an energised rod and a magnetometer.
  Eigen::Vector3d mounting_position_m{Eigen::Vector3d::Zero()};
  /// Per-unit override of the scenario sensor-noise switch (§6.2). Unset =
  /// inherit the global `sensor_noise_enabled`; set forces this sensor's noise
  /// on/off regardless. Ignored for actuators (no stochastic noise).
  std::optional<bool> noise_enabled;
};

/// Vehicle properties the truth plant needs, including the installed hardware.
struct SpacecraftConfig {
  std::string name;
  double mass_kg{0.0};
  Eigen::Matrix3d inertia_kgm2{Eigen::Matrix3d::Identity()};
  double drag_area_m2{0.0};
  double drag_cd{2.2};
  double srp_area_m2{0.0};
  double srp_cr{1.3};
  /// Centre of mass in the Body (structural) frame [m]. The dynamics currently
  /// place the body origin at the CoM, so this is carried, not yet consumed — it
  /// becomes load-bearing when propellant depletion / CG shift arrives (§17,
  /// Phase 8). Parsed rather than silently dropped: a schema-required field the
  /// sim discards would violate the §19.4 no-silent-default rule.
  math::Vec3<math::frames::Body> com_m{};
  /// Aerodynamic centre-of-pressure offset from the centre of mass, Body frame
  /// [m] — the lever arm of the §5.3 aero disturbance torque.
  math::Vec3<math::frames::Body> cp_offset_aero_m{};
  /// Optical centre-of-pressure offset from the centre of mass, Body frame [m].
  /// Separate from the aerodynamic one: the optical CP is set by the illuminated
  /// area and the aerodynamic CP by the ram area, and they generally differ.
  math::Vec3<math::frames::Body> cp_offset_srp_m{};
  /// Residual magnetic moment, Body frame [A·m^2].
  math::Vec3<math::frames::Body> residual_dipole_am2{};
  /// Installed sensors and actuators, in config order (`vehicle.hpp` turns these
  /// into models). A kind with no truth model yet is carried here regardless —
  /// dropping it at parse time would hide it from the vehicle's own report.
  std::vector<UnitConfig> sensors;
  std::vector<UnitConfig> actuators;
};

/// A scheduled GNSS fault (§9.2/§23.1.1): what the scenario does to a receiver,
/// as opposed to what the catalog says it is. Active for `start_s <= t < stop_s`
/// (seconds since epoch) on the named unit.
struct GnssFaultEvent {
  enum class Type { kOutage, kSpoof, kClockJump };
  std::string unit;
  Type type{Type::kOutage};
  double start_s{0.0};
  double stop_s{0.0};
  Eigen::Vector3d spoof_offset_ecef_m{Eigen::Vector3d::Zero()};  ///< for kSpoof
  double clock_jump_s{0.0};                                      ///< for kClockJump
};

/// One scheduled open-loop burn (§17); see EnvironmentConfig::thrust_events.
struct ThrustEvent {
  std::string unit;
  double start_s{0.0};
  double stop_s{0.0};
  double throttle{1.0};
};

/// Which perturbations are switched on, and at what fidelity.
struct EnvironmentConfig {
  /// Negative: no gravity (free-drift baseline). 0: point mass. Positive: the
  /// EGM2008 spherical-harmonic field truncated to this degree and order.
  int gravity_degree{8};
  /// Max harmonic order m. Negative means "same as the degree" — the usual
  /// square field. Exposed separately because degree and order are genuinely
  /// independent knobs: a zonal-only field (order 0) is the standard J2-class
  /// comparison case, and the onboard model is deliberately run at a lower
  /// fidelity than truth (`sim/CLAUDE.md`).
  int gravity_order{-1};
  bool sun_third_body{false};
  bool moon_third_body{false};
  /// Planetary third-body perturbers by name ("jupiter", "venus", …), validated
  /// against the ephemeris fixture's body set at load. Sun/Moon dominate for
  /// Earth orbits; these are for completeness studies (Jupiter and Venus are
  /// the largest, ~1e-7 of the lunar term in LEO).
  std::vector<std::string> planet_third_bodies;
  bool drag_enabled{true};
  bool srp_enabled{true};
  bool eclipse_enabled{true};
  /// @name §5.3 disturbance torques — one switch each
  /// Separate from the force switches above because isolating one disturbance is
  /// a routine MC study: `aero_torque_enabled: false` keeps the drag *force* (and
  /// so the orbit) while removing its couple. Each is physically present, so the
  /// default is on; the lever arms and the dipole they need are no-default
  /// vehicle-config fields, and asking for a torque without its field is a config
  /// error rather than a silently-zero torque.
  /// @{
  bool gravity_gradient_torque_enabled{true};
  bool aero_torque_enabled{true};
  bool srp_torque_enabled{true};
  bool residual_dipole_torque_enabled{true};
  /// @}
  AtmosphereModel atmosphere{AtmosphereModel::kExponential};
  MagneticModel magnetic_field{MagneticModel::kIgrf};
  /// Optically obstructing atmosphere thickness above the surface [m], for the
  /// optical-sensor occlusion model (§6.1). Separate from the drag atmosphere:
  /// this one is about what blocks a line of sight, not what produces force.
  double occultation_atmosphere_m{sensors::kDefaultAtmosphereHeight};
  /// Path to a KML of GNSS-jamming regions (§9.2), or empty for none. Carried
  /// verbatim from the config; the sim loads it into `sensors::JammingRegions`
  /// and binds it to each GNSS receiver.
  std::string gnss_jamming_kml;
  /// Master switch for geographic jamming — false keeps the KML referenced but
  /// binds no regions, so a run can toggle jamming without editing the path.
  bool gnss_jamming_enabled{true};
  /// Master switch for **all** sensor noise (IMU, star tracker, sun sensor,
  /// magnetometer, GNSS) — false flies ideal sensors (measurement = truth) for a
  /// noise-free baseline or with/without-noise comparison. Actuators have no
  /// stochastic noise, so there is no equivalent for them.
  bool sensor_noise_enabled{true};
  /// Master switch for GNSS measurement noise — false flies a truth-perfect
  /// receiver (position/velocity/clock error zeroed) for bring-up/debug. Applied
  /// on top of `sensor_noise_enabled` (both must be true for GNSS noise).
  bool gnss_noise_enabled{true};
  /// Scheduled GNSS faults injected during the run (§9.2/§23.1.1).
  std::vector<GnssFaultEvent> gnss_fault_events;
  /// Scheduled truth burns for the **open-loop** path (no FSW): a thruster is
  /// held at `throttle` over [start_s, stop_s). In SITL, burns are commanded
  /// by the flight burn executor and this schedule is ignored — the sim never
  /// fires a thruster the FSW did not ask for on a closed loop.
  std::vector<ThrustEvent> thrust_events;
};

/// Propagation span and RK89 step control.
struct PropagationConfig {
  double duration_s{0.0};
  double output_step_s{0.0};
  double abs_tol{1.0e-12};
  double rel_tol{1.0e-12};
  double max_step_s{60.0};
  /// FSW macro-step rate [Hz] for the §2.4 closed loop: sensor buffers publish,
  /// the FSW fires, and its commands apply on the next step at this cadence.
  double fsw_rate_hz{10.0};
};

/// Everything one scenario needs to run.
struct SimConfig {
  std::string scenario_name;
  /// SHA-256 of the resolved config, from the compiler. Carried into the
  /// trajectory header so an output file can be traced back to the exact input
  /// that produced it (REQ-CFG-003).
  std::string config_hash;
  /// Master RNG seed for the run (§3.5). Every stochastic source derives its own
  /// stream from this, so a run is bit-reproducible from `{config, seed}`.
  std::uint64_t seed{0};
  SpacecraftConfig spacecraft;
  EnvironmentConfig environment;
  PropagationConfig propagation;
  /// Initial truth state, epoch included.
  state::TruthState initial_state;
};

/// Parse a `sim_setup.json` emitted by `tools/configc`.
///
/// @param path  Path to the artifact.
/// @param leap  Leap-second table, needed to turn the artifact's UTC epoch into
///              the TAI master clock (design doc §3.2).
/// @param out   Filled on success; partially written on failure.
/// @param error If non-null, receives a human-readable reason on failure.
/// @return false if the file is missing, is not valid JSON, lacks a required
///         field, or carries a structurally impossible value (non-positive mass,
///         a non-normalised attitude quaternion, a non-positive duration). A
///         config error is a hard failure: a sim that silently defaulted a
///         missing mass would produce a plausible trajectory that answers a
///         different question than the one asked.
bool loadSimConfig(const std::string& path, const time::LeapSecondTable& leap, SimConfig& out,
                   std::string* error = nullptr);

}  // namespace polaris::sim::scenario

#endif  // POLARIS_SIM_SCENARIO_SIM_CONFIG_HPP
