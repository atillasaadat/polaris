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

#include <Eigen/Core>
#include <string>
#include <vector>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"
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

/// Vehicle properties the truth plant needs. Sensor/actuator suites are in the
/// artifact too but are not consumed yet — there are no sensor models.
struct SpacecraftConfig {
  std::string name;
  double mass_kg{0.0};
  Eigen::Matrix3d inertia_kgm2{Eigen::Matrix3d::Identity()};
  double drag_area_m2{0.0};
  double drag_cd{2.2};
  double srp_area_m2{0.0};
  double srp_cr{1.3};
  /// Centre-of-pressure offset from the centre of mass, Body frame [m]. Drives
  /// the aerodynamic and SRP disturbance torques.
  math::Vec3<math::frames::Body> cp_offset_m{};
  /// Residual magnetic moment, Body frame [A·m^2].
  math::Vec3<math::frames::Body> residual_dipole_am2{};
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
  bool drag_enabled{false};
  bool srp_enabled{false};
  bool eclipse_enabled{true};
  AtmosphereModel atmosphere{AtmosphereModel::kExponential};
  MagneticModel magnetic_field{MagneticModel::kIgrf};
};

/// Propagation span and RK89 step control.
struct PropagationConfig {
  double duration_s{0.0};
  double output_step_s{0.0};
  double abs_tol{1.0e-12};
  double rel_tol{1.0e-12};
  double max_step_s{60.0};
};

/// Everything one scenario needs to run.
struct SimConfig {
  std::string scenario_name;
  /// SHA-256 of the resolved config, from the compiler. Carried into the
  /// trajectory header so an output file can be traced back to the exact input
  /// that produced it (REQ-CFG-003).
  std::string config_hash;
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
