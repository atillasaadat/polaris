#include "scenario/sim_config.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <Eigen/Eigenvalues>
#include <fstream>
#include <nlohmann/json.hpp>

#include "math/quaternion.hpp"
#include "time/utc.hpp"
#include "world/ephemeris_file.hpp"

namespace polaris::sim::scenario {
namespace {

using nlohmann::json;

bool fail(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
  return false;
}

/// Fetch a required key, reporting the path on absence.
const json* require(const json& node, const char* key, const std::string& path,
                    std::string* error) {
  const auto it = node.find(key);
  if (it == node.end()) {
    fail(error, "missing required field '" + path + "." + key + "'");
    return nullptr;
  }
  return &*it;
}

/// Read a required number.
bool readNumber(const json& node, const char* key, const std::string& path, double& out,
                std::string* error) {
  const json* value = require(node, key, path, error);
  if (value == nullptr) {
    return false;
  }
  if (!value->is_number()) {
    return fail(error, "field '" + path + "." + key + "' is not a number");
  }
  out = value->get<double>();
  return true;
}

/// Read a required 3-element array into an Eigen vector.
bool readVec3(const json& node, const char* key, const std::string& path, Eigen::Vector3d& out,
              std::string* error) {
  const json* value = require(node, key, path, error);
  if (value == nullptr) {
    return false;
  }
  if (!value->is_array() || value->size() != 3) {
    return fail(error, "field '" + path + "." + key + "' must be a 3-element array");
  }
  for (std::size_t i = 0; i < 3; ++i) {
    if (!(*value)[i].is_number()) {
      return fail(error, "field '" + path + "." + key + "' has a non-numeric entry");
    }
    out[static_cast<Eigen::Index>(i)] = (*value)[i].get<double>();
  }
  return true;
}

/// Parse an ISO-8601 UTC timestamp, `YYYY-MM-DDThh:mm:ss[.fff]Z`.
///
/// Deliberately strict and hand-rolled rather than `strptime`: the artifact is
/// machine-generated with a known shape, and the locale- and timezone-dependent
/// behaviour of the C library parsers is exactly what should not sit under a
/// master clock. Anything not matching is rejected rather than partially read.
bool parseUtc(const std::string& text, time::UtcDateTime& out, std::string* error) {
  int year = 0;
  unsigned month = 0;
  unsigned day = 0;
  unsigned hour = 0;
  unsigned minute = 0;
  double second = 0.0;
  int consumed = 0;
  // %n reports how much of the string was matched, so trailing garbage after an
  // otherwise-valid timestamp is rejected instead of ignored. The trailing 'Z'
  // is mandatory: an artifact without it is not asserting UTC, and silently
  // treating a local-time string as UTC would shift the whole run.
  const int fields = std::sscanf(text.c_str(), "%4d-%2u-%2uT%2u:%2u:%lfZ%n", &year, &month, &day,
                                 &hour, &minute, &second, &consumed);
  if (fields != 6 || consumed <= 0 || static_cast<std::size_t>(consumed) != text.size()) {
    return fail(error, "epoch_utc '" + text +
                           "' is not an ISO-8601 UTC timestamp (YYYY-MM-DDThh:mm:ss[.fff]Z)");
  }
  // Width-limited conversions above cap each field, so nothing can overflow the
  // way an unbounded %d would; isValidUtc then rejects impossible dates.
  if (!(second >= 0.0) || second >= 61.0) {
    return fail(error, "epoch_utc '" + text + "' has an out-of-range seconds field");
  }

  out.year = year;
  out.month = month;
  out.day = day;
  out.hour = hour;
  out.minute = minute;
  const double whole = std::floor(second);
  out.second = static_cast<unsigned>(whole);
  out.nanosecond = static_cast<std::int32_t>(std::llround((second - whole) * 1.0e9));
  if (!time::isValidUtc(out)) {
    return fail(error, "epoch_utc '" + text + "' is not a valid calendar instant");
  }
  return true;
}

/// Whether @p name is a planet the ephemeris fixture can carry
/// (`world::kPlanetNames`) — the same set the Python schema's Literal admits.
bool isKnownPlanet(const std::string& name) {
  for (const char* planet : world::kPlanetNames) {
    if (name == planet) {
      return true;
    }
  }
  return false;
}

bool parseAtmosphere(const std::string& name, AtmosphereModel& out, std::string* error) {
  if (name == "exponential") {
    out = AtmosphereModel::kExponential;
    return true;
  }
  if (name == "nrlmsis") {
    out = AtmosphereModel::kNrlmsis;
    return true;
  }
  return fail(error, "unknown atmosphere model '" + name + "'");
}

bool parseMagnetic(const std::string& name, MagneticModel& out, std::string* error) {
  if (name == "none") {
    out = MagneticModel::kNone;
    return true;
  }
  if (name == "igrf") {
    out = MagneticModel::kIgrf;
    return true;
  }
  return fail(error, "unknown magnetic field model '" + name + "'");
}

/// Read one resolved hardware suite (`spacecraft.sensors` / `.actuators`).
///
/// The parameter map is copied verbatim: this layer deliberately knows no key
/// names, so a catalog entry can gain a parameter without touching the sim. What
/// it does enforce is that every value is a number and that the identifying
/// fields are present — a unit with no `kind` cannot be dispatched to a model,
/// and silently skipping it would fly a vehicle missing hardware the config asked
/// for.
bool readUnits(const json& parent, const char* key, const std::string& role,
               std::vector<UnitConfig>& out, std::string* error) {
  const auto node = parent.find(key);
  if (node == parent.end()) {
    return true;  // a vehicle with no units of this class is legal
  }
  if (!node->is_array()) {
    return fail(error, "spacecraft." + std::string(key) + " must be an array");
  }

  for (const json& entry : *node) {
    UnitConfig unit;
    if (!entry.is_object()) {
      return fail(error, role + " entry is not an object");
    }
    unit.name = entry.value("name", std::string{});
    unit.model_id = entry.value("model_id", std::string{});
    unit.kind = entry.value("kind", std::string{});
    if (unit.name.empty() || unit.model_id.empty() || unit.kind.empty()) {
      return fail(error, role + " entry needs a name, model_id, and kind");
    }

    const auto params = entry.find("params");
    if (params != entry.end() && !params->is_null()) {
      if (!params->is_object()) {
        return fail(error, role + " '" + unit.name + "' params must be an object");
      }
      for (const auto& [name, value] : params->items()) {
        if (!value.is_number()) {
          return fail(error, role + " '" + unit.name + "' param '" + name + "' is not a number");
        }
        unit.params[name] = value.get<double>();
      }
    }

    // Mounting is optional and emitted as JSON null when the config omitted it.
    const auto dcm = entry.find("mounting_dcm_row_major");
    if (dcm != entry.end() && !dcm->is_null()) {
      if (!dcm->is_array() || dcm->size() != 9) {
        return fail(error,
                    role + " '" + unit.name + "' mounting_dcm_row_major must be a 9-element array");
      }
      for (std::size_t i = 0; i < 9; ++i) {
        if (!(*dcm)[i].is_number()) {
          return fail(error, role + " '" + unit.name + "' mounting_dcm_row_major is non-numeric");
        }
        unit.mounting_dcm(static_cast<Eigen::Index>(i / 3), static_cast<Eigen::Index>(i % 3)) =
            (*dcm)[i].get<double>();
      }
      // A mounting that is not a rotation would silently scale or mirror every
      // quantity passing through it, which reads downstream as a sensor error.
      const Eigen::Matrix3d residual =
          unit.mounting_dcm.transpose() * unit.mounting_dcm - Eigen::Matrix3d::Identity();
      if (residual.cwiseAbs().maxCoeff() > 1.0e-9 || unit.mounting_dcm.determinant() < 0.0) {
        return fail(error,
                    role + " '" + unit.name + "' mounting_dcm_row_major is not a proper rotation");
      }
    }

    // Spin axis is optional and emitted null when unset. Need not be unit — the
    // assembly normalises it — but a zero vector is a wheel with no axis.
    const auto axis = entry.find("spin_axis");
    if (axis != entry.end() && !axis->is_null()) {
      if (!axis->is_array() || axis->size() != 3) {
        return fail(error, role + " '" + unit.name + "' spin_axis must be a 3-element array");
      }
      for (std::size_t i = 0; i < 3; ++i) {
        if (!(*axis)[i].is_number()) {
          return fail(error, role + " '" + unit.name + "' spin_axis is non-numeric");
        }
        unit.spin_axis(static_cast<Eigen::Index>(i)) = (*axis)[i].get<double>();
      }
      if (unit.spin_axis.norm() < 1.0e-9) {
        return fail(error, role + " '" + unit.name + "' spin_axis is a zero vector");
      }
    }

    // Mounting position is optional and emitted null when unset; the body origin
    // is the default, which for the near-field coupling is the worst case
    // (co-located rod and magnetometer) rather than a flattering one.
    const auto position = entry.find("mounting_position_m");
    if (position != entry.end() && !position->is_null()) {
      if (!position->is_array() || position->size() != 3) {
        return fail(error,
                    role + " '" + unit.name + "' mounting_position_m must be a 3-element array");
      }
      for (std::size_t i = 0; i < 3; ++i) {
        if (!(*position)[i].is_number()) {
          return fail(error, role + " '" + unit.name + "' mounting_position_m is non-numeric");
        }
        unit.mounting_position_m(static_cast<Eigen::Index>(i)) = (*position)[i].get<double>();
      }
      if (!unit.mounting_position_m.allFinite()) {
        return fail(error, role + " '" + unit.name + "' mounting_position_m is not finite");
      }
    }

    // Per-unit noise override is optional and emitted null when unset.
    const auto noise = entry.find("noise_enabled");
    if (noise != entry.end() && !noise->is_null()) {
      if (!noise->is_boolean()) {
        return fail(error, role + " '" + unit.name + "' noise_enabled must be a boolean");
      }
      unit.noise_enabled = noise->get<bool>();
    }
    out.push_back(std::move(unit));
  }
  return true;
}

bool readSpacecraft(const json& root, SpacecraftConfig& out, std::string* error) {
  const json* node = require(root, "spacecraft", "", error);
  if (node == nullptr) {
    return false;
  }
  out.name = node->value("name", std::string{});

  if (!readNumber(*node, "mass_kg", "spacecraft", out.mass_kg, error)) {
    return false;
  }
  if (!(out.mass_kg > 0.0)) {
    return fail(error, "spacecraft.mass_kg must be positive");
  }

  const json* inertia = require(*node, "inertia_kgm2", "spacecraft", error);
  if (inertia == nullptr) {
    return false;
  }
  double ixx = 0.0;
  double iyy = 0.0;
  double izz = 0.0;
  if (!readNumber(*inertia, "ixx", "spacecraft.inertia_kgm2", ixx, error) ||
      !readNumber(*inertia, "iyy", "spacecraft.inertia_kgm2", iyy, error) ||
      !readNumber(*inertia, "izz", "spacecraft.inertia_kgm2", izz, error)) {
    return false;
  }
  const double ixy = inertia->value("ixy", 0.0);
  const double ixz = inertia->value("ixz", 0.0);
  const double iyz = inertia->value("iyz", 0.0);
  out.inertia_kgm2 << ixx, ixy, ixz, ixy, iyy, iyz, ixz, iyz, izz;

  // The plant inverts this every construction, and Euler's equation is
  // meaningless without it. A non-positive-definite tensor is a config error,
  // not a runtime surprise to discover as a NaN twenty seconds into a run.
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(out.inertia_kgm2);
  if (eigensolver.info() != Eigen::Success || eigensolver.eigenvalues().minCoeff() <= 0.0) {
    return fail(error, "spacecraft.inertia_kgm2 is not positive definite");
  }
  // Physical realizability: no principal moment may exceed the sum of the other
  // two. A tensor violating this describes no rigid body, and its rotational
  // motion would look plausible while being unphysical.
  const Eigen::Vector3d principal = eigensolver.eigenvalues();
  for (int i = 0; i < 3; ++i) {
    if (principal[i] > principal.sum() - principal[i] + 1.0e-12) {
      return fail(error,
                  "spacecraft.inertia_kgm2 violates the triangle inequality "
                  "(a principal moment exceeds the sum of the other two)");
    }
  }

  out.drag_area_m2 = node->value("drag_area_m2", 0.0);
  out.drag_cd = node->value("drag_cd", 2.2);
  out.srp_area_m2 = node->value("srp_area_m2", 0.0);
  out.srp_cr = node->value("srp_cr", 1.3);
  // Mirror the schema's positivity constraints (§19.1) so a hand-edited artifact
  // cannot flip the sign of a force: negative areas invert drag/SRP direction,
  // and a non-positive coefficient is physically meaningless.
  if (out.drag_area_m2 < 0.0 || out.srp_area_m2 < 0.0) {
    return fail(error, "spacecraft drag/srp reference areas must be non-negative");
  }
  if (!(out.drag_cd > 0.0) || !(out.srp_cr > 0.0)) {
    return fail(error, "spacecraft drag_cd and srp_cr must be positive");
  }

  Eigen::Vector3d com = Eigen::Vector3d::Zero();
  Eigen::Vector3d cp_aero = Eigen::Vector3d::Zero();
  Eigen::Vector3d cp_srp = Eigen::Vector3d::Zero();
  Eigen::Vector3d dipole = Eigen::Vector3d::Zero();
  if (node->contains("com_m") && !readVec3(*node, "com_m", "spacecraft", com, error)) {
    return false;
  }
  if (node->contains("cp_offset_aero_m") &&
      !readVec3(*node, "cp_offset_aero_m", "spacecraft", cp_aero, error)) {
    return false;
  }
  if (node->contains("cp_offset_srp_m") &&
      !readVec3(*node, "cp_offset_srp_m", "spacecraft", cp_srp, error)) {
    return false;
  }
  if (node->contains("residual_dipole_am2") &&
      !readVec3(*node, "residual_dipole_am2", "spacecraft", dipole, error)) {
    return false;
  }
  out.com_m = math::Vec3<math::frames::Body>(com);
  out.cp_offset_aero_m = math::Vec3<math::frames::Body>(cp_aero);
  out.cp_offset_srp_m = math::Vec3<math::frames::Body>(cp_srp);
  out.residual_dipole_am2 = math::Vec3<math::frames::Body>(dipole);

  return readUnits(*node, "sensors", "sensor", out.sensors, error) &&
         readUnits(*node, "actuators", "actuator", out.actuators, error);
}

bool readEnvironment(const json& root, EnvironmentConfig& out, std::string* error) {
  const json* node = require(root, "environment", "", error);
  if (node == nullptr) {
    return false;
  }
  // Negative = no gravity (free-drift baseline), 0 = point mass, >0 = the
  // spherical-harmonic field to that degree. See SimRunner::build.
  out.gravity_degree = node->value("gravity_degree", 8);
  out.gravity_order = node->value("gravity_order", -1);
  if (out.gravity_order > out.gravity_degree) {
    return fail(error, "environment.gravity_order exceeds gravity_degree");
  }
  // Emitted in km (the unit a config is written in); held in metres like every
  // other length on this side of the boundary.
  out.occultation_atmosphere_m =
      node->value("occultation_atmosphere_km", sensors::kDefaultAtmosphereHeight / 1000.0) * 1000.0;
  if (!(out.occultation_atmosphere_m >= 0.0)) {
    return fail(error, "environment.occultation_atmosphere_km must be non-negative");
  }
  // Optional; the compiler emits null when unset, which is not a string.
  const auto jamming = node->find("gnss_jamming_kml");
  if (jamming != node->end() && jamming->is_string()) {
    out.gnss_jamming_kml = jamming->get<std::string>();
  }
  out.gnss_jamming_enabled = node->value("gnss_jamming_enabled", true);
  out.sensor_noise_enabled = node->value("sensor_noise_enabled", true);
  out.gnss_noise_enabled = node->value("gnss_noise_enabled", true);

  const auto faults = node->find("gnss_fault_events");
  if (faults != node->end()) {
    if (!faults->is_array()) {
      return fail(error, "environment.gnss_fault_events must be an array");
    }
    for (const json& ev : *faults) {
      GnssFaultEvent fe;
      fe.unit = ev.value("unit", std::string{});
      if (fe.unit.empty()) {
        return fail(error, "gnss_fault_events entry has no unit");
      }
      const std::string type = ev.value("type", std::string{});
      if (type == "outage") {
        fe.type = GnssFaultEvent::Type::kOutage;
      } else if (type == "spoof") {
        fe.type = GnssFaultEvent::Type::kSpoof;
      } else if (type == "clock_jump") {
        fe.type = GnssFaultEvent::Type::kClockJump;
      } else {
        return fail(error, "gnss_fault_events entry '" + fe.unit + "' has unknown type '" + type +
                               "' (outage|spoof|clock_jump)");
      }
      fe.start_s = ev.value("start_s", 0.0);
      fe.stop_s = ev.value("stop_s", 0.0);
      if (!(fe.stop_s > fe.start_s)) {
        return fail(error, "gnss_fault_events entry '" + fe.unit + "' has stop_s <= start_s");
      }
      const auto off = ev.find("spoof_offset_ecef_m");
      if (off != ev.end() && off->is_array() && off->size() == 3) {
        fe.spoof_offset_ecef_m = Eigen::Vector3d((*off)[0].get<double>(), (*off)[1].get<double>(),
                                                 (*off)[2].get<double>());
      }
      fe.clock_jump_s = ev.value("clock_jump_s", 0.0);
      out.gnss_fault_events.push_back(fe);
    }
  }

  out.drag_enabled = node->value("drag_enabled", true);
  out.srp_enabled = node->value("srp_enabled", true);
  out.eclipse_enabled = node->value("eclipse_enabled", true);
  // §5.3 disturbance torques. Default on: they are physically present, and a
  // scenario that omits the key wants the real environment.
  out.gravity_gradient_torque_enabled = node->value("gravity_gradient_torque_enabled", true);
  out.aero_torque_enabled = node->value("aero_torque_enabled", true);
  out.srp_torque_enabled = node->value("srp_torque_enabled", true);
  out.residual_dipole_torque_enabled = node->value("residual_dipole_torque_enabled", true);

  const auto bodies = node->find("third_bodies");
  if (bodies != node->end()) {
    if (!bodies->is_array()) {
      return fail(error, "environment.third_bodies must be an array");
    }
    for (const json& body : *bodies) {
      std::string name = body.get<std::string>();
      // Case-robust, mirroring the schema: "Jupiter"/"SUN" are obviously
      // intended, so normalise to the lowercase canonical form before matching.
      for (char& c : name) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      }
      if (name == "sun") {
        out.sun_third_body = true;
      } else if (name == "moon") {
        out.moon_third_body = true;
      } else if (isKnownPlanet(name)) {
        out.planet_third_bodies.push_back(name);
      } else {
        return fail(error, "unsupported third body '" + name + "'");
      }
    }
  }

  return parseAtmosphere(node->value("atmosphere", std::string{"exponential"}), out.atmosphere,
                         error) &&
         parseMagnetic(node->value("magnetic_field", std::string{"igrf"}), out.magnetic_field,
                       error);
}

bool readPropagation(const json& root, PropagationConfig& out, std::string* error) {
  const json* node = require(root, "propagation", "", error);
  if (node == nullptr) {
    return false;
  }
  if (!readNumber(*node, "duration_s", "propagation", out.duration_s, error) ||
      !readNumber(*node, "output_step_s", "propagation", out.output_step_s, error)) {
    return false;
  }
  if (!(out.duration_s > 0.0)) {
    return fail(error, "propagation.duration_s must be positive");
  }
  if (!(out.output_step_s > 0.0)) {
    return fail(error, "propagation.output_step_s must be positive");
  }
  out.abs_tol = node->value("abs_tol", 1.0e-12);
  out.rel_tol = node->value("rel_tol", 1.0e-12);
  out.max_step_s = node->value("max_step_s", 60.0);
  if (!(out.max_step_s > 0.0)) {
    return fail(error, "propagation.max_step_s must be positive");
  }
  // Mirror the schema (§19.1): a zero tolerance drives the RK89 step controller
  // to a zero step, which stalls the propagation rather than erroring.
  if (!(out.abs_tol > 0.0) || !(out.rel_tol > 0.0)) {
    return fail(error, "propagation.abs_tol and rel_tol must be positive");
  }
  out.fsw_rate_hz = node->value("fsw_rate_hz", 10.0);
  if (!(out.fsw_rate_hz > 0.0)) {
    return fail(error, "propagation.fsw_rate_hz must be positive");
  }
  return true;
}

bool readInitialState(const json& root, const time::LeapSecondTable& leap, state::TruthState& out,
                      std::string* error) {
  const json* node = require(root, "initial_state", "", error);
  if (node == nullptr) {
    return false;
  }

  const json* epoch_text = require(root, "epoch_utc", "", error);
  if (epoch_text == nullptr) {
    return false;
  }
  time::UtcDateTime utc;
  if (!parseUtc(epoch_text->get<std::string>(), utc, error)) {
    return false;
  }
  out.epoch = time::taiFromUtc(utc, leap);

  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d rate = Eigen::Vector3d::Zero();
  if (!readVec3(*node, "position_m", "initial_state", position, error) ||
      !readVec3(*node, "velocity_m_s", "initial_state", velocity, error)) {
    return false;
  }
  if (node->contains("body_rate_rad_s") &&
      !readVec3(*node, "body_rate_rad_s", "initial_state", rate, error)) {
    return false;
  }
  if (!(position.norm() > 0.0)) {
    return fail(error, "initial_state.position_m must be non-zero");
  }

  const json* quaternion = require(*node, "attitude_quaternion", "initial_state", error);
  if (quaternion == nullptr) {
    return false;
  }
  if (!quaternion->is_array() || quaternion->size() != 4) {
    return fail(error, "initial_state.attitude_quaternion must be a 4-element array");
  }
  Eigen::Vector4d coefficients;
  for (std::size_t i = 0; i < 4; ++i) {
    coefficients[static_cast<Eigen::Index>(i)] = (*quaternion)[i].get<double>();
  }
  // Scalar-first (q0, q1, q2, q3), the repo-wide JPL convention. A quaternion
  // off the unit manifold is rejected rather than silently renormalised: it
  // usually means the field was written in a different convention, and
  // renormalising would hide that while producing a wrong attitude.
  if (std::abs(coefficients.norm() - 1.0) > 1.0e-9) {
    return fail(error, "initial_state.attitude_quaternion is not a unit quaternion");
  }

  out.position = math::Vec3<math::frames::ECI>(position);
  out.velocity = math::Vec3<math::frames::ECI>(velocity);
  out.attitude = math::Quat<math::frames::Body, math::frames::ECI>(math::Quaternion(coefficients));
  out.body_rate = math::Vec3<math::frames::Body>(rate);
  return true;
}

/// Cross-check between the two sections: a §5.3 torque that is switched on needs
/// its lever arm (or dipole) to have been *written down*. These are no-default
/// vehicle-config fields — an absent one is a config error, not a zero, because a
/// silently-zero lever arm produces a run with no aero torque that looks exactly
/// like a run with a well-balanced vehicle. The schema makes them required on the
/// YAML path; this catches a hand-edited artifact.
bool checkTorqueFields(const json& root, const EnvironmentConfig& env, std::string* error) {
  const auto sc = root.find("spacecraft");
  if (sc == root.end()) {
    return true;  // already reported by readSpacecraft
  }

  struct Requirement {
    bool enabled;
    const char* key;
    const char* torque;
  };

  const Requirement required[] = {
      {env.drag_enabled && env.aero_torque_enabled, "cp_offset_aero_m", "aero_torque_enabled"},
      {env.srp_enabled && env.srp_torque_enabled, "cp_offset_srp_m", "srp_torque_enabled"},
      {env.magnetic_field == MagneticModel::kIgrf && env.residual_dipole_torque_enabled,
       "residual_dipole_am2", "residual_dipole_torque_enabled"},
  };
  for (const Requirement& r : required) {
    if (r.enabled && !sc->contains(r.key)) {
      return fail(error, std::string("environment.") + r.torque + " is set but spacecraft." +
                             r.key + " is missing");
    }
  }
  return true;
}

}  // namespace

bool loadSimConfig(const std::string& path, const time::LeapSecondTable& leap, SimConfig& out,
                   std::string* error) {
  std::ifstream file(path);
  if (!file.good()) {
    return fail(error, "cannot open sim config: " + path);
  }

  json root;
  try {
    file >> root;
  } catch (const json::parse_error& e) {
    // nlohmann signals malformed JSON by exception; the sim's own interface is
    // exceptions-free, so it stops here and becomes an error string.
    return fail(error, "malformed JSON in " + path + ": " + e.what());
  }
  if (!root.is_object()) {
    return fail(error, "sim config root is not an object: " + path);
  }

  out = SimConfig{};
  out.scenario_name = root.value("scenario_name", std::string{});
  // Absent seed = 0, which is a valid run: the point is that the seed is an
  // input, not that it is non-zero. A missing one must not fall back to entropy.
  out.seed = root.value("seed", static_cast<std::uint64_t>(0));
  const auto provenance = root.find("provenance");
  if (provenance != root.end()) {
    out.config_hash = provenance->value("config_hash", std::string{});
  }

  if (!readSpacecraft(root, out.spacecraft, error) ||
      !readEnvironment(root, out.environment, error) ||
      !readPropagation(root, out.propagation, error) ||
      !readInitialState(root, leap, out.initial_state, error)) {
    return false;
  }
  return checkTorqueFields(root, out.environment, error);
}

}  // namespace polaris::sim::scenario
