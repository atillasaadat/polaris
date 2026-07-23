#include "scenario/vehicle.hpp"

#include <set>
#include <utility>

namespace polaris::sim::scenario {
namespace {

bool fail(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
  return false;
}

/// FNV-1a over the instance name — the unit's noise-stream id.
///
/// Name-derived rather than index-derived on purpose: the §182 determinism rule
/// is that adding a noise source must not perturb the ones already there, and an
/// index shifts the moment a unit is inserted ahead of another. FNV-1a is used
/// only to spread names across the id space; `random::streamRng` does the actual
/// mixing, so the hash needs no statistical strength of its own.
std::uint64_t streamIdFor(const std::string& name) {
  std::uint64_t hash = 14695981039346656037ULL;
  for (const char c : name) {
    hash ^= static_cast<std::uint64_t>(static_cast<unsigned char>(c));
    hash *= 1099511628211ULL;
  }
  return hash;
}

}  // namespace

bool buildVehicle(const SpacecraftConfig& spacecraft, std::uint64_t seed, Vehicle& out,
                  std::string* error) {
  out = Vehicle{};

  std::set<std::string> names;
  for (const std::vector<UnitConfig>* suite : {&spacecraft.sensors, &spacecraft.actuators}) {
    for (const UnitConfig& unit : *suite) {
      if (!names.insert(unit.name).second) {
        return fail(error, "duplicate hardware unit name '" + unit.name +
                               "' — unit names key the per-unit random streams and "
                               "must be unique across the vehicle");
      }
      // Every modelled device takes all of its physics from these parameters, so
      // an empty map is not a degenerate unit, it is a resolution that went
      // wrong: the model would build as an ideal, noiseless, unlimited device.
      if (unit.params.empty()) {
        return fail(error, "hardware unit '" + unit.name + "' (model_id '" + unit.model_id +
                               "') resolved to no parameters");
      }

      const std::uint64_t stream = streamIdFor(unit.name);
      if (unit.kind == "imu") {
        out.imus.push_back({unit.name, unit.model_id, unit.mounting_dcm,
                            sensors::Imu(sensors::ImuSpec::fromParams(unit.params), seed, stream)});
      } else if (unit.kind == "star_tracker") {
        const auto spec = sensors::StarTrackerSpec::fromParams(unit.params);
        // With neither a field of view nor a quoted Earth exclusion angle the
        // Earth keep-out collapses to zero and the tracker solves happily while
        // staring at the ground. A silently over-optimistic sensor is worse than
        // a configuration error. Either key satisfies this — vendors quote an
        // exclusion angle, and it is what dominates when both are present.
        if (!(spec.keep_out.earth_rad > 0.0)) {
          return fail(error, "star tracker '" + unit.name +
                                 "' has neither fov_deg nor earth_exclusion_deg — it would "
                                 "report valid solutions while pointed at the Earth");
        }
        out.star_trackers.push_back({unit.name, unit.model_id, unit.mounting_dcm,
                                     sensors::StarTracker(spec, unit.mounting_dcm, seed, stream)});
      } else if (unit.kind == "reaction_wheel") {
        const auto spec = actuators::ReactionWheelSpec::fromParams(unit.params);
        if (!(spec.inertia() > 0.0)) {
          return fail(error, "reaction wheel '" + unit.name +
                                 "' has no rotor inertia — set rotor_inertia_kg_m2, or "
                                 "max_momentum_nms with max_speed_rpm");
        }
        out.wheels.push_back(
            {unit.name, unit.model_id, unit.mounting_dcm, actuators::ReactionWheel(spec)});
      } else if (unit.kind == "magnetorquer") {
        const auto spec = actuators::MagnetorquerSpec::fromParams(unit.params);
        if (!(spec.max_dipole_am2 > 0.0)) {
          return fail(error, "magnetorquer '" + unit.name + "' has no max_dipole_am2");
        }
        out.magnetorquers.push_back(
            {unit.name, unit.model_id, unit.mounting_dcm, actuators::Magnetorquer(spec)});
      } else {
        out.unmodelled.push_back(unit.name + ":" + unit.kind);
      }
    }
  }
  return true;
}

}  // namespace polaris::sim::scenario
